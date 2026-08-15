//! A separable gaussian blur over a region of a rendered texture.
//!
//! This is the filter half of `backdrop-filter`. The other half - stopping the
//! scene, rendering what is behind, and drawing the result back - lives in
//! [`crate::backdrop`].
//!
//! # Why a compute shader and not vello
//!
//! vello 0.9 has no layer filter of any kind, and its one blur,
//! `draw_blurred_rounded_rect`, blurs a shape it generates rather than an image
//! it is given. There is nothing here to reuse.
//!
//! Compute rather than a fragment pass because the textures already have the
//! usages for it. vello renders into `Rgba8Unorm` with
//! `STORAGE_BINDING | TEXTURE_BINDING`, which is exactly a `textureLoad` source
//! and a `textureStore` destination, so no render pipeline, vertex buffer or
//! attachment setup is needed to read one and write the other.
//!
//! # Separable
//!
//! A 2D gaussian is the product of two 1D ones, so a radius-r blur is 2(2r+1)
//! taps instead of (2r+1)^2. At the sigma=12 this application asks for that is
//! 146 taps against 5329, and it is the difference between a blur that fits in
//! a frame and one that does not.
//!
//! # Alpha
//!
//! The pixels are *straight*, not premultiplied. vello's own output is straight
//! (`wgpu_context` premultiplies on the way to the surface, and
//! `Renderer::register_texture` documents its input as unpremultiplied), and
//! blurring straight alpha does bleed colour out of fully transparent texels.
//!
//! It is nonetheless right here, and not by luck: the input is a snapshot of
//! everything painted *behind* an element, over an opaque page background, so
//! alpha is 1 across the region and there are no transparent texels to bleed
//! from. A backdrop captured over a transparent window is the case where this
//! stops holding, and it is worth knowing that before turning window
//! transparency on.
//!
//! # What it does not do yet
//!
//! Full resolution, and one kernel tap per pixel of radius. Downsampling is the
//! next multiplier and it is a large one - sigma=12 at quarter resolution is
//! sigma=3 over a sixteenth of the texels - but it changes the sampling and is
//! worth landing on its own rather than folded into the pass that first makes
//! the picture correct.

use wgpu::util::DeviceExt as _;

/// Threads per workgroup, per axis. Matches `@workgroup_size` in the shader.
const WORKGROUP: u32 = 8;

/// How far the kernel reaches, in standard deviations.
///
/// Past three standard deviations under 0.3% of a gaussian's weight remains,
/// which cannot move an 8-bit channel by one level. It is also the number
/// `anyrender::Filter::expansion_rect` uses to size the region, and the two
/// have to agree: a kernel reaching further than the region that was captured
/// would sample pixels nobody rendered.
const REACH: f32 = 3.0;

/// The largest kernel radius, in texels.
///
/// A guard on the loop rather than a quality decision. CSS will accept
/// `blur(400px)`, and at full resolution that is an 2401-tap kernel per pixel
/// per axis, which is not a slow frame but a hung one.
const MAX_RADIUS: i32 = 128;

/// Uniform block for one pass, laid out to match the WGSL `Params` struct.
///
/// Serialised by hand rather than through `bytemuck`. The block is 32 bytes of
/// four-byte scalars in `vec2` pairs, so every member is naturally aligned and
/// there is no padding to get wrong; taking a dependency to express that would
/// be more machinery than the thing it describes.
#[derive(Clone, Copy)]
struct Params {
    /// Which way this pass walks: `[1, 0]` then `[0, 1]`.
    dir: [i32; 2],
    /// Where in the source texture the region starts.
    src_origin: [i32; 2],
    /// Size of the region, and so of the destination texture.
    size: [u32; 2],
    radius: i32,
    sigma: f32,
}

impl Params {
    fn to_bytes(self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        let words = [
            self.dir[0].to_ne_bytes(),
            self.dir[1].to_ne_bytes(),
            self.src_origin[0].to_ne_bytes(),
            self.src_origin[1].to_ne_bytes(),
            self.size[0].to_ne_bytes(),
            self.size[1].to_ne_bytes(),
            self.radius.to_ne_bytes(),
            self.sigma.to_ne_bytes(),
        ];
        for (slot, word) in bytes.chunks_exact_mut(4).zip(words) {
            slot.copy_from_slice(&word);
        }
        bytes
    }
}

pub(crate) struct BlurPipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl BlurPipeline {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("anyrender_vello backdrop blur"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("anyrender_vello backdrop blur"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("anyrender_vello backdrop blur"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("anyrender_vello backdrop blur"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("blur"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// Blur `region` of `source` into `destination`, via `scratch`.
    ///
    /// All three textures are `Rgba8Unorm`. `scratch` and `destination` are the
    /// size of the region; `source` is the whole frame, and `origin` says where
    /// in it the region sits.
    ///
    /// Two dispatches: horizontal out of the frame into the scratch, vertical
    /// out of the scratch into the destination. The second reads only the
    /// scratch, so it clamps at the region's edge rather than the frame's -
    /// wrong by up to the kernel radius there, and harmless, because the region
    /// was grown by exactly that radius before capture and nothing inside the
    /// element's own bounds is within reach of the edge.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::TextureView,
        scratch: &wgpu::TextureView,
        destination: &wgpu::TextureView,
        origin: [i32; 2],
        size: [u32; 2],
        sigma: f32,
    ) {
        if size[0] == 0 || size[1] == 0 {
            return;
        }
        let radius = kernel_radius(sigma);

        let passes = [
            (
                source,
                scratch,
                Params {
                    dir: [1, 0],
                    src_origin: origin,
                    size,
                    radius,
                    sigma,
                },
            ),
            (
                scratch,
                destination,
                Params {
                    dir: [0, 1],
                    src_origin: [0, 0],
                    size,
                    radius,
                    sigma,
                },
            ),
        ];

        for (from, to, params) in passes {
            let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("anyrender_vello backdrop blur params"),
                contents: &params.to_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("anyrender_vello backdrop blur"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(from),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(to),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: uniforms.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("anyrender_vello backdrop blur"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(size[0].div_ceil(WORKGROUP), size[1].div_ceil(WORKGROUP), 1);
        }
    }
}

/// How many texels each side of a pixel the kernel reads.
pub(crate) fn kernel_radius(sigma: f32) -> i32 {
    ((sigma * REACH).ceil() as i32).clamp(0, MAX_RADIUS)
}

const SHADER: &str = r#"
struct Params {
    dir: vec2<i32>,
    src_origin: vec2<i32>,
    size: vec2<u32>,
    radius: i32,
    sigma: f32,
};

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size.x || gid.y >= params.size.y) {
        return;
    }

    let base = vec2<i32>(i32(gid.x), i32(gid.y)) + params.src_origin;
    let bounds = vec2<i32>(textureDimensions(src)) - vec2<i32>(1, 1);

    // A sigma of zero is a filter that does nothing, and the weight below would
    // divide by it. Copy through instead of guarding every tap.
    if (params.sigma <= 0.0 || params.radius == 0) {
        let p = clamp(base, vec2<i32>(0, 0), bounds);
        textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), textureLoad(src, p, 0));
        return;
    }

    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var total = 0.0;
    let denom = 2.0 * params.sigma * params.sigma;
    for (var i = -params.radius; i <= params.radius; i = i + 1) {
        let w = exp(-f32(i * i) / denom);
        // Clamped rather than wrapped or zeroed. Zeroing darkens every edge of
        // the region toward transparent, which reads as a shadow around each
        // panel; clamping repeats the edge texel, which is what every browser
        // does for `edge-mode: duplicate` and is invisible.
        let p = clamp(base + params.dir * i, vec2<i32>(0, 0), bounds);
        sum = sum + textureLoad(src, p, 0) * w;
        total = total + w;
    }

    textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), sum / total);
}
"#;
