//! Cutting a frame into segments so `backdrop-filter` has something to filter.
//!
//! A filtered layer's input is the pixels behind it, and a scene being built
//! has no pixels. The only way to produce them is to stop, render what has been
//! drawn, filter that, and carry on with the result available as an image. That
//! is what a segment is: the run of drawing commands between two such stops.
//!
//! # The shape of a frame
//!
//! ```text
//!   segment 0  ->  snapshot A  ->  blur regions of A  ->  segment 1 -> ...
//! ```
//!
//! Each segment after the first begins by drawing the previous snapshot across
//! the whole frame, because [`vello::Renderer::render_to_texture`] clears its
//! target: there is no way to render *onto* what is already there, so the way
//! back is to draw it. The blurred regions are then drawn over that snapshot,
//! clipped to their elements, and the segment's own content on top.
//!
//! # What a pass costs, and why the batching matters so much
//!
//! Per boundary: one full-frame vello render, one copy of that texture into
//! vello's image atlas (`register_texture` documents the copy, and it is per
//! frame, not once), one blur, and one full-frame image draw in the next
//! segment. The scene's own commands are split across segments rather than
//! repeated, so the *encoding* does not multiply, but the full-frame
//! rasterisation and the two full-frame bandwidth passes do.
//!
//! That is the number that makes `anyrender::BackdropPlanner`'s batching worth
//! having, and it is also why a layout whose panels sit closer together than
//! their blur reaches is expensive in a way no renderer can fix.
//!
//! # The layer stack has to be replayed
//!
//! A filtered element is nested inside whatever clips its ancestors pushed, and
//! cutting the scene there leaves those layers open in a segment that is about
//! to be handed to the rasteriser. So the painter keeps a shadow copy of the
//! open stack, pops it into the outgoing segment and pushes it again into the
//! incoming one. Without that, every glass panel below a scrolling container
//! would paint unclipped over its own scrollbar.

use crate::blur::BlurPipeline;
use anyrender::ResourceId;
use kurbo::{Affine, BezPath};
use peniko::{BlendMode, ImageData};
use rustc_hash::FxHashMap;
use vello::{Renderer as VelloRenderer, Scene as VelloScene};

/// One entry in the painter's shadow copy of the open layer stack.
///
/// Enough to push the layer again, and nothing else: a segment boundary has to
/// reproduce the clipping state, not the drawing that happened inside it.
#[derive(Clone)]
pub(crate) struct OpenLayer {
    /// `None` for a plain clip layer, which is the overwhelming majority.
    pub blend: Option<(BlendMode, f32)>,
    pub transform: Affine,
    pub clip: BezPath,
}

/// A blur to run between two segments.
pub(crate) struct BlurJob {
    /// Pool slot holding the result, and the id the next segment draws it by.
    pub slot: usize,
    /// Where in the snapshot the region starts.
    pub origin: [i32; 2],
    pub size: [u32; 2],
    pub sigma: f32,
}

/// What follows one segment.
///
/// Named apart from `anyrender::Boundary`, which is the planner's answer to
/// "does this op need a new segment". This is the segment itself.
pub(crate) struct SegmentBoundary {
    /// Snapshot texture the preceding segment renders into.
    pub snapshot: usize,
    pub jobs: Vec<BlurJob>,
}

/// A frame cut into segments, ready to execute.
#[derive(Default)]
pub(crate) struct FrameSegments {
    /// Completed segments. The one still being built lives in the painter.
    pub scenes: Vec<VelloScene>,
    /// `boundaries[i]` follows `scenes[i]`.
    pub boundaries: Vec<SegmentBoundary>,
}

/// A texture the pool owns, and the handle vello knows it by.
struct Slot {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    image: ImageData,
    id: ResourceId,
    size: (u32, u32),
}

/// Textures reused across frames for snapshots and blur results.
///
/// Allocating these per frame would mean a texture creation and a vello
/// registration per panel per frame, and registration is not free: it inserts
/// into vello's override map and forces an atlas copy. They are keyed by
/// position in the frame rather than by element, which is enough because the
/// same page produces the same sequence of boundaries every frame.
#[derive(Default)]
pub struct BackdropPool {
    snapshots: Vec<Slot>,
    scratch: Vec<Slot>,
    blurred: Vec<Slot>,
    pipeline: Option<BlurPipeline>,
}

/// Everything a snapshot or blur slot hands back to the scene being built.
pub(crate) struct SlotIds {
    pub snapshot: ResourceId,
    pub blurred: ResourceId,
}

impl BackdropPool {
    /// Reserve the textures one backdrop op needs, and give back the ids the
    /// scene will draw them by.
    ///
    /// `boundary` and `job` index the pool: the same page lays out the same way
    /// every frame, so slot 3 is the same panel frame after frame and nothing is
    /// reallocated.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reserve(
        &mut self,
        device: &wgpu::Device,
        renderer: &mut VelloRenderer,
        handles: &mut FxHashMap<ResourceId, ImageData>,
        boundary: usize,
        job: usize,
        frame: (u32, u32),
        region: (u32, u32),
    ) -> SlotIds {
        let snapshot = ensure(
            &mut self.snapshots,
            boundary,
            device,
            renderer,
            handles,
            frame,
            "backdrop snapshot",
        );
        // Not registered with vello: nothing ever draws the scratch, it exists
        // only between the two halves of the separable blur, so it needs no id.
        ensure_unregistered(&mut self.scratch, job, device, region);
        let blurred = ensure(
            &mut self.blurred,
            job,
            device,
            renderer,
            handles,
            region,
            "backdrop blurred",
        );
        SlotIds { snapshot, blurred }
    }

    /// Run the blurs for one boundary, out of the snapshot it follows.
    pub(crate) fn encode_boundary(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        boundary: &SegmentBoundary,
    ) {
        let pipeline = self
            .pipeline
            .get_or_insert_with(|| BlurPipeline::new(device));
        let Some(snapshot) = self.snapshots.get(boundary.snapshot) else {
            return;
        };
        for job in &boundary.jobs {
            let (Some(scratch), Some(blurred)) =
                (self.scratch.get(job.slot), self.blurred.get(job.slot))
            else {
                continue;
            };
            pipeline.encode(
                device,
                encoder,
                &snapshot.view,
                &scratch.view,
                &blurred.view,
                job.origin,
                job.size,
                job.sigma,
            );
        }
    }

    pub(crate) fn snapshot_view(&self, index: usize) -> Option<&wgpu::TextureView> {
        self.snapshots.get(index).map(|slot| &slot.view)
    }

    /// Mark only the textures one boundary actually wrote.
    ///
    /// A dirty override image is copied into vello's atlas at the start of the
    /// next render, a full frame each. This used to mark every slot the pool
    /// owns, before every segment, so a frame cut into N boundaries paid N+1
    /// copies of *all* of them rather than of the one or two that changed. On
    /// a page whose panels each force their own boundary that is the
    /// difference between a handful of full-frame copies and a few dozen.
    ///
    /// Everything a segment reads was written by the boundary immediately
    /// before it: that boundary's snapshot, and the blur results it produced.
    /// Nothing else can have changed since the last render, so nothing else
    /// needs recopying.
    pub(crate) fn mark_boundary_dirty(
        &self,
        renderer: &mut VelloRenderer,
        boundary: &SegmentBoundary,
    ) {
        if let Some(slot) = self.snapshots.get(boundary.snapshot) {
            renderer.mark_override_image_dirty(&slot.image);
        }
        for job in &boundary.jobs {
            if let Some(slot) = self.blurred.get(job.slot) {
                renderer.mark_override_image_dirty(&slot.image);
            }
        }
    }

    /// Release every texture and its vello registration.
    ///
    /// Called on suspend, because the registrations name a device that is about
    /// to go away.
    pub fn clear(
        &mut self,
        renderer: &mut VelloRenderer,
        handles: &mut FxHashMap<ResourceId, ImageData>,
    ) {
        for slot in self.snapshots.drain(..).chain(self.blurred.drain(..)) {
            handles.remove(&slot.id);
            renderer.unregister_texture(slot.image);
        }
        self.scratch.clear();
    }
}

/// Render a segmented frame: snapshot, blur, snapshot, blur, then the surface.
///
/// One submission per boundary rather than one for the whole frame. The blur
/// reads the texture the render before it wrote, and the render after it reads
/// what the blur wrote, so the ordering is a hard dependency chain; vello owns
/// its own submissions and gives no way to interleave a compute pass into one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute(
    pool: &mut BackdropPool,
    renderer: &mut VelloRenderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    segments: &FrameSegments,
    final_scene: &VelloScene,
    final_target: &wgpu::TextureView,
    params: &vello::RenderParams,
) -> Result<(), vello::Error> {
    /*
     * Mark before every render, not once per frame: the atlas copy happens at
     * the start of a render, so a texture written by the previous submission
     * is only picked up if it has been marked since.
     *
     * Only the *previous* boundary's textures, though. A segment reads what
     * the boundary immediately before it wrote and nothing else, so marking
     * every slot the pool owns re-copied a full-frame texture per slot per
     * segment for no benefit. On a page where each panel forces its own
     * boundary that was the dominant cost in a 450ms frame.
     *
     * The first segment reads nothing, so it marks nothing.
     */
    let mut previous: Option<&SegmentBoundary> = None;

    for (scene, boundary) in segments.scenes.iter().zip(&segments.boundaries) {
        let Some(target) = pool.snapshot_view(boundary.snapshot).cloned() else {
            continue;
        };
        if let Some(previous) = previous {
            pool.mark_boundary_dirty(renderer, previous);
        }
        renderer.render_to_texture(device, queue, scene, &target, params)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("anyrender_vello backdrop"),
        });
        pool.encode_boundary(device, &mut encoder, boundary);
        queue.submit([encoder.finish()]);
        previous = Some(boundary);
    }

    if let Some(previous) = previous {
        pool.mark_boundary_dirty(renderer, previous);
    }
    renderer.render_to_texture(device, queue, final_scene, final_target, params)
}

/// Grow `slots` to cover `index` and make sure that slot is `size`.
fn ensure(
    slots: &mut Vec<Slot>,
    index: usize,
    device: &wgpu::Device,
    renderer: &mut VelloRenderer,
    handles: &mut FxHashMap<ResourceId, ImageData>,
    size: (u32, u32),
    label: &str,
) -> ResourceId {
    if let Some(slot) = slots.get(index) {
        if slot.size == size {
            return slot.id;
        }
        let old = slots.swap_remove(index);
        // `swap_remove` on the last element is a plain pop, and on any other it
        // moves the tail into the hole. Either way the vacated index is filled
        // below, so the loop that grows the vector still terminates.
        handles.remove(&old.id);
        renderer.unregister_texture(old.image);
        slots.insert(index, new_slot(device, renderer, handles, size, label));
        return slots[index].id;
    }
    while slots.len() <= index {
        let slot = new_slot(device, renderer, handles, size, label);
        slots.push(slot);
    }
    slots[index].id
}

fn ensure_unregistered(
    slots: &mut Vec<Slot>,
    index: usize,
    device: &wgpu::Device,
    size: (u32, u32),
) {
    if let Some(slot) = slots.get(index) {
        if slot.size == size {
            return;
        }
        slots[index] = raw_slot(device, size, "backdrop blur scratch");
        return;
    }
    while slots.len() <= index {
        slots.push(raw_slot(device, size, "backdrop blur scratch"));
    }
}

fn new_slot(
    device: &wgpu::Device,
    renderer: &mut VelloRenderer,
    handles: &mut FxHashMap<ResourceId, ImageData>,
    size: (u32, u32),
    label: &str,
) -> Slot {
    let mut slot = raw_slot(device, size, label);
    let image = renderer.register_texture(slot.texture.clone());
    let id = ResourceId::new();
    handles.insert(id, image.clone());
    slot.image = image;
    slot.id = id;
    slot
}

fn raw_slot(device: &wgpu::Device, size: (u32, u32), label: &str) -> Slot {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0.max(1),
            height: size.1.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // `Rgba8Unorm` and `COPY_SRC` are both required by
        // `Renderer::register_texture`, which copies into vello's atlas.
        // `STORAGE_BINDING` is what lets the blur write into it and
        // `TEXTURE_BINDING` what lets the blur read it.
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    Slot {
        image: ImageData {
            data: peniko::Blob::new(std::sync::Arc::new(&[] as &[u8])),
            format: peniko::ImageFormat::Rgba8,
            alpha_type: peniko::ImageAlphaType::Alpha,
            width: size.0.max(1),
            height: size.1.max(1),
        },
        id: ResourceId::new(),
        size,
        texture,
        view,
    }
}
