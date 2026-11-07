use anyrender::{ImageRenderer, RenderContext, ResourceId};
use peniko::ImageData;
use rustc_hash::FxHashMap;
use vello::{Renderer as VelloRenderer, RendererOptions, Scene as VelloScene};
use wgpu::TextureUsages;
use wgpu_context::{BufferRenderer, BufferRendererConfig, WGPUContext};

use crate::backdrop::BackdropPool;
use crate::scene::BackdropState;
use crate::{DEFAULT_THREADS, VelloScenePainter};
use anyrender::BackdropPlanner;

pub struct VelloImageRenderer {
    buffer_renderer: BufferRenderer,
    vello_renderer: VelloRenderer,
    scene: VelloScene,
    texture_handles: FxHashMap<ResourceId, ImageData>,
    backdrop: BackdropPool,
}

impl RenderContext for VelloImageRenderer {}
impl ImageRenderer for VelloImageRenderer {
    type ScenePainter<'a>
        = VelloScenePainter<'a, 'a>
    where
        Self: 'a;

    fn new(width: u32, height: u32) -> Self {
        // Create WGPUContext
        let mut context = WGPUContext::new();

        // Create wgpu_context::BufferRenderer
        let buffer_renderer =
            pollster::block_on(context.create_buffer_renderer(BufferRendererConfig {
                width,
                height,
                usage: TextureUsages::STORAGE_BINDING,
            }))
            .expect("No compatible device found");

        // Create vello::Renderer
        let vello_renderer = VelloRenderer::new(
            buffer_renderer.device(),
            RendererOptions {
                use_cpu: false,
                num_init_threads: DEFAULT_THREADS,
                antialiasing_support: vello::AaSupport::area_only(),
                pipeline_cache: None,
            },
        )
        .expect("Got non-Send/Sync error from creating renderer");

        Self {
            buffer_renderer,
            vello_renderer,
            scene: VelloScene::new(),
            texture_handles: FxHashMap::default(),
            backdrop: BackdropPool::default(),
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.buffer_renderer.resize(width, height);
    }

    fn reset(&mut self) {
        self.scene.reset();
    }

    fn render_to_vec<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        cpu_buffer: &mut Vec<u8>,
    ) {
        let size = self.buffer_renderer.size();
        cpu_buffer.resize((size.width * size.height * 4) as usize, 0);
        self.render(draw_fn, cpu_buffer);
    }

    fn render<F: FnOnce(&mut Self::ScenePainter<'_>)>(
        &mut self,
        draw_fn: F,
        cpu_buffer: &mut [u8],
    ) {
        // A previous frame may have returned early after recording into the
        // retained scene. Start every attempt from a clean scene.
        self.scene.reset();
        let size = self.buffer_renderer.size();
        let mut painter = VelloScenePainter {
            inner: &mut self.scene,
            renderer: Some(&mut self.vello_renderer),
            device_handle: None,
            texture_handles: Some(&mut self.texture_handles),
            // Backdrops work here as well as in a window. That is deliberate:
            // it is the only way to assert in pixels that a backdrop actually
            // blurred, without a surface, a window or an event loop.
            backdrop: Some(BackdropState {
                device: self.buffer_renderer.device(),
                pool: &mut self.backdrop,
                frame: (size.width, size.height),
                planner: BackdropPlanner::new(),
                stack: Vec::new(),
                segments: Default::default(),
                jobs: 0,
            }),
        };
        draw_fn(&mut painter);
        let (segments, _plan) = painter.finish_backdrops();

        let result = crate::backdrop::execute(
            &mut self.backdrop,
            &mut self.vello_renderer,
            self.buffer_renderer.device(),
            self.buffer_renderer.queue(),
            &segments,
            &self.scene,
            &self.buffer_renderer.target_texture_view(),
            &vello::RenderParams {
                base_color: vello::peniko::Color::TRANSPARENT,
                width: size.width,
                height: size.height,
                antialiasing_method: vello::AaConfig::Area,
            },
        );
        self.backdrop
            .end_frame(&mut self.vello_renderer, &mut self.texture_handles);
        result.expect("Got non-Send/Sync error from rendering");

        self.buffer_renderer.copy_texture_to_buffer(cpu_buffer);

        // Empty the Vello scene (memory optimisation)
        self.scene.reset();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use anyrender::{
        filters::{Filter, FilterEffect},
        PaintScene,
    };
    use kurbo::{Affine, Rect};
    use peniko::{Color, Fill, Mix};
    use std::sync::Arc;

    const WIDTH: u32 = 200;
    const HEIGHT: u32 = 100;
    const SEAM: u32 = WIDTH / 2;

    fn gpu_available() -> bool {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or_default(),
            flags: wgpu::InstanceFlags::from_build_config().with_env(),
            backend_options: wgpu::BackendOptions::from_env_or_default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .is_ok()
    }

    fn draw_backdrop(scene: &mut VelloScenePainter<'_, '_>) {
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::BLACK,
            None,
            &Rect::new(0.0, 0.0, SEAM as f64, HEIGHT as f64),
        );
        scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::WHITE,
            None,
            &Rect::new(SEAM as f64, 0.0, WIDTH as f64, HEIGHT as f64),
        );
        let panel = Rect::new(40.0, 0.0, 160.0, HEIGHT as f64);
        scene.push_layer(
            Mix::Normal,
            1.0,
            Affine::IDENTITY,
            &panel,
            None,
            Some(Arc::new(Filter::single(FilterEffect::blur(12.0)))),
        );
        scene.pop_layer();
    }

    fn pixel(buffer: &[u8], x: u32, y: u32) -> u8 {
        buffer[((y * WIDTH + x) * 4) as usize]
    }

    #[test]
    fn reused_headless_renderer_releases_idle_backdrop_registrations() {
        if !gpu_available() {
            eprintln!(
                "reused_headless_renderer_releases_idle_backdrop_registrations: \
                 no GPU adapter; skipped"
            );
            return;
        }

        let mut renderer = VelloImageRenderer::new(WIDTH, HEIGHT);
        let mut buffer = Vec::new();
        renderer.render_to_vec(draw_backdrop, &mut buffer);
        assert_eq!(renderer.texture_handles.len(), 2);

        let ramp: Vec<u8> = (0..7)
            .map(|i| pixel(&buffer, SEAM - 18 + i * 6, HEIGHT / 2))
            .collect();
        assert!(ramp.windows(2).all(|window| window[1] >= window[0]));
        assert!(ramp[0] < 96 && *ramp.last().unwrap() > 160, "got {ramp:?}");
        assert!(ramp.iter().any(|v| (32..=224).contains(v)), "blur must survive trim/reuse: {ramp:?}");

        for _ in 0..179 {
            renderer.render_to_vec(|_| {}, &mut buffer);
        }
        assert_eq!(renderer.texture_handles.len(), 2);

        renderer.render_to_vec(|_| {}, &mut buffer);
        assert!(renderer.texture_handles.is_empty());

        renderer.render_to_vec(draw_backdrop, &mut buffer);
        assert_eq!(renderer.texture_handles.len(), 2);
        let ramp: Vec<u8> = (0..7)
            .map(|i| pixel(&buffer, SEAM - 18 + i * 6, HEIGHT / 2))
            .collect();
        assert!(ramp.windows(2).all(|window| window[1] >= window[0]));
        assert!(ramp[0] < 96 && *ramp.last().unwrap() > 160, "got {ramp:?}");
        assert!(ramp.iter().any(|v| (32..=224).contains(v)), "blur must survive trim/reuse: {ramp:?}");
    }
}
