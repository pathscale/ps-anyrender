use std::sync::Arc;

use crate::backdrop::{BackdropPool, BlurJob, FrameSegments, OpenLayer, SegmentBoundary};
use anyrender::{
    BackdropPlanner, Boundary, Filter, FramePlan, NormalizedCoord, Paint, PaintRef, PaintScene,
    RenderContext, ResourceId,
};
use kurbo::{Affine, Rect, Shape, Stroke};
use peniko::{
    BlendMode, BrushRef, Color, Fill, FontData, ImageBrush, ImageBrushRef, ImageData, StyleRef,
};

const DEFAULT_TOLERANCE: f64 = 0.1;
use rustc_hash::FxHashMap;
use vello::Renderer as VelloRenderer;
use wgpu::Texture;
use wgpu_context::DeviceHandle;

pub struct VelloScenePainter<'r, 's> {
    pub(crate) renderer: Option<&'r mut VelloRenderer>,
    pub(crate) device_handle: Option<&'r DeviceHandle>,
    pub(crate) texture_handles: Option<&'r mut FxHashMap<ResourceId, ImageData>>,
    pub(crate) inner: &'s mut vello::Scene,
    /// Everything needed to satisfy `backdrop-filter`, or `None`.
    ///
    /// `None` is the honest state for a painter with no device behind it: the
    /// filter needs a rendered snapshot, and a scene being recorded into a
    /// `vello::Scene` for later has nothing to snapshot. Those painters drop
    /// backdrop filters, exactly as every vello painter did before this.
    pub(crate) backdrop: Option<BackdropState<'r>>,
}

/// Per-frame segmentation state.
///
/// Lives on the painter rather than the renderer because the cut has to happen
/// at the moment the filtered layer is pushed, in the middle of the walk, and
/// the walk is the painter's.
pub(crate) struct BackdropState<'r> {
    pub device: &'r wgpu::Device,
    pub pool: &'r mut BackdropPool,
    pub frame: (u32, u32),
    /// Decides which ops can share a snapshot. See `anyrender::backdrop`.
    pub planner: BackdropPlanner,
    /// The layers open right now, so a cut can close and reopen them.
    pub stack: Vec<OpenLayer>,
    pub segments: FrameSegments,
    /// How many blur slots have been handed out this frame.
    pub jobs: usize,
}

impl RenderContext for VelloScenePainter<'_, '_> {
    fn try_register_custom_resource(
        &mut self,
        resource: Box<dyn std::any::Any>,
    ) -> Result<ResourceId, anyrender::RegisterResourceError> {
        if let Some(renderer) = &mut self.renderer
            && let Some(texture_handles) = &mut self.texture_handles
        {
            if let Ok(texture) = resource.downcast::<Texture>() {
                let id = ResourceId::new();
                texture_handles.insert(id, renderer.register_texture(*texture));
                Ok(id)
            } else {
                Err(anyrender::RegisterResourceErrorKind::UnsupportedResourceKind.into())
            }
        } else {
            Err(anyrender::RegisterResourceErrorKind::Unimplemented.into())
        }
    }

    fn unregister_resource(&mut self, resource_id: ResourceId) {
        if let Some(renderer) = &mut self.renderer
            && let Some(texture_handles) = &mut self.texture_handles
            && let Some(handle) = texture_handles.remove(&resource_id)
        {
            renderer.unregister_texture(handle);
        }
    }

    fn renderer_specific_context(&self) -> Option<Box<dyn std::any::Any>> {
        self.device_handle
            .map(|device_handle| Box::new(device_handle.clone()) as _)
    }
}

impl VelloScenePainter<'_, '_> {
    pub fn new<'s>(scene: &'s mut vello::Scene) -> VelloScenePainter<'static, 's> {
        VelloScenePainter {
            renderer: None,
            device_handle: None,
            texture_handles: None,
            inner: scene,
            backdrop: None,
        }
    }

    /// Hand back the segments this frame was cut into, and the plan behind them.
    ///
    /// The scene still being built stays where it is: it is the last segment,
    /// and the caller renders it to the surface rather than to a snapshot.
    pub(crate) fn finish_backdrops(&mut self) -> (FrameSegments, FramePlan) {
        match self.backdrop.take() {
            Some(state) => (state.segments, state.planner.finish()),
            None => (FrameSegments::default(), FramePlan::default()),
        }
    }
}

/// Draw a registered texture at one texel per pixel, with its top-left at `origin`.
///
/// The resource indirection the rest of this file uses cannot be reached from
/// here - it lives inside `fill`, which needs `&mut self` while the segment
/// state is already borrowed - so the handle is resolved directly, which is the
/// same lookup `fill` performs.
fn draw_registered_image(
    scene: &mut vello::Scene,
    handles: &FxHashMap<ResourceId, ImageData>,
    id: ResourceId,
    origin: kurbo::Point,
    size: (u32, u32),
) {
    let Some(image) = handles.get(&id) else {
        return;
    };
    let rect = Rect::from_origin_size(origin, (f64::from(size.0), f64::from(size.1)));
    scene.fill(
        Fill::NonZero,
        Affine::IDENTITY,
        BrushRef::Image(ImageBrushRef {
            image,
            sampler: peniko::ImageSampler::default(),
        }),
        // The brush is in its own texel space, so it has to be moved to where
        // the rectangle is. Without this every snapshot and every blurred panel
        // draws from the frame's top-left corner.
        Some(Affine::translate(origin.to_vec2())),
        &rect,
    );
}

impl VelloScenePainter<'_, '_> {
    /// Cut the scene here, or join the batch already open.
    ///
    /// On return the incoming segment is ready for the filtered layer to be
    /// pushed onto it: the previous segment closed, its snapshot drawn back,
    /// the ancestor clips reopened, and this element's blurred backdrop drawn
    /// through its own shape.
    fn record_backdrop(
        &mut self,
        backdrop_filter: Arc<Filter>,
        transform: Affine,
        clip: &impl Shape,
    ) {
        // Reserving a slot needs the pool, the renderer and the handle map at
        // once, and they are three separate fields, so the borrows have to be
        // taken apart by hand.
        let Self {
            renderer,
            texture_handles,
            inner,
            backdrop,
            ..
        } = self;
        let (Some(state), Some(renderer), Some(handles)) = (
            backdrop.as_mut(),
            renderer.as_mut(),
            texture_handles.as_mut(),
        ) else {
            return;
        };

        let device_clip = transform * clip.into_path(DEFAULT_TOLERANCE);
        let bounds = device_clip.bounding_box();
        let expansion = backdrop_filter.expansion_rect();
        let source = Rect::new(
            bounds.x0 + expansion.x0,
            bounds.y0 + expansion.y0,
            bounds.x1 + expansion.x1,
            bounds.y1 + expansion.y1,
        );
        let Some((origin, size)) = clamp_to_frame(source, state.frame) else {
            return;
        };

        let cut = state
            .planner
            .backdrop(backdrop_filter.clone(), device_clip.clone(), bounds)
            == Boundary::NewSegment;

        let job = state.jobs;
        state.jobs += 1;
        let boundary = if cut {
            state.segments.boundaries.len()
        } else {
            state.segments.boundaries.len().saturating_sub(1)
        };
        let ids = state.pool.reserve(
            state.device,
            renderer,
            handles,
            boundary,
            job,
            state.frame,
            (size[0], size[1]),
        );

        if cut {
            // Close the open clips. A segment handed to the rasteriser with an
            // unbalanced layer stack is not a rendering artifact, it is a panic
            // or a silently dropped clip depending on how far it got.
            for _ in 0..state.stack.len() {
                inner.pop_layer();
            }
            let finished = std::mem::replace(*inner, vello::Scene::new());
            state.segments.scenes.push(finished);
            state.segments.boundaries.push(SegmentBoundary {
                snapshot: boundary,
                jobs: Vec::new(),
            });

            // Draw the snapshot back, unclipped, before anything else. This is
            // the whole reason a segment costs a full-frame pass:
            // `render_to_texture` clears its target, so the only route back to
            // the previous segment's pixels is to draw them again.
            draw_registered_image(
                inner,
                handles,
                ids.snapshot,
                kurbo::Point::ZERO,
                state.frame,
            );

            // Reopen what was closed above, outermost first.
            for layer in &state.stack {
                match layer.blend {
                    Some((blend, alpha)) => {
                        inner.push_layer(Fill::NonZero, blend, alpha, layer.transform, &layer.clip)
                    }
                    None => inner.push_clip_layer(Fill::NonZero, layer.transform, &layer.clip),
                }
            }
        }

        if let Some(record) = state.segments.boundaries.last_mut() {
            record.jobs.push(BlurJob {
                slot: job,
                origin,
                size,
                sigma: blur_sigma(&backdrop_filter),
            });
        }

        // The blurred backdrop, through this element's own shape. Drawn before
        // the element's layer is pushed, which is what puts it behind the
        // element's background and borders rather than over them.
        inner.push_clip_layer(Fill::NonZero, Affine::IDENTITY, &device_clip);
        draw_registered_image(
            inner,
            handles,
            ids.blurred,
            kurbo::Point::new(f64::from(origin[0]), f64::from(origin[1])),
            (size[0], size[1]),
        );
        inner.pop_layer();
    }

    /// Remember a layer so a cut inside it can put it back.
    fn note_layer(
        &mut self,
        blend: Option<(BlendMode, f32)>,
        transform: Affine,
        clip: &impl Shape,
    ) {
        if let Some(state) = self.backdrop.as_mut() {
            state
                .planner
                .push_layer(transform.transform_rect_bbox(clip.bounding_box()));
            state.stack.push(OpenLayer {
                blend,
                transform,
                clip: clip.into_path(DEFAULT_TOLERANCE),
            });
        }
    }

    fn note_pop(&mut self) {
        if let Some(state) = self.backdrop.as_mut() {
            state.planner.pop_layer();
            state.stack.pop();
        }
    }

    /// Tell the planner about a draw that no layer bounds.
    ///
    /// Gated on the planner actually tracking something, because otherwise this
    /// is a bounding box computed per drawing command per frame for a page with
    /// no glass on it, which is the per-frame cost the whole design is trying
    /// not to introduce.
    fn note_draw(&mut self, transform: Affine, shape: &impl Shape) {
        if let Some(state) = self.backdrop.as_mut()
            && state.planner.is_tracking()
        {
            let bounds = transform.transform_rect_bbox(shape.bounding_box());
            state.planner.draw(bounds);
        }
    }
}

/// The largest standard deviation any blur in the graph asks for.
///
/// A `Filter` is a graph and can hold several primitives. Only the blur is
/// implemented here, so a graph carrying anything else has its other primitives
/// silently skipped - which is the state every vello backend was already in,
/// rather than a regression introduced by this.
fn blur_sigma(filter: &Filter) -> f32 {
    // Read back off the expansion the graph reports, which is where the
    // authoritative number already lives: `expansion_rect` is built from the
    // same primitives and is public, whereas the primitive list is not.
    (filter.expansion_rect().width() as f32 / 6.0).max(0.0)
}

/// The frame-clamped, texel-aligned region a filter has to read.
///
/// Rounded outward. Rounded inward, an element's own edge samples texels the
/// blur never wrote, which reads as a one-pixel dark line around every panel.
fn clamp_to_frame(source: Rect, frame: (u32, u32)) -> Option<([i32; 2], [u32; 2])> {
    let x0 = source.x0.max(0.0).floor();
    let y0 = source.y0.max(0.0).floor();
    let x1 = source.x1.min(f64::from(frame.0)).ceil();
    let y1 = source.y1.min(f64::from(frame.1)).ceil();
    if x1 <= x0 || y1 <= y0 {
        return None;
    }
    Some(([x0 as i32, y0 as i32], [(x1 - x0) as u32, (y1 - y0) as u32]))
}

impl PaintScene for VelloScenePainter<'_, '_> {
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        _filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) {
        // Before the layer, not after: `backdrop-filter` blurs what is already
        // behind the element, so the cut belongs on the far side of the layer
        // that will hold the element's own background and borders.
        if let Some(backdrop_filter) = backdrop_filter {
            self.record_backdrop(backdrop_filter, transform, clip);
        }
        let blend = blend.into();
        self.note_layer(Some((blend, alpha)), transform, clip);
        self.inner
            .push_layer(Fill::NonZero, blend, alpha, transform, clip);
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        self.note_layer(None, transform, clip);
        self.inner.push_clip_layer(Fill::NonZero, transform, clip);
    }

    fn pop_layer(&mut self) {
        self.note_pop();
        self.inner.pop_layer();
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint_ref: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.note_draw(transform, shape);
        let paint_ref: PaintRef<'_> = paint_ref.into();
        let brush_ref: BrushRef<'_> = paint_ref.into();
        self.inner
            .stroke(style, transform, brush_ref, brush_transform, shape);
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.note_draw(transform, shape);
        let paint: PaintRef<'_> = paint.into();
        let brush_ref: BrushRef<'_> = match paint {
            Paint::Solid(color) => BrushRef::Solid(color),
            Paint::Gradient(gradient) => BrushRef::Gradient(gradient),
            Paint::Image(image) => BrushRef::Image(image),
            Paint::Resource(brush) => {
                let resource_id = brush.image;
                if let Some(texture_handle) = self
                    .texture_handles
                    .as_ref()
                    .and_then(|texture_handles| texture_handles.get(&resource_id))
                {
                    peniko::Brush::Image(ImageBrush {
                        image: texture_handle,
                        sampler: brush.sampler,
                    })
                } else {
                    BrushRef::Solid(Color::TRANSPARENT)
                }
            }
            Paint::Custom(_) => BrushRef::Solid(Color::TRANSPARENT),
        };

        self.inner
            .fill(style, transform, brush_ref, brush_transform, shape);
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'a mut self,
        font: &'a FontData,
        font_size: f32,
        hint: bool,
        normalized_coords: &'a [NormalizedCoord],
        embolden: kurbo::Vec2,
        style: impl Into<StyleRef<'a>>,
        paint: impl Into<PaintRef<'a>>,
        brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph>,
    ) {
        self.inner
            .draw_glyphs(font)
            .font_size(font_size)
            .hint(hint)
            .normalized_coords(normalized_coords)
            .font_embolden(vello::FontEmbolden::new(kurbo::Diagonal2::new(
                embolden.x, embolden.y,
            )))
            .brush(paint.into())
            .brush_alpha(brush_alpha)
            .transform(transform)
            .glyph_transform(glyph_transform)
            .draw(
                style,
                glyphs.map(|g: anyrender::Glyph| vello::Glyph {
                    id: g.id,
                    x: g.x,
                    y: g.y,
                }),
            );
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        // A box shadow spreads well past its rectangle, and a panel with an
        // opaque background draws one with no clip layer around it, so it is a
        // draw the planner cannot account for at a layer.
        let reach = std_dev * 3.0 + radius;
        self.note_draw(transform, &rect.inflate(reach, reach));
        self.inner
            .draw_blurred_rounded_rect(transform, rect, brush, radius, std_dev);
    }
}
