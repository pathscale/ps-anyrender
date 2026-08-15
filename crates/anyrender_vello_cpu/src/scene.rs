use std::sync::Arc;

use anyrender::{Filter, NormalizedCoord, Paint, PaintRef, PaintScene, RenderContext};
use glifo::FontEmbolden;
use kurbo::{Affine, Diagonal2, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, ImageBrush, StyleRef};
use vello_cpu::{ImageSource, PaintType, Pixmap};

const DEFAULT_TOLERANCE: f64 = 0.1;

fn anyrender_paint_to_vello_cpu_paint<'a>(paint: PaintRef<'a>) -> PaintType {
    match paint {
        Paint::Solid(alpha_color) => PaintType::Solid(alpha_color),
        Paint::Gradient(gradient) => PaintType::Gradient(gradient.clone()),
        Paint::Image(image) => PaintType::Image(ImageBrush {
            #[cfg(not(feature = "experimental_image_cache"))]
            image: ImageSource::from_peniko_image_data(image.image),
            #[cfg(feature = "experimental_image_cache")]
            image: convert_image_cached(image.image),
            sampler: image.sampler,
        }),
        // TODO: custom paint
        Paint::Resource(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
        Paint::Custom(_) => PaintType::Solid(peniko::color::palette::css::TRANSPARENT),
    }
}

#[cfg(feature = "experimental_image_cache")]
fn convert_image_cached(image: &peniko::ImageData) -> ImageSource {
    use std::collections::HashMap;
    use std::sync::{LazyLock, Mutex};
    static CACHE: LazyLock<Mutex<HashMap<u64, ImageSource>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));

    let mut map = CACHE.lock().unwrap();
    let id = image.data.id();
    map.entry(id)
        .or_insert_with(|| ImageSource::from_peniko_image_data(image))
        .clone()
}

pub struct VelloCpuScenePainter {
    pub render_ctx: vello_cpu::RenderContext,
    pub resources: vello_cpu::Resources,
    /// How many layers are open on `render_ctx` right now.
    ///
    /// Tracked because a backdrop snapshot is only legal at depth zero:
    /// `render_to_pixmap` asserts `!wide.has_layers()`, so asking for one while
    /// a layer is open aborts the process. See `paint_filtered_backdrop`.
    ///
    /// Public because the other fields are: this struct is constructed by
    /// literal outside this crate, so hiding one field would break those
    /// callers for no gain. Start it at zero and leave it alone.
    pub open_layers: u32,
}

impl VelloCpuScenePainter {
    pub fn finish(mut self) -> Pixmap {
        let mut pixmap = Pixmap::new(self.render_ctx.width(), self.render_ctx.height());
        self.render_ctx
            .render_to_pixmap(&mut self.resources, &mut pixmap);
        pixmap
    }

    /// Draw the scene so far back over itself, filtered, inside `clip`.
    ///
    /// This is `backdrop-filter`. A layer filter runs over the layer's *own*
    /// content, so it can never express "blur what is behind me" — the input
    /// for that is `FilterSource::BackgroundImage`, which vello declares and
    /// implements nowhere. Rather than wait for it, take the backdrop the only
    /// way it is actually available: render what has been drawn up to this
    /// point, then draw that image back through a filtered layer clipped to the
    /// same shape. The element's own content is then drawn on top by the
    /// ordinary layer the caller pushes next.
    ///
    /// `render_to_pixmap` takes `&self`, which is what makes this possible
    /// mid-scene: the context is snapshotted, not consumed, and keeps
    /// accumulating afterwards.
    ///
    /// It costs a full render of the frame per backdrop-filtered layer, so it
    /// is priced for a handful of glass panels and not for a page of them.
    #[cfg(feature = "filters")]
    fn paint_filtered_backdrop(
        &mut self,
        backdrop_filter: Option<Arc<Filter>>,
        transform: Affine,
        clip: &impl Shape,
    ) {
        let Some(backdrop_filter) = backdrop_filter else {
            return;
        };
        // Same restriction as the layer filter path above: vello_cpu declines
        // to apply filters while multithreaded.
        if cfg!(feature = "multithreading") {
            return;
        }
        let Some(filter) = crate::filters::convert_filter(backdrop_filter) else {
            return;
        };
        // A backdrop snapshot renders the scene so far into a pixmap, and
        // `render_to_pixmap` asserts that no layer is open. Blitz-style
        // painters reach `push_layer` several frames deep, so that assert does
        // not hold on any real page and the process aborts with
        //
        //     some layers haven't been popped yet
        //
        // rather than dropping an effect. Declining is the only honest answer
        // this backend can give until it renders backdrops as their own pass
        // the way `anyrender_vello` does; a panic is not an answer at all.
        //
        // The effect still applies at depth zero, which is where a page-level
        // backdrop sits. What is lost is a backdrop nested inside another
        // layer, and it is lost visibly rather than fatally.
        if self.open_layers > 0 {
            return;
        }

        let (width, height) = (self.render_ctx.width(), self.render_ctx.height());
        if width == 0 || height == 0 {
            return;
        }

        let mut backdrop = Pixmap::new(width, height);
        self.render_ctx
            .render_to_pixmap(&mut self.resources, &mut backdrop);

        let canvas = Rect::new(0.0, 0.0, f64::from(width), f64::from(height));
        self.render_ctx.set_transform(transform);
        self.render_ctx.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            None,
            None,
            None,
            Some(filter),
        );
        // The snapshot is in device pixels, so it is drawn untransformed
        // whatever transform the layer itself carries.
        self.render_ctx.set_transform(Affine::IDENTITY);
        self.render_ctx.set_paint(PaintType::Image(ImageBrush {
            image: ImageSource::Pixmap(Arc::new(backdrop)),
            sampler: peniko::ImageSampler::default(),
        }));
        self.render_ctx.fill_rect(&canvas);
        self.render_ctx.pop_layer();
    }
}

impl RenderContext for VelloCpuScenePainter {}
impl PaintScene for VelloCpuScenePainter {
    fn reset(&mut self) {
        self.render_ctx.reset();
        self.open_layers = 0;
    }

    fn push_layer(
        &mut self,
        blend: impl Into<BlendMode>,
        alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) {
        #[cfg(feature = "filters")]
        let filter = filter
            .and_then(crate::filters::convert_filter)
            .filter(|_| cfg!(not(feature = "multithreading")));

        #[cfg(not(feature = "filters"))]
        let filter = {
            let _ = filter;
            None
        };

        #[cfg(feature = "filters")]
        self.paint_filtered_backdrop(backdrop_filter, transform, clip);
        #[cfg(not(feature = "filters"))]
        let _ = backdrop_filter;

        self.render_ctx.set_transform(transform);
        self.render_ctx.push_layer(
            Some(&clip.into_path(DEFAULT_TOLERANCE)),
            Some(blend.into()),
            Some(alpha),
            None,
            filter,
        );
        self.open_layers += 1;
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        self.render_ctx.set_transform(transform);
        self.render_ctx
            .push_clip_layer(&clip.into_path(DEFAULT_TOLERANCE));
        self.open_layers += 1;
    }

    fn pop_layer(&mut self) {
        self.render_ctx.pop_layer();
        // Saturating: a caller that pops more than it pushed is already in
        // trouble, and going through zero here would silently re-enable a
        // snapshot inside a layer, which is the abort this counter prevents.
        self.open_layers = self.open_layers.saturating_sub(1);
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_stroke(style.clone());
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .stroke_path(&shape.into_path(DEFAULT_TOLERANCE));
    }

    fn fill<'a>(
        &mut self,
        style: Fill,
        transform: Affine,
        paint: impl Into<PaintRef<'a>>,
        brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_fill_rule(style);
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));
        self.render_ctx
            .set_paint_transform(brush_transform.unwrap_or(Affine::IDENTITY));
        self.render_ctx
            .fill_path(&shape.into_path(DEFAULT_TOLERANCE));
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
        _brush_alpha: f32,
        transform: Affine,
        glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = anyrender::Glyph> + Clone,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx
            .set_paint(anyrender_paint_to_vello_cpu_paint(paint.into()));

        let style: StyleRef<'a> = style.into();
        match style {
            StyleRef::Fill(fill) => {
                self.render_ctx.set_fill_rule(fill);
                self.render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .font_embolden(FontEmbolden::new(Diagonal2::new(embolden.x, embolden.y)))
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .fill_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
            StyleRef::Stroke(stroke) => {
                self.render_ctx.set_stroke(stroke.clone());
                self.render_ctx
                    .glyph_run(&mut self.resources, font)
                    .font_size(font_size)
                    .hint(hint)
                    .normalized_coords(normalized_coords)
                    .glyph_transform(glyph_transform.unwrap_or_default())
                    .stroke_glyphs(glyphs.map(|g| vello_cpu::Glyph {
                        id: g.id,
                        x: g.x,
                        y: g.y,
                    }));
            }
        }
    }
    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        color: Color,
        radius: f64,
        std_dev: f64,
    ) {
        self.render_ctx.set_transform(transform);
        self.render_ctx.set_paint(PaintType::Solid(color));
        self.render_ctx
            .fill_blurred_rounded_rect(&rect, radius as f32, std_dev as f32);
    }
}
