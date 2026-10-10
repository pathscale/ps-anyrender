use std::sync::Arc;

use kurbo::{Affine, Rect};
use peniko::{Extend, Fill, ImageBrush, ImageQuality, ImageSampler};
use vello_cpu::{ImageSource, PaintType, Pixmap};

use crate::VelloCpuScenePainter;

/// An immutable native, premultiplied RGBA8 rendering.
///
/// Cloning retains the allocation. The bytes are not straight RGBA.
#[derive(Clone)]
pub struct CpuPixmap {
    pixmap: Arc<Pixmap>,
    width: u16,
    height: u16,
}

impl AsRef<[u8]> for CpuPixmap {
    fn as_ref(&self) -> &[u8] {
        self.pixmap.data_as_u8_slice()
    }
}

impl VelloCpuScenePainter {
    /// Finish without copying or unpremultiplying the native pixel allocation.
    pub fn finish_shared(self) -> CpuPixmap {
        let width = self.render_ctx.width();
        let height = self.render_ctx.height();
        CpuPixmap {
            pixmap: Arc::new(self.finish()),
            width,
            height,
        }
    }

    /// Seed a fresh, equally sized painter from an immutable native rendering.
    ///
    /// This records an image fill, but does not copy pixels, premultiply them,
    /// or insert an entry in the image conversion cache. Call before recording
    /// the new content. Small canvas tiles bound the cost of the image fill.
    pub fn seed_pixmap(&mut self, source: &CpuPixmap) {
        assert_eq!(self.render_ctx.width(), source.width);
        assert_eq!(self.render_ctx.height(), source.height);
        assert_eq!(self.open_layers(), 0);
        self.render_ctx.set_transform(Affine::IDENTITY);
        self.render_ctx.set_fill_rule(Fill::NonZero);
        self.render_ctx.set_paint_transform(Affine::IDENTITY);
        self.render_ctx.set_paint(PaintType::Image(ImageBrush {
            image: ImageSource::Pixmap(Arc::clone(&source.pixmap)),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        }));
        self.render_ctx.fill_rect(&Rect::new(
            0.0,
            0.0,
            f64::from(source.width),
            f64::from(source.height),
        ));
    }
}
