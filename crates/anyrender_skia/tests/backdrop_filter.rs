//! Does this backend blur what is *behind* a layer?
//!
//! Asked because the vello backends do not. `anyrender_vello` discards both
//! `filter` and `backdrop_filter`; `vello_hybrid` has a complete GPU filter
//! pipeline but feeds it the layer's own content, and the enum member for the
//! other input, `FilterSource::BackgroundImage`, appears exactly once in all of
//! linebender/vello, at its declaration. So `backdrop-filter` has nowhere to
//! land there today.
//!
//! Skia has had the mechanism since forever: `SkCanvas::saveLayer` takes a
//! backdrop image filter, which is applied to the destination pixels the layer
//! is drawn over. This backend already passes it through, and this test is here
//! to prove that end to end rather than by reading.

use anyrender::{
    PaintScene,
    filters::{Filter, FilterEffect},
    render_to_buffer,
};
use anyrender_skia::SkiaImageRenderer;
use kurbo::{Affine, Rect};
use peniko::{Color, Fill, Mix};
use std::sync::Arc;

const WIDTH: u32 = 200;
const HEIGHT: u32 = 100;
/// The seam sits at the halfway mark, so a blur has to bleed across it.
const SEAM: u32 = WIDTH / 2;

fn pixel(buffer: &[u8], x: u32, y: u32) -> [u8; 3] {
    let offset = ((y * WIDTH + x) * 4) as usize;
    [buffer[offset], buffer[offset + 1], buffer[offset + 2]]
}

/// Paint a hard black/white seam, then optionally draw a transparent layer over
/// the middle carrying a backdrop blur.
fn scene(backdrop: Option<Arc<Filter>>) -> Vec<u8> {
    render_to_buffer::<SkiaImageRenderer, _>(
        |scene| {
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

            if let Some(backdrop) = backdrop {
                let panel = Rect::new(40.0, 0.0, 160.0, HEIGHT as f64);
                // Alpha 1.0 and nothing drawn inside: whatever shows through is
                // the backdrop, filtered or not.
                scene.push_layer(Mix::Normal, 1.0, Affine::IDENTITY, &panel, None, Some(backdrop));
                scene.pop_layer();
            }
        },
        WIDTH,
        HEIGHT,
    )
}

/// The fixture has to be sound before its result means anything: with no layer
/// at all, both halves must stay pure.
#[test]
fn the_unfiltered_seam_is_hard() {
    let buffer = scene(None);

    assert_eq!(pixel(&buffer, SEAM - 8, 50), [0, 0, 0]);
    assert_eq!(pixel(&buffer, SEAM + 8, 50), [255, 255, 255]);
}

#[test]
fn a_backdrop_blur_bleeds_the_seam_across_itself() {
    let blurred = scene(Some(Arc::new(Filter::single(FilterEffect::blur(12.0)))));

    // Sampled as a ramp rather than at one point. A single sample far from the
    // seam reads pure black, which is correct, and passes against a backend
    // that does nothing at all.
    let ramp: Vec<u8> = (0..7)
        .map(|i| pixel(&blurred, SEAM - 18 + i * 6, 50)[0])
        .collect();

    assert!(
        ramp.windows(2).all(|w| w[1] >= w[0]),
        "crossing the seam should brighten monotonically, got {ramp:?}"
    );
    assert!(
        ramp[0] < 96 && *ramp.last().unwrap() > 160,
        "the ramp should run dark to light, got {ramp:?}"
    );
    assert!(
        ramp.iter().any(|v| (32..=224).contains(v)),
        "a blurred seam must produce intermediate values, got {ramp:?}"
    );
}
