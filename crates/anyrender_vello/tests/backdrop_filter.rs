//! Does this backend blur what is *behind* a layer?
//!
//! Deliberately the same fixture and the same assertions as
//! `anyrender_skia/tests/backdrop_filter.rs`, so the two are read side by side.
//! That test was written to establish that Skia could do this and vello could
//! not: `anyrender_vello` took `_backdrop_filter` and dropped it, and vello 0.9
//! has no layer filter to forward it to.
//!
//! It does now, built above vello rather than inside it. The scene is cut at
//! the filtered layer, what has been drawn so far is rendered to a texture, a
//! separable gaussian runs over the region the filter reads, and the result is
//! drawn back through the element's own shape before the next segment carries
//! on. See `src/backdrop.rs`.
//!
//! Needs a GPU. It runs through `VelloImageRenderer`, which is headless - no
//! window, no surface, no event loop - but still a real adapter, so this is a
//! local and macOS/Metal-or-Vulkan check rather than something a CPU-only
//! runner can answer. The GPU-free half of the same work, how many passes a
//! page costs, is asserted in `ps-blitz`'s `glass_pass_count.rs`.

use anyrender::{
    PaintScene,
    filters::{Filter, FilterEffect},
    render_to_buffer,
};
use kurbo::{Affine, Rect};
use peniko::{Color, Fill, Mix};
use ps_anyrender_vello::VelloImageRenderer;
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
    render_to_buffer::<VelloImageRenderer, _>(
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
                scene.push_layer(
                    Mix::Normal,
                    1.0,
                    Affine::IDENTITY,
                    &panel,
                    None,
                    Some(backdrop),
                );
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
    println!("ramp across the seam: {ramp:?}");

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

/// Two panels, which is where the segmenting is actually exercised.
///
/// One filtered layer needs one cut and proves the mechanism. Two need the
/// parts that are easy to get wrong and invisible in a single-panel test: a
/// second snapshot, a second set of pool slots, and the decision about whether
/// they can share a snapshot at all. Both must blur, not just the first.
#[test]
fn two_panels_both_blur() {
    let blurred = render_to_buffer::<VelloImageRenderer, _>(
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
            // Both straddle the seam, stacked vertically, far enough apart in y
            // that neither reads the other.
            for band in [(0.0, 40.0), (60.0, 100.0)] {
                let panel = Rect::new(40.0, band.0, 160.0, band.1);
                scene.push_layer(
                    Mix::Normal,
                    1.0,
                    Affine::IDENTITY,
                    &panel,
                    None,
                    Some(Arc::new(Filter::single(FilterEffect::blur(8.0)))),
                );
                scene.pop_layer();
            }
        },
        WIDTH,
        HEIGHT,
    );

    for y in [20, 80] {
        let ramp: Vec<u8> = (0..5)
            .map(|i| pixel(&blurred, SEAM - 12 + i * 6, y)[0])
            .collect();
        println!("ramp at y={y}: {ramp:?}");
        assert!(
            ramp.windows(2).all(|w| w[1] >= w[0]),
            "the panel at y={y} should ramp across the seam, got {ramp:?}"
        );
        assert!(
            ramp.iter().any(|v| (32..=224).contains(v)),
            "the panel at y={y} did not blur, got {ramp:?}"
        );
    }
}

/// A filtered layer inside a clip has to stay inside it after the cut.
///
/// Cutting the scene leaves the ancestor clips open in a segment about to be
/// rasterised, so the painter closes them and pushes them again on the far
/// side. If that replay is wrong the panel's blur paints outside the container
/// that was clipping it, which in the application is a glass panel drawn over
/// its own scrollbar.
#[test]
fn a_clip_around_a_filtered_layer_survives_the_cut() {
    let buffer = render_to_buffer::<VelloImageRenderer, _>(
        |scene| {
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::BLACK,
                None,
                &Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64),
            );
            // The container clips away everything below y=50.
            scene.push_clip_layer(Affine::IDENTITY, &Rect::new(0.0, 0.0, WIDTH as f64, 50.0));
            // A panel that would otherwise cover the whole frame in white.
            let panel = Rect::new(0.0, 0.0, WIDTH as f64, HEIGHT as f64);
            scene.push_layer(
                Mix::Normal,
                1.0,
                Affine::IDENTITY,
                &panel,
                None,
                Some(Arc::new(Filter::single(FilterEffect::blur(4.0)))),
            );
            scene.fill(Fill::NonZero, Affine::IDENTITY, Color::WHITE, None, &panel);
            scene.pop_layer();
            scene.pop_layer();
        },
        WIDTH,
        HEIGHT,
    );

    assert_eq!(
        pixel(&buffer, 100, 20),
        [255, 255, 255],
        "inside the clip the panel paints"
    );
    assert_eq!(
        pixel(&buffer, 100, 80),
        [0, 0, 0],
        "below the clip it must not: the ancestor clip was lost across the cut"
    );
}

/// Outside the panel nothing is filtered, and the seam stays hard.
///
/// The failure this catches is a blur that ran over the whole frame instead of
/// the element's region, which would look right in every sample above and be
/// the entire performance problem.
#[test]
fn the_blur_stays_inside_the_element() {
    let blurred = scene(Some(Arc::new(Filter::single(FilterEffect::blur(12.0)))));

    // The panel spans x 40..160. Well outside it, on both sides.
    assert_eq!(
        pixel(&blurred, 8, 50),
        [0, 0, 0],
        "left of the panel is untouched black"
    );
    assert_eq!(
        pixel(&blurred, 190, 50),
        [255, 255, 255],
        "right of the panel is untouched white"
    );
}
