//! Does this backend blur what is *behind* a layer, at the depth a page uses?
//!
//! The depth is the whole test. A backdrop snapshot renders the scene so far
//! into a pixmap, and `render_to_pixmap` asserts no layer is open. Blitz-style
//! painters reach `push_layer` several frames deep, so on a real page the depth
//! is never zero: this backend first aborted the process there, then declined
//! every backdrop it was asked for, and both were reported as glass not
//! working. Only the nested case tells those apart from a working one, so it is
//! the case that is asserted.
//!
//! The two blur cases are compiled out under `multithreading`. vello_cpu
//! applies no filter at all while multithreaded, so this backend declines every
//! backdrop there and a blur assertion has nothing to assert. CI runs
//! `--all-features`, which is why these went red there while passing locally.
//! The stack test stays, because balancing the layer stack is not a filtering
//! question.

#![cfg(feature = "filters")]

use std::sync::Arc;

use anyrender::{
    PaintScene,
    filters::{Filter, FilterEffect},
};
use kurbo::{Affine, Rect};
use peniko::Mix;
use ps_anyrender_vello_cpu::VelloCpuImageRenderer;

#[cfg(not(feature = "multithreading"))]
use anyrender::render_to_buffer;
#[cfg(not(feature = "multithreading"))]
use peniko::{Color, Fill};

const WIDTH: u32 = 200;
const HEIGHT: u32 = 100;
/// The seam sits at the halfway mark, so a blur has to bleed across it.
#[cfg(not(feature = "multithreading"))]
const SEAM: u32 = WIDTH / 2;

#[cfg(not(feature = "multithreading"))]
fn pixel(buffer: &[u8], x: u32, y: u32) -> [u8; 3] {
    let offset = ((y * WIDTH + x) * 4) as usize;
    [buffer[offset], buffer[offset + 1], buffer[offset + 2]]
}

fn blur(std_deviation: f32) -> Arc<Filter> {
    Arc::new(Filter::single(FilterEffect::blur(std_deviation)))
}

/// A hard black/white seam, then a transparent panel over the middle carrying a
/// backdrop blur, all inside `depth` nested clip layers.
///
/// Nothing is drawn inside the panel and its alpha is 1.0, so whatever shows
/// through it is the backdrop and nothing else.
#[cfg(not(feature = "multithreading"))]
fn scene(backdrop: Option<Arc<Filter>>, depth: usize) -> Vec<u8> {
    render_to_buffer::<VelloCpuImageRenderer, _>(
        |scene| {
            let canvas = Rect::new(0.0, 0.0, f64::from(WIDTH), f64::from(HEIGHT));
            for _ in 0..depth {
                scene.push_clip_layer(Affine::IDENTITY, &canvas);
            }

            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::BLACK,
                None,
                &Rect::new(0.0, 0.0, f64::from(SEAM), f64::from(HEIGHT)),
            );
            scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::WHITE,
                None,
                &Rect::new(f64::from(SEAM), 0.0, f64::from(WIDTH), f64::from(HEIGHT)),
            );

            if let Some(backdrop) = backdrop {
                let panel = Rect::new(40.0, 0.0, 160.0, f64::from(HEIGHT));
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

            for _ in 0..depth {
                scene.pop_layer();
            }
        },
        WIDTH,
        HEIGHT,
    )
}

/// A blur reaches across the seam: the pixels either side of it stop being
/// pure black and pure white.
///
#[cfg(not(feature = "multithreading"))]
fn assert_blurred(buffer: &[u8], depth: usize) {
    let left = pixel(buffer, SEAM - 6, HEIGHT / 2);
    let right = pixel(buffer, SEAM + 6, HEIGHT / 2);
    assert!(
        left[0] > 20,
        "at depth {depth} the dark side of the seam is still {left:?}; \
         nothing was blurred into it"
    );
    assert!(
        right[0] < 235,
        "at depth {depth} the light side of the seam is still {right:?}; \
         nothing was blurred into it"
    );
}

#[cfg(not(feature = "multithreading"))]
#[test]
fn a_backdrop_blurs_the_scene_behind_it() {
    let plain = scene(None, 0);
    assert_eq!(pixel(&plain, SEAM - 6, HEIGHT / 2), [0, 0, 0]);
    assert_eq!(pixel(&plain, SEAM + 6, HEIGHT / 2), [255, 255, 255]);

    assert_blurred(&scene(Some(blur(8.0)), 0), 0);
}

/// The case that matters, and the one that was broken.
///
/// Two clip layers is conservative: blitz-paint opens one for the document, one
/// per stacking context and one per element with overflow, so a panel in a page
/// sits far deeper than this.
#[cfg(not(feature = "multithreading"))]
#[test]
fn a_backdrop_blurs_from_inside_nested_layers() {
    for depth in 1..=3 {
        assert_blurred(&scene(Some(blur(8.0)), depth), depth);
    }
}

/// The snapshot closes every open layer and reopens it. If it reopened one too
/// few, the layers below would leak; one too many, and the context would
/// underflow. Either shows up as the painter losing track of its own depth.
#[test]
fn the_snapshot_leaves_the_layer_stack_exactly_as_it_found_it() {
    use anyrender::ImageRenderer;
    let mut renderer = VelloCpuImageRenderer::new(WIDTH, HEIGHT);
    let mut buffer = vec![0; (WIDTH * HEIGHT * 4) as usize];
    renderer.render(
        |scene| {
            let canvas = Rect::new(0.0, 0.0, f64::from(WIDTH), f64::from(HEIGHT));
            scene.push_clip_layer(Affine::IDENTITY, &canvas);
            scene.push_clip_layer(Affine::IDENTITY, &canvas);
            assert_eq!(scene.open_layers(), 2);

            scene.push_layer(
                Mix::Normal,
                1.0,
                Affine::IDENTITY,
                &canvas,
                None,
                Some(blur(4.0)),
            );
            assert_eq!(
                scene.open_layers(),
                3,
                "the backdrop snapshot did not restore the stack it unwound"
            );
            scene.pop_layer();
            scene.pop_layer();
            scene.pop_layer();
            assert_eq!(scene.open_layers(), 0);
        },
        &mut buffer,
    );
}
