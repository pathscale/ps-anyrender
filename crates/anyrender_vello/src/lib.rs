//! A [`vello`] backend for the [`anyrender`] 2D drawing abstraction

// `chunks_exact_to_as_chunks` is new in clippy 1.98 and suggests
// `as_chunks_mut::<4>()` where `blur.rs` packs its uniform words. Not taken:
// this workspace builds on an MSRV of 1.92, where that method does not exist,
// so the suggestion trades a lint for a broken MSRV job. `unknown_lints` goes
// with it because the lint does not exist before 1.98, and there the allow is
// itself a warning that `-D warnings` turns into a failure.
#![allow(unknown_lints)]
#![allow(clippy::chunks_exact_to_as_chunks)]

mod backdrop;
mod blur;
#[cfg(not(target_arch = "wasm32"))]
mod image_renderer;
mod scene;
mod window_renderer;

pub use backdrop::BackdropPool;
#[cfg(not(target_arch = "wasm32"))]
pub use image_renderer::VelloImageRenderer;
pub use scene::VelloScenePainter;
pub use wgpu_context::DeviceHandle;
pub use window_renderer::{VelloRendererOptions, VelloWindowRenderer};

pub use wgpu;

use std::num::NonZeroUsize;

#[cfg(target_os = "macos")]
const DEFAULT_THREADS: Option<NonZeroUsize> = NonZeroUsize::new(1);
#[cfg(not(target_os = "macos"))]
const DEFAULT_THREADS: Option<NonZeroUsize> = None;
