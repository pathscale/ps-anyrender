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

/// Frames a slot may go unused before it is released.
///
/// Not zero, because the count of boundaries moves with what is on screen: a
/// menu opening and closing must not free the textures it needed and allocate
/// them again a frame later. A little over a second at 120Hz is long enough
/// that ordinary interaction never pays a reallocation, and short enough that
/// leaving a heavy view gives the memory back while the user is still looking
/// at the lighter one.
const IDLE_FRAMES_BEFORE_RELEASE: u32 = 180;

/// Textures reused across frames for snapshots and blur results.
///
/// Allocating these per frame would mean a texture creation and a vello
/// registration per panel per frame, and registration is not free: it inserts
/// into vello's override map and forces an atlas copy. They are keyed by
/// position in the frame rather than by element, which is enough because the
/// same page produces the same sequence of boundaries every frame.
///
/// # Why the pool shrinks
///
/// Each snapshot slot is a full-frame `Rgba8Unorm` texture: 19 MB at 2688x1800.
/// The pool used to grow to the worst frame a session ever rendered and hold
/// that for the life of the process, because the only release was [`Self::clear`]
/// and that runs on suspend. A window resize made it worse, since the resized
/// slot was replaced at its own index while the vector kept its length.
///
/// Measured on AgencyZero: 94 GPU allocations, 898 MB, in a window whose busiest
/// frame wanted 25 layers and two backdrop boundaries.
#[derive(Default)]
pub struct BackdropPool {
    snapshots: Vec<Slot>,
    scratch: Vec<Slot>,
    blurred: Vec<Slot>,
    pipeline: Option<BlurPipeline>,
    /// Highest `boundary`/`job` index reserved during the frame being built.
    /// `None` before the first reservation, which is how a frame that wants no
    /// backdrops at all is told apart from one that wants a single slot 0.
    frame_snapshot_high_water: Option<usize>,
    frame_job_high_water: Option<usize>,
    /// Consecutive frames whose high water mark was below the pool's length.
    idle_frames: u32,
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
        // What this frame actually asked for, so `end_frame` can tell a pool
        // that is merely large from one that is still in use.
        self.frame_snapshot_high_water = Some(
            self.frame_snapshot_high_water
                .map_or(boundary, |high| high.max(boundary)),
        );
        self.frame_job_high_water =
            Some(self.frame_job_high_water.map_or(job, |high| high.max(job)));

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

    /// Close the frame, and give back textures it did not need.
    ///
    /// Called once per frame after the last reservation. A pool that stayed
    /// larger than the frame wanted, for a run of consecutive frames a little
    /// over a second long at 120Hz, is trimmed to that high water mark.
    //
    // The exact count is `IDLE_FRAMES_BEFORE_RELEASE`, which is private, so it
    // is described rather than linked: a public doc comment cannot name it
    // without failing the documentation build.
    ///
    /// The delay is what makes this safe to do at all: boundary counts move
    /// with the page, and trimming on the first small frame would reallocate a
    /// 19 MB texture as soon as the user reopened whatever they just closed.
    /// Waiting means the common case pays nothing and only a real change of
    /// view releases anything.
    pub fn end_frame(
        &mut self,
        renderer: &mut VelloRenderer,
        handles: &mut FxHashMap<ResourceId, ImageData>,
    ) {
        // `None` means no backdrop was reserved at all this frame, so every
        // slot is spare. `Some(n)` means indices 0..=n were used.
        let snapshots_wanted = self.frame_snapshot_high_water.map_or(0, |high| high + 1);
        let jobs_wanted = self.frame_job_high_water.map_or(0, |high| high + 1);
        self.frame_snapshot_high_water = None;
        self.frame_job_high_water = None;

        let oversized = self.snapshots.len() > snapshots_wanted
            || self.blurred.len() > jobs_wanted
            || self.scratch.len() > jobs_wanted;
        match release_decision(oversized, self.idle_frames) {
            Release::Keep { idle_frames } => {
                self.idle_frames = idle_frames;
                return;
            }
            Release::Now => self.idle_frames = 0,
        }

        // Registered textures have to leave vello's override map as well, or
        // the atlas keeps a copy of a texture nothing will ever draw again.
        for slot in self.snapshots.drain(snapshots_wanted..) {
            handles.remove(&slot.id);
            renderer.unregister_texture(slot.image);
        }
        for slot in self.blurred.drain(jobs_wanted..) {
            handles.remove(&slot.id);
            renderer.unregister_texture(slot.image);
        }
        // Scratch is never registered, so dropping it is the whole release.
        self.scratch.truncate(jobs_wanted);
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
        self.frame_snapshot_high_water = None;
        self.frame_job_high_water = None;
        self.idle_frames = 0;
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

/// Whether `end_frame` should release the spare slots yet.
///
/// Split out from [`BackdropPool::end_frame`] because the pool itself owns GPU
/// textures and cannot be built without a device, while this decision is the
/// part that was wrong: the pool used to have no release path at all outside
/// suspend, so it held the largest frame a session ever rendered forever.
#[derive(Debug, PartialEq, Eq)]
enum Release {
    /// Not yet. Carries the idle count to store back.
    Keep { idle_frames: u32 },
    /// Long enough: trim to the frame's high water mark.
    Now,
}

fn release_decision(oversized: bool, idle_frames: u32) -> Release {
    if !oversized {
        // In use at its current size, so the streak restarts.
        return Release::Keep { idle_frames: 0 };
    }
    let idle_frames = idle_frames.saturating_add(1);
    if idle_frames < IDLE_FRAMES_BEFORE_RELEASE {
        Release::Keep { idle_frames }
    } else {
        Release::Now
    }
}

#[cfg(test)]
mod release_tests {
    use super::{IDLE_FRAMES_BEFORE_RELEASE, Release, release_decision};

    /// A pool the frame is still filling must never be trimmed.
    #[test]
    fn a_pool_in_use_is_kept_and_resets_the_streak() {
        assert_eq!(
            release_decision(false, 5),
            Release::Keep { idle_frames: 0 },
            "a frame that wanted every slot must restart the idle count",
        );
    }

    /// The delay is the whole reason this is safe: a menu closing must not free
    /// a 19 MB texture that reopening it needs a frame later.
    #[test]
    fn a_briefly_oversized_pool_is_kept() {
        assert_eq!(release_decision(true, 0), Release::Keep { idle_frames: 1 },);
        assert_eq!(
            release_decision(true, IDLE_FRAMES_BEFORE_RELEASE - 2),
            Release::Keep {
                idle_frames: IDLE_FRAMES_BEFORE_RELEASE - 1
            },
        );
    }

    /// The bug: without this the pool never released anything outside suspend.
    #[test]
    fn a_persistently_oversized_pool_is_released() {
        assert_eq!(
            release_decision(true, IDLE_FRAMES_BEFORE_RELEASE - 1),
            Release::Now,
            "after the full idle run the spare slots must be given back",
        );
    }

    /// One busy frame in the middle of a quiet run defers the release rather
    /// than letting a stale streak trim a pool that is needed again.
    #[test]
    fn a_single_busy_frame_restarts_the_wait() {
        let mut idle = 0;
        for _ in 0..(IDLE_FRAMES_BEFORE_RELEASE - 1) {
            match release_decision(true, idle) {
                Release::Keep { idle_frames } => idle = idle_frames,
                Release::Now => panic!("released before the idle run completed"),
            }
        }
        match release_decision(false, idle) {
            Release::Keep { idle_frames } => idle = idle_frames,
            Release::Now => panic!("a frame in use must not release"),
        }
        assert_eq!(idle, 0);
        assert_eq!(
            release_decision(true, idle),
            Release::Keep { idle_frames: 1 }
        );
    }
}
