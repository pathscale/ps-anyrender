//! Deciding how many passes a frame's `backdrop-filter` layers cost.
//!
//! A `backdrop-filter` layer needs, as its input, the pixels behind it. No
//! immediate-mode scene API can express that while the scene is still being
//! built, so the only way to produce one is to stop, render what has been drawn
//! so far, filter it, and carry on. That is a render pass, and passes are the
//! unit of cost this module exists to control.
//!
//! The naive rule is one pass per filtered layer. Six glass panels would then
//! cost seven passes for a frame that changes nothing between them. The rule
//! here is one pass per *backdrop root*: panels sitting at the same level, not
//! overlapping each other, all see the same thing behind them, so one snapshot
//! serves all six and the frame costs two passes.
//!
//! # What makes a batch
//!
//! A batch is a maximal run of backdrop ops that can share one snapshot. An op
//! joins the open batch when the region its filter reads from has not been
//! painted into since the snapshot would have been taken. Anything else - a
//! panel drawn over another panel's blur region, a stray fill between them,
//! glass stacked on glass - closes the batch and buys another pass.
//!
//! That last case is deliberate and is an application constraint rather than
//! something to hide in the renderer: glass over glass costs an extra pass, and
//! the counters say so.
//!
//! # Why the planner is separate from any backend
//!
//! Pass count is decided entirely by the shape of the scene, not by the GPU. A
//! planner that is pure data can be driven by a real document and asked "how
//! many passes?" on a machine with no GPU at all, which is what makes the
//! anti-regression assertion runnable in CI. The backend's job is to execute a
//! plan, not to decide one.

use crate::{Filter, Glyph, NormalizedCoord, PaintRef, PaintScene, RenderContext};
use kurbo::{Affine, BezPath, Rect, Shape, Stroke};
use peniko::{BlendMode, Color, Fill, FontData, StyleRef};
use std::sync::Arc;

/// Whether two device-space rectangles share any area.
///
/// Written out rather than taken from [`Rect::intersect`], whose result for
/// disjoint inputs is a rectangle with `x1 < x0` - not an empty one - so the
/// obvious `is_zero_area` test on it reports disjoint rectangles as
/// overlapping. Touching edges are not an overlap: a blur reading up to `x1`
/// and a fill starting at `x1` share no pixel.
fn overlaps(a: Rect, b: Rect) -> bool {
    a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1
}

/// One `backdrop-filter` layer: blur what is behind it, inside `clip`.
///
/// Every geometry here is device space. The planner is fed by a backend that
/// has already applied the layer transform, because the whole point of the
/// occupancy test below is comparing regions from different parts of the tree
/// against each other, and they are only comparable once flattened.
#[derive(Debug, Clone)]
pub struct BackdropOp {
    /// The filter graph to run over the backdrop.
    pub filter: Arc<Filter>,
    /// The shape the filtered backdrop is drawn through.
    pub clip: BezPath,
    /// Device-space bounding box of [`clip`](Self::clip). What the result covers.
    pub bounds: Rect,
    /// What the filter has to *read* to produce [`bounds`](Self::bounds).
    ///
    /// Wider than `bounds` by the filter's expansion, roughly 3 standard
    /// deviations for a gaussian: a blurred pixel at the edge of the panel
    /// samples from outside it. This, not `bounds`, is what the occupancy test
    /// uses, because a fill landing just outside the panel still changes the
    /// pixels inside it.
    pub source: Rect,
}

/// Backdrop ops that can share one snapshot, and so cost one pass between them.
#[derive(Debug, Clone, Default)]
pub struct BackdropBatch {
    pub ops: Vec<BackdropOp>,
}

/// What one painted frame's backdrop layers cost.
///
/// `batches[i]` runs after segment `i` and before segment `i + 1`, so a frame
/// with `n` batches renders `n + 1` segments.
#[derive(Debug, Clone, Default)]
pub struct FramePlan {
    pub batches: Vec<BackdropBatch>,
}

impl FramePlan {
    /// Scene renders this frame costs. Always at least one.
    ///
    /// The number the six-panel anti-regression asserts on: it must be 2 for a
    /// page of non-overlapping glass, not one per panel.
    pub fn render_passes(&self) -> u32 {
        self.batches.len() as u32 + 1
    }

    /// Filtered regions produced this frame, across every batch.
    ///
    /// Distinct from [`render_passes`](Self::render_passes) and it is worth
    /// keeping them apart: batching removes render passes, it does not remove
    /// blurs. Six panels in one batch is 2 render passes and still 6 blurs.
    /// Removing those is what caching them across frames is for, and this is
    /// the number that will have to reach zero on a still frame.
    pub fn blur_passes(&self) -> u32 {
        self.batches
            .iter()
            .map(|batch| batch.ops.len() as u32)
            .sum()
    }

    /// Whether the frame has any backdrop-filtered layer at all.
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

/// Whether a backdrop op could share the open batch's snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// It joined the open batch. No additional render pass.
    SameSegment,
    /// It could not, so the scene is cut here and a pass is added.
    NewSegment,
}

/// How many separate painted regions a batch tracks before it stops trying.
///
/// The occupied region has to be a *list*, not one union rectangle. A page has
/// content at the top and content at the bottom, and their union is the whole
/// page: a single rectangle would report every panel as painted-over and cut
/// the batch at every op, which is the naive one-pass-per-panel behaviour this
/// module exists to avoid.
///
/// A list needs a bound, because it is walked once per op. Sixteen is far more
/// than a real backdrop root holds - occupancy is only recorded at the batch's
/// own depth, so a panel's whole subtree contributes one entry - and past it
/// the batch collapses to a union and gets conservative. Conservative here
/// means an extra render pass, never a stale backdrop.
const MAX_OCCUPIED_REGIONS: usize = 16;

/// The open batch, and what has been painted into its plane since it opened.
#[derive(Debug)]
struct OpenBatch {
    /// Layer depth the batch opened at. Content pushed deeper than this is
    /// bounded by the layer that was pushed, so it is accounted once at the
    /// push instead of once per draw.
    depth: u32,
    /// Everything that could have landed in this plane since the snapshot. A
    /// new op reading from any of it cannot use the snapshot.
    occupied: Vec<Rect>,
    ops: Vec<BackdropOp>,
}

impl OpenBatch {
    fn occupy(&mut self, bounds: Rect) {
        if self.occupied.len() < MAX_OCCUPIED_REGIONS {
            self.occupied.push(bounds);
            return;
        }
        // Collapse into the first slot rather than growing without bound. From
        // here the batch is answering with a bounding box, which can only ever
        // say "painted over" where the truth was "clear".
        let collapsed = self
            .occupied
            .iter()
            .copied()
            .fold(bounds, |acc, rect| acc.union(rect));
        self.occupied.clear();
        self.occupied.push(collapsed);
    }

    fn is_clear(&self, region: Rect) -> bool {
        !self
            .occupied
            .iter()
            .any(|painted| overlaps(*painted, region))
    }
}

/// Turns a painted scene's layer and draw events into a [`FramePlan`].
///
/// Fed by a backend as it walks the scene. Costs nothing at all until the first
/// backdrop op arrives: a page with no glass on it never allocates and never
/// computes a bounding box.
#[derive(Debug, Default)]
pub struct BackdropPlanner {
    depth: u32,
    batches: Vec<BackdropBatch>,
    open: Option<OpenBatch>,
}

impl BackdropPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the planner is currently tracking anything.
    ///
    /// A backend checks this before computing a bounding box to hand to
    /// [`draw`](Self::draw): while it is false, that box would be discarded,
    /// and computing one per draw command is exactly the per-frame CPU cost
    /// this whole design exists to avoid.
    pub fn is_tracking(&self) -> bool {
        self.open.is_some()
    }

    /// Record a layer push whose clip bounds everything drawn until its pop.
    ///
    /// Accounting content at the layer rather than per draw is what keeps this
    /// affordable. A panel's whole subtree is clipped to the layer, so one
    /// bounding box covers all of it however many commands it contains.
    pub fn push_layer(&mut self, bounds: Rect) {
        if let Some(open) = &mut self.open {
            if self.depth <= open.depth {
                open.occupy(bounds);
            }
        }
        self.depth += 1;
    }

    pub fn pop_layer(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Record a draw issued with no layer bounding it.
    ///
    /// Only meaningful at or above the open batch's depth; deeper draws are
    /// already covered by the layer they are inside. Backends should gate the
    /// call on [`is_tracking`](Self::is_tracking) rather than compute a
    /// bounding box unconditionally.
    pub fn draw(&mut self, bounds: Rect) {
        if let Some(open) = &mut self.open {
            if self.depth <= open.depth {
                open.occupy(bounds);
            }
        }
    }

    /// Record a backdrop-filtered layer that is about to be pushed.
    ///
    /// Returns whether the scene has to be cut here. The op's own bounds join
    /// the occupied region, so a second panel overlapping this one gets its own
    /// segment whether or not the backend also reports the layer push.
    pub fn backdrop(&mut self, filter: Arc<Filter>, clip: BezPath, bounds: Rect) -> Boundary {
        let expansion = filter.expansion_rect();
        let source = Rect::new(
            bounds.x0 + expansion.x0,
            bounds.y0 + expansion.y0,
            bounds.x1 + expansion.x1,
            bounds.y1 + expansion.y1,
        );
        let op = BackdropOp {
            filter,
            clip,
            bounds,
            source,
        };

        let can_share = self.open.as_ref().is_some_and(|open| open.is_clear(source));

        if can_share {
            let open = self.open.as_mut().expect("checked above");
            open.occupy(bounds);
            open.ops.push(op);
            return Boundary::SameSegment;
        }

        if let Some(previous) = self.open.take() {
            self.batches.push(BackdropBatch { ops: previous.ops });
        }
        self.open = Some(OpenBatch {
            depth: self.depth,
            occupied: vec![bounds],
            ops: vec![op],
        });
        Boundary::NewSegment
    }

    /// Close the frame and hand back what it costs.
    pub fn finish(mut self) -> FramePlan {
        if let Some(open) = self.open.take() {
            self.batches.push(BackdropBatch { ops: open.ops });
        }
        FramePlan {
            batches: self.batches,
        }
    }
}

/// A [`PaintScene`] that draws nothing and only plans.
///
/// The point of it is that pass count is a property of the scene, not of the
/// GPU. Painting a real document into this answers "how many passes does this
/// page cost?" on a machine with no adapter, no surface and no window, which is
/// what lets the six-panels-two-passes assertion run in CI instead of being a
/// thing someone checks by eye on a laptop.
///
/// It is also the reference for how a backend should feed the planner: the
/// order of the calls in here is the order a backend has to make them in.
#[derive(Debug, Default)]
pub struct PlanningScene {
    planner: BackdropPlanner,
}

impl PlanningScene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn finish(self) -> FramePlan {
        self.planner.finish()
    }

    /// The bounding box of a shape, in device space, or `None` when nothing is
    /// tracking and the answer would be thrown away.
    fn device_bounds(&self, transform: Affine, shape: &impl Shape) -> Option<Rect> {
        self.planner
            .is_tracking()
            .then(|| transform.transform_rect_bbox(shape.bounding_box()))
    }
}

impl RenderContext for PlanningScene {}

impl PaintScene for PlanningScene {
    fn reset(&mut self) {
        self.planner = BackdropPlanner::new();
    }

    fn push_layer(
        &mut self,
        _blend: impl Into<BlendMode>,
        _alpha: f32,
        transform: Affine,
        clip: &impl Shape,
        _filter: Option<Arc<Filter>>,
        backdrop_filter: Option<Arc<Filter>>,
    ) {
        let bounds = transform.transform_rect_bbox(clip.bounding_box());
        if let Some(backdrop_filter) = backdrop_filter {
            let clip = transform * clip.into_path(0.1);
            self.planner.backdrop(backdrop_filter, clip, bounds);
        }
        self.planner.push_layer(bounds);
    }

    fn push_clip_layer(&mut self, transform: Affine, clip: &impl Shape) {
        let bounds = transform.transform_rect_bbox(clip.bounding_box());
        self.planner.push_layer(bounds);
    }

    fn pop_layer(&mut self) {
        self.planner.pop_layer();
    }

    fn stroke<'a>(
        &mut self,
        style: &Stroke,
        transform: Affine,
        _brush: impl Into<PaintRef<'a>>,
        _brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        if let Some(bounds) = self.device_bounds(transform, shape) {
            // A stroke straddles the path by half its width on each side.
            let half = style.width / 2.0;
            self.planner.draw(bounds.inflate(half, half));
        }
    }

    fn fill<'a>(
        &mut self,
        _style: Fill,
        transform: Affine,
        _brush: impl Into<PaintRef<'a>>,
        _brush_transform: Option<Affine>,
        shape: &impl Shape,
    ) {
        if let Some(bounds) = self.device_bounds(transform, shape) {
            self.planner.draw(bounds);
        }
    }

    fn draw_glyphs<'a, 's: 'a>(
        &'s mut self,
        _font: &'a FontData,
        font_size: f32,
        _hint: bool,
        _normalized_coords: &'a [NormalizedCoord],
        _embolden: kurbo::Vec2,
        _style: impl Into<StyleRef<'a>>,
        _brush: impl Into<PaintRef<'a>>,
        _brush_alpha: f32,
        transform: Affine,
        _glyph_transform: Option<Affine>,
        glyphs: impl Iterator<Item = Glyph> + Clone,
    ) {
        if !self.planner.is_tracking() {
            return;
        }
        // Estimated from the run's origins and its size rather than measured
        // from outlines: a glyph's ink stays well inside one em above the
        // baseline and a third of one below it for the scripts in use, and
        // resolving real outlines here would mean shaping the run twice per
        // frame to answer a question whose only consumer is an overlap test.
        // Wrong in the loose direction costs a render pass; measuring costs
        // every frame.
        let size = f64::from(font_size);
        let mut run: Option<Rect> = None;
        for glyph in glyphs {
            let x = f64::from(glyph.x);
            let y = f64::from(glyph.y);
            let cell = Rect::new(x, y - size, x + size, y + size / 3.0);
            run = Some(match run {
                Some(existing) => existing.union(cell),
                None => cell,
            });
        }
        if let Some(run) = run {
            self.planner.draw(transform.transform_rect_bbox(run));
        }
    }

    fn draw_box_shadow(
        &mut self,
        transform: Affine,
        rect: Rect,
        _brush: Color,
        radius: f64,
        std_dev: f64,
    ) {
        if !self.planner.is_tracking() {
            return;
        }
        // A gaussian reaches about three standard deviations, and the corner
        // radius pushes the drawn shape out no further than the rect itself.
        let reach = std_dev * 3.0 + radius;
        self.planner
            .draw(transform.transform_rect_bbox(rect.inflate(reach, reach)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::FilterEffect;
    use kurbo::Shape;

    /// Built the way `blitz-paint` builds it from `filter: blur(<px>)`.
    fn blur(std_dev: f32) -> Arc<Filter> {
        Arc::new(Filter::single(FilterEffect::blur(std_dev)))
    }

    /// A panel `width` wide at `x`, the shape the app's `.rounded-panel` has.
    fn panel(x: f64, width: f64) -> (BezPath, Rect) {
        let bounds = Rect::new(x, 0.0, x + width, 100.0);
        (bounds.into_path(0.1), bounds)
    }

    fn plan_panels(count: usize, pitch: f64, width: f64, std_dev: f32) -> FramePlan {
        let mut planner = BackdropPlanner::new();
        for index in 0..count {
            let (clip, bounds) = panel(index as f64 * pitch, width);
            planner.backdrop(blur(std_dev), clip, bounds);
            // Every panel's own content goes inside its effect layer.
            planner.push_layer(bounds);
            planner.pop_layer();
        }
        planner.finish()
    }

    #[test]
    fn a_frame_with_no_glass_costs_one_pass() {
        let plan = BackdropPlanner::new().finish();
        assert!(plan.is_empty());
        assert_eq!(plan.render_passes(), 1);
        assert_eq!(plan.blur_passes(), 0);
    }

    /// The anti-regression, in miniature: six panels, two passes.
    #[test]
    fn six_separated_panels_share_one_snapshot() {
        // Pitch 200 for 100-wide panels leaves a 100px gap, comfortably more
        // than the ~36px a 12px blur reaches on each side.
        let plan = plan_panels(6, 200.0, 100.0, 12.0);
        assert_eq!(plan.batches.len(), 1, "{plan:?}");
        assert_eq!(plan.batches[0].ops.len(), 6);
        assert_eq!(plan.render_passes(), 2);
        // Batching removes passes, not blurs. Six panels still blur six times.
        assert_eq!(plan.blur_passes(), 6);
    }

    #[test]
    fn overlapping_panels_cannot_share_a_snapshot() {
        // Pitch 50 on 100-wide panels: each panel sits over the previous one.
        let plan = plan_panels(6, 50.0, 100.0, 12.0);
        assert_eq!(plan.batches.len(), 6, "glass over glass costs a pass each");
        assert_eq!(plan.render_passes(), 7);
    }

    /// The gap has to clear the blur's *reach*, not just the panel.
    ///
    /// Two panels 10px apart do not overlap and would pass a bounds-only test,
    /// but a 12px blur reads about 36px past its own edge, so the second
    /// panel's blur samples the first panel's pixels. Sharing the snapshot
    /// there would blur a stale backdrop.
    #[test]
    fn a_gap_smaller_than_the_blur_radius_still_cuts() {
        let touching = plan_panels(2, 110.0, 100.0, 12.0);
        assert_eq!(
            touching.batches.len(),
            2,
            "a 10px gap is inside a 12px blur's reach"
        );

        // The same geometry with a blur small enough to stay inside the gap.
        let clear = plan_panels(2, 110.0, 100.0, 1.0);
        assert_eq!(clear.batches.len(), 1);
    }

    #[test]
    fn a_fill_between_two_panels_cuts_only_if_it_lands_in_the_blur_region() {
        let elsewhere = {
            let mut planner = BackdropPlanner::new();
            let (clip, bounds) = panel(0.0, 100.0);
            planner.backdrop(blur(4.0), clip, bounds);
            planner.draw(Rect::new(1000.0, 0.0, 1100.0, 100.0));
            let (clip, bounds) = panel(200.0, 100.0);
            planner.backdrop(blur(4.0), clip, bounds);
            planner.finish()
        };
        assert_eq!(elsewhere.batches.len(), 1, "a distant fill is irrelevant");

        let underneath = {
            let mut planner = BackdropPlanner::new();
            let (clip, bounds) = panel(0.0, 100.0);
            planner.backdrop(blur(4.0), clip, bounds);
            // Straight across where the second panel is about to go.
            planner.draw(Rect::new(200.0, 0.0, 300.0, 100.0));
            let (clip, bounds) = panel(200.0, 100.0);
            planner.backdrop(blur(4.0), clip, bounds);
            planner.finish()
        };
        assert_eq!(
            underneath.batches.len(),
            2,
            "the second panel's backdrop changed after the snapshot"
        );
    }

    /// Content inside a panel is accounted once, at the layer, not per command.
    ///
    /// This is what makes the planner affordable: a panel with ten thousand
    /// glyphs in it costs one bounding box, and the draws inside it are ignored
    /// because the layer already bounds them.
    #[test]
    fn draws_inside_a_layer_are_covered_by_the_layer() {
        let mut planner = BackdropPlanner::new();
        let (clip, bounds) = panel(0.0, 100.0);
        planner.backdrop(blur(4.0), clip, bounds);
        planner.push_layer(bounds);
        // A draw the layer clips away. Reported at the wrong depth it would
        // occupy the second panel's region and cut the batch for nothing.
        planner.draw(Rect::new(200.0, 0.0, 300.0, 100.0));
        planner.pop_layer();

        let (clip, bounds) = panel(200.0, 100.0);
        planner.backdrop(blur(4.0), clip, bounds);
        let plan = planner.finish();
        assert_eq!(plan.batches.len(), 1, "{plan:?}");
    }

    /// Popping out past the batch's depth must not stop the accounting.
    ///
    /// The panels are siblings, so the walk returns to the container between
    /// them and can go shallower still. A depth test written as equality would
    /// silently drop every draw made out there.
    #[test]
    fn draws_shallower_than_the_batch_still_count() {
        let mut planner = BackdropPlanner::new();
        planner.push_layer(Rect::new(0.0, 0.0, 1000.0, 100.0));
        let (clip, bounds) = panel(0.0, 100.0);
        planner.backdrop(blur(4.0), clip, bounds);
        planner.pop_layer();

        planner.draw(Rect::new(200.0, 0.0, 300.0, 100.0));

        planner.push_layer(Rect::new(0.0, 0.0, 1000.0, 100.0));
        let (clip, bounds) = panel(200.0, 100.0);
        planner.backdrop(blur(4.0), clip, bounds);
        let plan = planner.finish();
        assert_eq!(plan.batches.len(), 2, "{plan:?}");
    }

    /// Past [`MAX_OCCUPIED_REGIONS`] the batch answers with a bounding box.
    ///
    /// Worth asserting rather than leaving as a comment, because the failure it
    /// guards against is the silent kind: the collapse must lose passes, never
    /// correctness. Twenty scattered fills that leave the second panel's region
    /// clear still cut the batch once the list has collapsed, and that is the
    /// intended trade.
    #[test]
    fn a_batch_gets_conservative_once_it_is_tracking_too_many_regions() {
        let mut planner = BackdropPlanner::new();
        let (clip, bounds) = panel(0.0, 100.0);
        planner.backdrop(blur(4.0), clip, bounds);
        // All far away, none of them near the second panel.
        for index in 0..MAX_OCCUPIED_REGIONS + 4 {
            let x = 1000.0 + index as f64 * 10.0;
            planner.draw(Rect::new(x, 0.0, x + 5.0, 100.0));
        }
        let (clip, bounds) = panel(200.0, 100.0);
        planner.backdrop(blur(4.0), clip, bounds);
        let plan = planner.finish();
        assert_eq!(
            plan.batches.len(),
            2,
            "a collapsed occupancy list spans the gap, so the batch cuts"
        );
    }

    #[test]
    fn the_source_region_is_the_panel_grown_by_the_filter() {
        let mut planner = BackdropPlanner::new();
        let (clip, bounds) = panel(100.0, 100.0);
        planner.backdrop(blur(10.0), clip, bounds);
        let plan = planner.finish();
        let op = &plan.batches[0].ops[0];
        assert_eq!(op.bounds, bounds);
        assert!(
            op.source.x0 < bounds.x0 && op.source.x1 > bounds.x1,
            "the read region must grow past the panel, got {:?}",
            op.source
        );
    }
}
