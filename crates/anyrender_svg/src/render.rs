// Copyright 2024 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::util;
use anyrender::PaintScene;
use kurbo::{Affine, BezPath};
use peniko::{BlendMode, Color, Compose, Fill};
use usvg::{Node, Path};

pub(crate) fn render_group<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    group: &usvg::Group,
    transform: Affine,
    global_transform: Affine,
    error_handler: &mut F,
) {
    for node in group.children() {
        let transform = transform * util::to_affine(&node.abs_transform());
        match node {
            usvg::Node::Group(g) => {
                let alpha = g.opacity().get();
                let is_fully_opaque = alpha >= 1.0;
                let mix = util::to_mix(g.blend_mode());

                let layer_transform = global_transform * transform;
                let clip_path = g.clip_path();

                #[allow(deprecated)]
                let needs_layer =
                    clip_path.is_some() || mix != peniko::Mix::Normal || !is_fully_opaque;
                if !needs_layer {
                    render_group(scene, g, Affine::IDENTITY, global_transform, error_handler);
                    continue;
                }

                let bounding_box = g.layer_bounding_box();
                let rect = kurbo::Rect::from_origin_size(
                    (bounding_box.x(), bounding_box.y()),
                    (bounding_box.width() as f64, bounding_box.height() as f64),
                );
                scene.push_layer(
                    BlendMode {
                        mix,
                        compose: Compose::SrcOver,
                    },
                    alpha,
                    layer_transform,
                    &rect,
                    None,
                    None,
                );

                if let Some(clip_path) = clip_path {
                    render_clip_path(scene, clip_path, layer_transform, layer_transform, &rect);
                    scene.push_layer(
                        BlendMode {
                            mix: peniko::Mix::Normal,
                            compose: Compose::SrcIn,
                        },
                        1.0,
                        layer_transform,
                        &rect,
                        None,
                        None,
                    );
                }

                render_group(scene, g, Affine::IDENTITY, global_transform, error_handler);

                if clip_path.is_some() {
                    scene.pop_layer();
                }
                if needs_layer {
                    scene.pop_layer();
                }
            }
            usvg::Node::Path(path) => {
                if !path.is_visible() {
                    continue;
                }
                let local_path = util::to_bez_path(path);

                let transform = global_transform * transform;
                match path.paint_order() {
                    usvg::PaintOrder::FillAndStroke => {
                        fill(scene, error_handler, path, transform, &local_path, node);
                        stroke(scene, error_handler, path, transform, &local_path, node);
                    }
                    usvg::PaintOrder::StrokeAndFill => {
                        stroke(scene, error_handler, path, transform, &local_path, node);
                        fill(scene, error_handler, path, transform, &local_path, node);
                    }
                }
            }
            usvg::Node::Image(img) => {
                if !img.is_visible() {
                    continue;
                }
                match img.kind() {
                    usvg::ImageKind::JPEG(_)
                    | usvg::ImageKind::PNG(_)
                    | usvg::ImageKind::GIF(_)
                    | usvg::ImageKind::WEBP(_) => {
                        #[cfg(feature = "image")]
                        {
                            let Ok(decoded_image) = util::decode_raw_raster_image(img.kind())
                            else {
                                error_handler(scene, node);
                                continue;
                            };
                            let image = util::into_image(decoded_image);
                            let image_ts = global_transform * util::to_affine(&img.abs_transform());
                            scene.draw_image(image.as_ref(), image_ts);
                        }

                        #[cfg(not(feature = "image"))]
                        {
                            error_handler(scene, node);
                            continue;
                        }
                    }
                    usvg::ImageKind::SVG(svg) => {
                        render_group(
                            scene,
                            svg.root(),
                            transform,
                            global_transform,
                            error_handler,
                        );
                    }
                }
            }
            usvg::Node::Text(text) => {
                render_group(
                    scene,
                    text.flattened(),
                    transform,
                    global_transform,
                    error_handler,
                );
            }
        }
    }
}

fn render_clip_path<S: PaintScene>(
    scene: &mut S,
    clip_path: &usvg::ClipPath,
    transform: Affine,
    bounds_transform: Affine,
    bounds: &kurbo::Rect,
) {
    if let Some(linked_clip) = clip_path.clip_path() {
        render_clip_path(scene, linked_clip, transform, bounds_transform, bounds);
        scene.push_layer(
            BlendMode {
                mix: peniko::Mix::Normal,
                compose: Compose::SrcIn,
            },
            1.0,
            bounds_transform,
            bounds,
            None,
            None,
        );
    }

    let clip_transform = transform * util::to_affine(&clip_path.transform());
    // Opaque white fills combine by alpha, so overlapping children form a union
    // independently of their winding direction. Each fill still applies its own rule.
    render_clip_group(
        scene,
        clip_path.root(),
        clip_transform,
        bounds_transform,
        bounds,
    );

    if clip_path.clip_path().is_some() {
        scene.pop_layer();
    }
}

fn render_clip_group<S: PaintScene>(
    scene: &mut S,
    group: &usvg::Group,
    transform: Affine,
    bounds_transform: Affine,
    bounds: &kurbo::Rect,
) {
    for node in group.children() {
        let node_transform = transform * util::to_affine(&node.abs_transform());
        match node {
            Node::Path(path) if path.is_visible() => {
                let Some(fill) = path.fill() else {
                    continue;
                };
                let local_path = util::to_bez_path(path);
                let rule = match fill.rule() {
                    usvg::FillRule::NonZero => Fill::NonZero,
                    usvg::FillRule::EvenOdd => Fill::EvenOdd,
                };
                scene.fill(rule, node_transform, Color::WHITE, None, &local_path);
            }
            Node::Group(child_group) => {
                if let Some(child_clip) = child_group.clip_path() {
                    scene.push_layer(
                        BlendMode {
                            mix: peniko::Mix::Normal,
                            compose: Compose::SrcOver,
                        },
                        1.0,
                        bounds_transform,
                        bounds,
                        None,
                        None,
                    );
                    render_clip_group(scene, child_group, transform, bounds_transform, bounds);
                    // Replace the group's temporary mask with the intersection of
                    // its content and the child clip's mask.
                    scene.push_layer(
                        BlendMode {
                            mix: peniko::Mix::Normal,
                            compose: Compose::SrcIn,
                        },
                        1.0,
                        bounds_transform,
                        bounds,
                        None,
                        None,
                    );
                    render_clip_path(scene, child_clip, node_transform, bounds_transform, bounds);
                    scene.pop_layer();
                    scene.pop_layer();
                } else {
                    render_clip_group(scene, child_group, transform, bounds_transform, bounds);
                }
            }
            Node::Text(text) => {
                render_clip_group(scene, text.flattened(), transform, bounds_transform, bounds);
            }
            Node::Path(_) | Node::Image(_) => {}
        }
    }
}

fn fill<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    error_handler: &mut F,
    path: &Path,
    transform: Affine,
    local_path: &BezPath,
    node: &Node,
) {
    if let Some(fill) = &path.fill() {
        if let Some((paint, brush_transform)) = util::to_brush(fill.paint(), fill.opacity()) {
            scene.fill(
                match fill.rule() {
                    usvg::FillRule::NonZero => Fill::NonZero,
                    usvg::FillRule::EvenOdd => Fill::EvenOdd,
                },
                transform,
                paint.as_ref(),
                Some(brush_transform),
                local_path,
            );
        } else {
            error_handler(scene, node);
        }
    }
}

fn stroke<S: PaintScene, F: FnMut(&mut S, &usvg::Node)>(
    scene: &mut S,
    error_handler: &mut F,
    path: &Path,
    transform: Affine,
    local_path: &BezPath,
    node: &Node,
) {
    if let Some(stroke) = &path.stroke() {
        if let Some((paint, brush_transform)) = util::to_brush(stroke.paint(), stroke.opacity()) {
            let conv_stroke = util::to_stroke(stroke);
            scene.stroke(
                &conv_stroke,
                transform,
                paint.as_ref(),
                Some(brush_transform),
                local_path,
            );
        } else {
            error_handler(scene, node);
        }
    }
}
