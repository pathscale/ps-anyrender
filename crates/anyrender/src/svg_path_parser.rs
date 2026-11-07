// Copyright 2018 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! A fork of Kurbo's SVG path parser with finite numbers and bounded arc subdivision.

use core::error::Error;
use core::fmt::{self, Display, Formatter};

use peniko::kurbo::{Arc, BezPath, PathEl, Point, SvgArc};

const MAX_ARC_SEGMENTS: usize = 64;
const MAX_PATH_ELEMENTS: usize = 262_144;

fn point_is_finite(point: Point) -> bool {
    point.x.is_finite() && point.y.is_finite()
}

fn ensure_path_room(
    path: &BezPath,
    additional: usize,
    max_elements: usize,
) -> Result<(), SvgParseError> {
    if additional > max_elements.saturating_sub(path.elements().len()) {
        return Err(SvgParseError::PathElementLimit);
    }
    Ok(())
}

fn append_bounded_arc(
    path: &mut BezPath,
    arc: Arc,
    max_elements: usize,
) -> Result<(), SvgParseError> {
    if !point_is_finite(arc.center)
        || !arc.radii.x.is_finite()
        || !arc.radii.y.is_finite()
        || !arc.start_angle.is_finite()
        || !arc.sweep_angle.is_finite()
        || !arc.x_rotation.is_finite()
    {
        return Err(SvgParseError::Wrong);
    }
    if arc.sweep_angle == 0.0 {
        return Ok(());
    }

    let radius = arc.radii.x.abs().max(arc.radii.y.abs());
    let n_err = (1.1163 * (radius / 0.1)).powf(1.0 / 6.0).max(3.999_999);
    let count = (n_err * arc.sweep_angle.abs() / core::f64::consts::TAU)
        .ceil()
        .clamp(1.0, MAX_ARC_SEGMENTS as f64) as usize;
    ensure_path_room(path, count, max_elements)?;
    let step = arc.sweep_angle / count as f64;
    let arm = (4.0 / 3.0) * (0.25 * step).abs().tan() * arc.sweep_angle.signum();

    fn sample(arc: &Arc, angle: f64) -> peniko::kurbo::Vec2 {
        let (sin, cos) = angle.sin_cos();
        let u = arc.radii.x * cos;
        let v = arc.radii.y * sin;
        let (rotation_sin, rotation_cos) = arc.x_rotation.sin_cos();
        peniko::kurbo::Vec2::new(
            u * rotation_cos - v * rotation_sin,
            u * rotation_sin + v * rotation_cos,
        )
    }

    for index in 0..count {
        let angle0 = arc.start_angle + step * index as f64;
        let angle1 = arc.start_angle + step * (index + 1) as f64;
        let p0 = sample(&arc, angle0);
        let p3 = sample(&arc, angle1);
        let p1 = arc.center + p0 + arm * sample(&arc, angle0 + core::f64::consts::FRAC_PI_2);
        let p2 = arc.center + p3 - arm * sample(&arc, angle1 + core::f64::consts::FRAC_PI_2);
        let p3 = arc.center + p3;
        if !point_is_finite(p1) || !point_is_finite(p2) || !point_is_finite(p3) {
            return Err(SvgParseError::Wrong);
        }
        path.curve_to(p1, p2, p3);
    }
    Ok(())
}

/// Try to parse a bezier path from an SVG path element.
///
/// This is implemented on a best-effort basis, intended for cases where the
/// user controls the source of paths, and is not intended as a replacement
/// for a general, robust SVG parser.
pub(crate) fn parse_svg_path(data: &str) -> Result<BezPath, SvgParseError> {
    parse_svg_path_with_limit(data, MAX_PATH_ELEMENTS)
}

fn parse_svg_path_with_limit(
    data: &str,
    max_elements: usize,
) -> Result<BezPath, SvgParseError> {
    let mut lexer = SvgLexer::new(data);
    let mut path = BezPath::new();
    let mut last_cmd = 0;
    let mut last_ctrl = None;
    let mut first_pt = Point::ORIGIN;
    let mut implicit_moveto = None;
    while let Some(c) = lexer.get_cmd(last_cmd) {
        if c != b'm' && c != b'M' {
            if path.elements().is_empty() {
                return Err(SvgParseError::UninitializedPath);
            }

            if let Some(pt) = implicit_moveto.take() {
                ensure_path_room(&path, 1, max_elements)?;
                path.move_to(pt);
            }
        }
        match c {
            b'm' | b'M' => {
                implicit_moveto = None;
                let pt = lexer.get_maybe_relative(c)?;
                ensure_path_room(&path, 1, max_elements)?;
                path.move_to(pt);
                lexer.last_pt = pt;
                first_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c - (b'M' - b'L');
            }
            b'l' | b'L' => {
                let pt = lexer.get_maybe_relative(c)?;
                ensure_path_room(&path, 1, max_elements)?;
                path.line_to(pt);
                lexer.last_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c;
            }
            b'h' | b'H' => {
                let mut x = lexer.get_number()?;
                lexer.opt_comma();
                if c == b'h' {
                    x += lexer.last_pt.x;
                }
                let pt = Point::new(x, lexer.last_pt.y);
                ensure_path_room(&path, 1, max_elements)?;
                path.line_to(pt);
                lexer.last_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c;
            }
            b'v' | b'V' => {
                let mut y = lexer.get_number()?;
                lexer.opt_comma();
                if c == b'v' {
                    y += lexer.last_pt.y;
                }
                let pt = Point::new(lexer.last_pt.x, y);
                ensure_path_room(&path, 1, max_elements)?;
                path.line_to(pt);
                lexer.last_pt = pt;
                last_ctrl = Some(pt);
                last_cmd = c;
            }
            b'q' | b'Q' => {
                let p1 = lexer.get_maybe_relative(c)?;
                let p2 = lexer.get_maybe_relative(c)?;
                ensure_path_room(&path, 1, max_elements)?;
                path.quad_to(p1, p2);
                last_ctrl = Some(p1);
                lexer.last_pt = p2;
                last_cmd = c;
            }
            b't' | b'T' => {
                let p1 = match last_ctrl {
                    Some(ctrl) => (2.0 * lexer.last_pt.to_vec2() - ctrl.to_vec2()).to_point(),
                    None => lexer.last_pt,
                };
                let p2 = lexer.get_maybe_relative(c)?;
                ensure_path_room(&path, 1, max_elements)?;
                path.quad_to(p1, p2);
                last_ctrl = Some(p1);
                lexer.last_pt = p2;
                last_cmd = c;
            }
            b'c' | b'C' => {
                let p1 = lexer.get_maybe_relative(c)?;
                let p2 = lexer.get_maybe_relative(c)?;
                let p3 = lexer.get_maybe_relative(c)?;
                ensure_path_room(&path, 1, max_elements)?;
                path.curve_to(p1, p2, p3);
                last_ctrl = Some(p2);
                lexer.last_pt = p3;
                last_cmd = c;
            }
            b's' | b'S' => {
                let p1 = match last_ctrl {
                    Some(ctrl) => (2.0 * lexer.last_pt.to_vec2() - ctrl.to_vec2()).to_point(),
                    None => lexer.last_pt,
                };
                let p2 = lexer.get_maybe_relative(c)?;
                let p3 = lexer.get_maybe_relative(c)?;
                ensure_path_room(&path, 1, max_elements)?;
                path.curve_to(p1, p2, p3);
                last_ctrl = Some(p2);
                lexer.last_pt = p3;
                last_cmd = c;
            }
            b'a' | b'A' => {
                let radii = lexer.get_number_pair()?;
                let x_rotation = lexer.get_number()?.rem_euclid(360.0).to_radians();
                lexer.opt_comma();
                let large_arc = lexer.get_flag()?;
                lexer.opt_comma();
                let sweep = lexer.get_flag()?;
                lexer.opt_comma();
                let p = lexer.get_maybe_relative(c)?;
                if !point_is_finite(lexer.last_pt) || !point_is_finite(p) {
                    return Err(SvgParseError::Wrong);
                }
                let svg_arc = SvgArc {
                    from: lexer.last_pt,
                    to: p,
                    radii: radii.to_vec2(),
                    x_rotation,
                    large_arc,
                    sweep,
                };

                match Arc::from_svg_arc(&svg_arc) {
                    Some(arc) => {
                        // TODO: consider making tolerance configurable
                        append_bounded_arc(&mut path, arc, max_elements)?;
                    }
                    None => {
                        ensure_path_room(&path, 1, max_elements)?;
                        path.line_to(p);
                    }
                }

                last_ctrl = Some(p);
                lexer.last_pt = p;
                last_cmd = c;
            }
            b'z' | b'Z' => {
                ensure_path_room(&path, 1, max_elements)?;
                path.close_path();
                lexer.last_pt = first_pt;
                implicit_moveto = Some(first_pt);
            }
            _ => return Err(SvgParseError::UnknownCommand(c as char)),
        }
    }
    for element in path.elements() {
        let finite = match element {
            PathEl::MoveTo(point) | PathEl::LineTo(point) => point_is_finite(*point),
            PathEl::QuadTo(p1, p2) => point_is_finite(*p1) && point_is_finite(*p2),
            PathEl::CurveTo(p1, p2, p3) => {
                point_is_finite(*p1) && point_is_finite(*p2) && point_is_finite(*p3)
            }
            PathEl::ClosePath => true,
        };
        if !finite {
            return Err(SvgParseError::Wrong);
        }
    }
    Ok(path)
}

/// An error which can be returned when parsing an SVG.
#[derive(Debug)]
#[non_exhaustive]
pub(crate) enum SvgParseError {
    /// A number was expected.
    Wrong,
    /// The input string ended while still expecting input.
    UnexpectedEof,
    /// Encountered an unknown command letter.
    UnknownCommand(char),
    /// Encountered a command that precedes expected 'moveto' command.
    UninitializedPath,
    /// The path emitted more elements than the configured limit.
    PathElementLimit,
}

impl Display for SvgParseError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            SvgParseError::Wrong => write!(f, "Unable to parse a number"),
            SvgParseError::UnexpectedEof => write!(f, "Unexpected EOF"),
            SvgParseError::UnknownCommand(letter) => write!(f, "Unknown command, \"{letter}\""),
            SvgParseError::UninitializedPath => {
                write!(f, "Uninitialized path (missing moveto command)")
            }
            SvgParseError::PathElementLimit => {
                write!(f, "SVG path exceeds the element limit")
            }
        }
    }
}

impl Error for SvgParseError {}

struct SvgLexer<'a> {
    data: &'a str,
    ix: usize,
    pub last_pt: Point,
}

impl SvgLexer<'_> {
    fn new(data: &str) -> SvgLexer<'_> {
        SvgLexer {
            data,
            ix: 0,
            last_pt: Point::ORIGIN,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&c) = self.data.as_bytes().get(self.ix) {
            if !(c == b' ' || c == 9 || c == 10 || c == 12 || c == 13) {
                break;
            }
            self.ix += 1;
        }
    }

    fn get_cmd(&mut self, last_cmd: u8) -> Option<u8> {
        self.skip_ws();
        if let Some(c) = self.get_byte() {
            if c.is_ascii_lowercase() || c.is_ascii_uppercase() {
                return Some(c);
            } else if last_cmd != 0 && (c == b'-' || c == b'.' || c.is_ascii_digit()) {
                // Plausible number start
                self.unget();
                return Some(last_cmd);
            } else {
                self.unget();
            }
        }
        None
    }

    fn get_byte(&mut self) -> Option<u8> {
        self.data.as_bytes().get(self.ix).map(|&c| {
            self.ix += 1;
            c
        })
    }

    fn unget(&mut self) {
        self.ix -= 1;
    }

    fn get_number(&mut self) -> Result<f64, SvgParseError> {
        self.skip_ws();
        let start = self.ix;
        let mut c = self.get_byte().ok_or(SvgParseError::UnexpectedEof)?;

        // If first byte is + or - then read the next byte
        if c == b'-' || c == b'+' {
            c = self.get_byte().ok_or(SvgParseError::UnexpectedEof)?;
        }
        if c.is_ascii_alphabetic() {
            return Err(SvgParseError::Wrong);
        }
        self.unget();

        let mut digit_count = 0;
        let mut seen_period = false;
        while let Some(c) = self.get_byte() {
            if c.is_ascii_digit() {
                digit_count += 1;
            } else if c == b'.' && !seen_period {
                seen_period = true;
            } else {
                self.unget();
                break;
            }
        }
        if let Some(c) = self.get_byte() {
            if c == b'e' || c == b'E' {
                let mut c = self.get_byte().ok_or(SvgParseError::Wrong)?;
                if c == b'-' || c == b'+' {
                    c = self.get_byte().ok_or(SvgParseError::Wrong)?;
                }
                if !c.is_ascii_digit() {
                    return Err(SvgParseError::Wrong);
                }
                while let Some(c) = self.get_byte() {
                    if !c.is_ascii_digit() {
                        self.unget();
                        break;
                    }
                }
            } else {
                self.unget();
            }
        }
        if digit_count > 0 {
            let value = self.data[start..self.ix]
                .parse::<f64>()
                .map_err(|_| SvgParseError::Wrong)?;
            if !value.is_finite() {
                return Err(SvgParseError::Wrong);
            }
            Ok(value)
        } else {
            Err(SvgParseError::Wrong)
        }
    }

    fn get_flag(&mut self) -> Result<bool, SvgParseError> {
        self.skip_ws();
        match self.get_byte().ok_or(SvgParseError::UnexpectedEof)? {
            b'0' => Ok(false),
            b'1' => Ok(true),
            _ => Err(SvgParseError::Wrong),
        }
    }

    fn get_number_pair(&mut self) -> Result<Point, SvgParseError> {
        let x = self.get_number()?;
        self.opt_comma();
        let y = self.get_number()?;
        self.opt_comma();
        Ok(Point::new(x, y))
    }

    fn get_maybe_relative(&mut self, cmd: u8) -> Result<Point, SvgParseError> {
        let pt = self.get_number_pair()?;
        if cmd.is_ascii_lowercase() {
            Ok(self.last_pt + pt.to_vec2())
        } else {
            Ok(pt)
        }
    }

    fn opt_comma(&mut self) {
        self.skip_ws();
        if let Some(c) = self.get_byte() {
            if c != b',' {
                self.unget();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_svg_path, parse_svg_path_with_limit, SvgParseError, MAX_PATH_ELEMENTS};

    fn repeated_line_path(line_count: usize) -> String {
        let mut data = String::from("M0 0 L1 1");
        for _ in 1..line_count {
            data.push_str(" 1 1");
        }
        data
    }

    #[test]
    fn emitted_path_element_limit_is_inclusive_with_implicit_repeats() {
        let at_limit = repeated_line_path(MAX_PATH_ELEMENTS - 1);
        assert_eq!(parse_svg_path(&at_limit).unwrap().elements().len(), MAX_PATH_ELEMENTS);

        let over_limit = repeated_line_path(MAX_PATH_ELEMENTS);
        assert!(matches!(
            parse_svg_path(&over_limit),
            Err(SvgParseError::PathElementLimit)
        ));
    }

    #[test]
    fn arc_checks_its_full_emission_against_remaining_room() {
        let path = "M0 0 A10000 10000 0 0 1 10000 10000";
        assert!(matches!(
            parse_svg_path_with_limit(path, 2),
            Err(SvgParseError::PathElementLimit)
        ));
    }

    #[test]
    fn close_path_implicit_move_counts_toward_the_limit() {
        assert!(matches!(
            parse_svg_path_with_limit("M0 0 Z L1 1", 3),
            Err(SvgParseError::PathElementLimit)
        ));
    }

    #[test]
    fn absolute_and_relative_paths_keep_the_same_elements() {
        let absolute = "M1 2 L4 6 Q5 7 8 9 C9 10 11 12 13 14 A10 10 0 0 1 23 24 Z";
        let relative = "m1 2 l3 4 q1 1 4 3 c1 1 3 3 5 5 a10 10 0 0 1 10 10 z";
        let absolute = parse_svg_path(absolute).unwrap();
        let relative = parse_svg_path(relative).unwrap();
        assert_eq!(absolute.elements(), relative.elements());
    }
}
