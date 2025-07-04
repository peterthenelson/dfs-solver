/// Tui implementations can be implemented however they choose, but these are
/// some generic implementations of a collection of useful tasks. They are used
/// to implement the standard Tui impls.
use std::{f64, fmt::Display};
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{self, Direction, Layout, Rect}, style::{Color, Style, Stylize}, symbols::border, text::{Line, Span, Text}, widgets::{Block, Padding, Paragraph}, Frame
};
use crate::{
    color_util::color_lerp, constraint::{Constraint, MultiConstraint}, core::{BranchOver, Index, Key, Overlay, RegionLayer, VMap, VSet, Value, BOXES_LAYER, COLS_LAYER, ROWS_LAYER}, ranker::Ranker, solver::{DfsSolverView, PuzzleSetter}, sudoku::StdOverlay, tui::{Mode, Pane, TuiState}
};

pub trait ConstraintSplitter<P: PuzzleSetter> {
    fn as_multi(constraint: &P::Constraint) -> Option<&MultiConstraint<P::Value, P::Overlay>>;
}
pub struct NullConstraintSplitter;
impl <P: PuzzleSetter> ConstraintSplitter<P> for NullConstraintSplitter {
    fn as_multi(_: &P::Constraint) -> Option<&MultiConstraint<P::Value, P::Overlay>> {
        None
    }
}
impl <V: Value, O: Overlay, P: PuzzleSetter<Value = V, Overlay = O, Constraint = MultiConstraint<V, O>>>
ConstraintSplitter<P> for P {
    fn as_multi(constraint: &P::Constraint) -> Option<&MultiConstraint<P::Value, P::Overlay>> {
        Some(constraint)
    }
}

pub fn grid_wasd<'a, P: PuzzleSetter>(state: &mut TuiState<'a, P>, key_event: KeyEvent) -> bool {
    let [r, c] = state.grid_pos;
    match key_event.code {
        KeyCode::Char('w') => if r > 0 {
            state.grid_pos = [r-1, c];
        },
        KeyCode::Char('s') => if r+1 < state.grid_dims[0] {
            state.grid_pos = [r+1, c];
        },
        KeyCode::Char('a') => if c > 0 {
            state.grid_pos = [r, c-1];
        },
        KeyCode::Char('d') => if c+1 < state.grid_dims[1] {
            state.grid_pos = [r, c+1];
        },
        _ => return false,
    }
    true
}

pub fn constraint_c<'a, P: PuzzleSetter, CS: ConstraintSplitter<P>>(
    state: &mut TuiState<'a, P>,
    key_event: KeyEvent,
) -> bool {
    match state.mode {
        Mode::Constraints | Mode::ConstraintsRaw => {},
        _ => return false,
    };
    match key_event.code {
        KeyCode::Char('c') => {
            state.constraint_index = if let Some(m) = CS::as_multi(state.solver.constraint()) {
                if let Some(i) = state.constraint_index {
                    if i+1 < m.num_constraints() {
                        // Increase the index if possible
                        Some(i+1)
                    } else {
                        // Wraps back around to unselected
                        None
                    }
                } else {
                    // Unselected goes to constraint [0]
                    Some(0)
                }
            } else {
                None
            };
            true
        },
        _ => false,
    }
}

pub fn text_area_ws<'a, P: PuzzleSetter>(state: &mut TuiState<'a, P>, key_event: KeyEvent) -> bool {
    match key_event.code {
        KeyCode::Char('w') => if state.scroll_pos > 0 {
            state.scroll_pos -= 1;
        },
        KeyCode::Char('s') => if state.scroll_pos+1 < state.scroll_lines.len() {
            state.scroll_pos += 1;
        },
        _ => return false,
    }
    true
}

pub fn to_lines(s: &str) -> Vec<Line<'static>> {
    s.lines().map(|line| Line::from(line.to_string())).collect()
}

pub fn readme_lines() -> Vec<Line<'static>> {
    vec![
        Line::from(vec!["Ctrl+C".blue(), " -- Exit the debugger".into()]),
        Line::from(vec!["[N]".blue(), "ext -- Move forward one step".into()]),
        Line::from(vec!["Ctrl+Z".blue(), " -- Undo and/or pop decision".into()]),
        Line::from(vec!["[J]".blue(), "ump -- Move forward to next non-trivial decision".into()]),
        Line::from(vec!["Ctrl+R".blue(), " -- Reset puzzle".into()]),
        Line::from(vec!["[P]".blue(), "lay/Pause -- Move forward until pressed again".into()]),
        // TODO: Manual step, force backtrack, breakpoints.
    ]
}

pub fn stack_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = vec![];
    for bp in state.solver.stack() {
        let step: Span<'_> = format!("@step#{} ", bp.branch_step).italic();
        let step_len = step.content.len();
        let mut alt2: Option<Span<'_>> = None;
        let alt1: Vec<Span<'_>> = match &bp.choices {
            BranchOver::Empty => vec!["EMPTY".yellow()],
            BranchOver::Cell(cell, vals, i) => {
                let start = format!("{:?} = ", cell);
                if vals.len() == 1 {
                    vec![start.blue(), format!("{{{}}}", vals[0]).green()]
                } else {
                    let parts = vals.iter().map(|v| format!("{}", v)).collect::<Vec<_>>();
                    let mut underline = " ".repeat(start.len()+1);
                    let underlen = parts[0..*i].iter().map(|s| s.len()+2).fold(0, |a, b| a+b);
                    underline.push_str(&*"-".repeat(underlen));
                    underline.push_str("^");
                    alt2 = Some(underline.gray());
                    vec![start.blue(), format!("{{{}}}", parts.join(", ")).green()]
                }
            },
            BranchOver::Value(val, cells, i) => {
                let start = format!("{} = ", val);
                if cells.len() == 1 {
                    vec![start.green(), format!("{{{:?}}}", cells[0]).blue()]
                } else {
                    let parts = cells.iter().map(|c| format!("{:?}", c)).collect::<Vec<_>>();
                    let mut underline = " ".repeat(start.len()+1);
                    let underlen = parts[0..*i].iter().map(|s| s.len()+2).fold(0, |a, b| a+b);
                    underline.push_str(&*"-".repeat(underlen));
                    underline.push_str("^");
                    alt2 = Some(underline.gray());
                    vec![start.green(), format!("{{{}}}", parts.join(", ")).blue()]
                }
            },
        };
        let attr: Span<'_> = format!("({})", bp.branch_attribution.name()).cyan();
        let mut first_line = vec![step];
        first_line.extend(alt1);
        first_line.push("  ".into());
        first_line.push(attr);
        lines.push(Line::from(first_line));
        if let Some(s) = alt2 {
            lines.push(Line::from(vec![" ".repeat(step_len).into(), s]));
        }
    }
    lines
}

pub fn constraint_lines<'a, P: PuzzleSetter, CS: ConstraintSplitter<P>>(
    state: &TuiState<'a, P>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let (name, debug_str) = if let Some((m, i)) = CS::as_multi(state.solver.constraint()).zip(state.constraint_index) {
        let constraint = m.constraint(i);
        (
            format!("Constraint[{}]", i),
            constraint.debug_at(state.solver.state(), state.grid_pos),
        )
    } else {
        (
            "Constraint[*]".to_string(),
            state.solver.constraint().debug_at(state.solver.state(), state.grid_pos),
        )
    };
    lines.push(Line::from(vec![name.into(), " -- ".into(), "[C]".blue(), " to change focused constraint".italic()]));
    lines.push(Line::from(""));
    lines.extend(if let Some(s) = debug_str {
        s.lines()
            .map(|line| Line::from(if line.chars().nth(0) == Some(' ') {
                    line.to_string().into()
                } else {
                    line.to_string().cyan()
                }))
            .collect::<Vec<Line>>()
    } else {
        vec!["No constraint information for this cell".italic().into()]
    });
    lines
}

pub fn constraint_raw_lines<'a, P: PuzzleSetter, CS: ConstraintSplitter<P>>(
    state: &TuiState<'a, P>,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some((m, i)) = CS::as_multi(state.solver.constraint()).zip(state.constraint_index) {
        let constraint = m.constraint(i);
        lines.push(Line::from(vec![
            format!("Constraint[{}]", i).into(), " -- ".into(),
            "[C]".blue(), " to change focused constraint".italic(),
        ]));
        lines.push(Line::from(""));
        lines.extend(to_lines(&*format!("{:?}", constraint)));

    } else {
        lines.push(Line::from(vec![
            "Constraint[*]".into(), " -- ".into(),
            "[C]".blue(), " to change focused constraint".italic(),
        ]));
        lines.push(Line::from(""));
        lines.extend(to_lines(&*format!("{:?}", state.solver.constraint())));
    }
    lines
}

pub fn possible_value_lines<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>
) -> Vec<Line<'static>> {
    if state.solver.ranking_info().is_none() {
        return vec!["No RankingInfo Available".italic().into()];
    }
    let ranking = state.solver.ranking_info().as_ref().unwrap();
    let (cursor, vb) = heatmap_cursor::<N, M>(state.grid_pos);
    let layer = match vb {
        // This case returns early.
        ViewBy::Cell => {
            let vals = ranking.cells().get(cursor).0
                .iter().map(|v| v.to_string()).collect::<Vec<String>>().join(", ");
            let mut lines = vec![
                Line::from(vec!["Possible values in cell ".into(), format!("{:?}:", state.grid_pos).blue()]).italic(),
                format!("{{{}}}", vals).green().into(),
                "".into(),
                "Features:".italic().into(),
            ];
            lines.extend(
                to_lines(&*format!("{}", ranking.cells().get(state.grid_pos).1))
                    .into_iter().map(|l| l.cyan()),
            );
            return lines;
        },
        // These 3 cases are all handled the same, indexed by layer.
        vb => vb.region_layer().unwrap(),
    };
    let at = val_at_index::<P, N, M>(state, vb, cursor);
    match (at.val, at.region_index) {
        (Some(v), Some(p)) => {
            let info = ranking.region_info(
                state.solver.state(), layer, p,
            );
            let cells = info.cell_choices.get(&v).iter()
                .map(|c| format!("{:?}", c))
                .collect::<Vec<_>>().join(", ");
            let mut lines = vec![
                Line::from(vec![
                    format!("Possible cells for {} #{}'s ", vb, p+1).into(),
                    v.to_string().green(),
                    ":".into(),
                ]).italic(),
                format!("{{{}}}", cells).blue().into(),
                "".into(),
                "Features:".italic().into(),
            ];
            lines.extend(
                to_lines(&*format!("{}", info.feature_vecs.get(&v)))
                    .into_iter().map(|l| l.cyan()),
            );
            lines
        },
        _ => vec!["Overlay could not derive a value from the cursor".italic().into()],
    }
}

pub fn score_lines<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>
) -> Vec<Line<'static>> {
    if state.solver.ranking_info().is_none() {
        return vec!["No RankingInfo Available".italic().into()];
    }
    let ranking = state.solver.ranking_info().as_ref().unwrap();
    let (cursor, vb) = heatmap_cursor::<N, M>(state.grid_pos);
    let layer = match vb {
        // This case returns early.
        ViewBy::Cell => {
            if let Some(_) = state.solver.state().get(cursor) {
                return vec![];
            }
            let fv = ranking.cells().get(state.grid_pos).1;
            if fv.try_scored().is_err() {
                // This can occur during extended backtracking
                return vec![];
            }
            let mut lines = vec![
                Line::from(vec!["Score for cell ".into(), format!("{:?}:", state.grid_pos).blue()]).italic(),
                format!("{}", fv.try_scored().unwrap().score()).magenta().into(),
                "".into(),
                "Features:".italic().into(),
            ];
            lines.extend(
                to_lines(&*format!("{}", fv))
                    .into_iter().map(|l| l.cyan()),
            );
            return lines;
        },
        // These 3 cases are all handled the same, indexed by layer.
        vb => vb.region_layer().unwrap(),
    };
    let at = val_at_index::<P, N, M>(state, vb, cursor);
    match (at.val, at.region_index) {
        (Some(v), Some(p)) => {
            let mut info = ranking.region_info(
                state.solver.state(), layer, p,
            );
            if info.filled.contains(&v) {
                return vec![];
            }
            state.solver.ranker().score_region_info(&mut info);
            let fv = info.feature_vecs.get(&v);
            let mut lines = vec![
                Line::from(vec![
                    format!("Score for {} #{}'s ", vb, p+1).into(),
                    v.to_string().green(),
                    ":".into(),
                ]).italic(),
                format!("{}", fv.try_scored().unwrap().score()).magenta().into(),
                "".into(),
                "Features:".italic().into(),
            ];
            lines.extend(
                to_lines(&*format!("{}", fv))
                    .into_iter().map(|l| l.cyan()),
            );
            lines
        },
        _ => vec!["Overlay could not derive a value from the cursor".italic().into()],
    }
}

pub fn scroll_lines<'a, P: PuzzleSetter, const N: usize, const M: usize, CS: ConstraintSplitter<P>>(
    state: &TuiState<'a, P>,
) -> Vec<Line<'static>> {
    match state.mode {
        Mode::Readme => readme_lines(),
        Mode::Stack => stack_lines(state),
        Mode::PossibilityHeatmap => possible_value_lines::<P, N, M>(state),
        Mode::ScoreHeatmap => score_lines::<P, N, M>(state),
        Mode::Constraints => constraint_lines::<P, CS>(state),
        Mode::ConstraintsRaw => constraint_raw_lines::<P, CS>(state),
    }
}

/// Concrete information about what values should be rendered in what cells,
/// what color they should be, how they should be highlighted, etc. when
/// rendering a grid.
pub struct GridStyledValues<P: PuzzleSetter, const N: usize, const M: usize> {
    cursor: Option<Index>,
    vals: [[Option<P::Value>; M]; N],
    fg: [[Option<Color>; M]; N],
    bg: [[Option<Color>; M]; N],
}

/// The relevant ways to view the grid (as a grid of cells with values or as a
/// grid of values in some region layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewBy { Cell, Row, Col, Box }
impl ViewBy {
    pub fn region_layer(&self) -> Option<Key<RegionLayer>> {
        match self {
            ViewBy::Row => Some(ROWS_LAYER),
            ViewBy::Col => Some(COLS_LAYER),
            ViewBy::Box => Some(BOXES_LAYER),
            ViewBy::Cell => None,
        }
    }
}
impl Display for ViewBy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self {
            ViewBy::Cell => "Cell",
            ViewBy::Row => "Row",
            ViewBy::Col => "Col",
            ViewBy::Box => "Box",
        })
    }
}

/// The relevant heatmap types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorBy { Possibilities, Score }

/// The different types of grid display you can render.
#[derive(Debug, PartialEq, Eq)]
pub enum GridType<P: PuzzleSetter> {
    // Just show the puzzle itself, like normal.
    Puzzle,
    // Highlight cells based on the constraints.
    Constraints,
    // Highlight cells which could be a particular value.
    // Note: The usize is the index in the relevant region layer
    HighlightPossible(P::Value, Key<RegionLayer>, usize),
    // Heatmap of possibilities.
    Heatmap(ViewBy, ColorBy),
}
// Why the hell did derive not succeed in doing this for me?
impl <P: PuzzleSetter> Clone for GridType<P> { fn clone(&self) -> Self { *self } }
impl <P: PuzzleSetter> Copy for GridType<P> {}
impl <P: PuzzleSetter> GridType<P> {
    pub fn view_by(&self) -> ViewBy {
        match self {
            GridType::HighlightPossible(_, _, _) => ViewBy::Cell,
            GridType::Heatmap(vb, _) => *vb,
            GridType::Constraints | GridType::Puzzle => ViewBy::Cell,
        }
    }
}

fn heatmap_cursor<const N: usize, const M: usize>(
    grid_pos: Index
) -> (Index, ViewBy) {
    let (mut left, mut upper) = (true, true);
    let [mut r, mut c] = grid_pos;
    if r >= N {
        upper = false;
        r -= N;
    }
    if c >= M {
        left = false;
        c -= M;
    }
    (
        [r, c], 
        match (left, upper) {
            (true, true) => ViewBy::Cell,
            (false, true) => ViewBy::Row,
            (true, false) => ViewBy::Col,
            (false, false) => ViewBy::Box,
        },
    )
}

pub struct AtIndex<V: Value> {
    val: Option<V>,
    region_index: Option<usize>,
}

fn val_at_index<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    view_by: ViewBy,
    index: Index,
) -> AtIndex<P::Value> {
    let [r, c] = index;
    let empty = AtIndex { val: None, region_index: None };
    let overlay = state.solver.state().overlay();
    match view_by {
        // We don't actually need a StdOverlay per se, so long as the overlay
        // has the relevant ROWS/COLS/BOXES layer.
        ViewBy::Row => overlay.enclosing_region_and_offset(ROWS_LAYER, index).map(|(ri, ro)|
            AtIndex {
                val: Some(P::Value::nth(ro)),
                region_index: Some(ri),
            }
        ).unwrap_or(empty),
        ViewBy::Col => overlay.enclosing_region_and_offset(COLS_LAYER, index).map(|(ci, co)|
            AtIndex {
                val: Some(P::Value::nth(co)),
                region_index: Some(ci),
            }
        ).unwrap_or(empty),
        ViewBy::Box => overlay.enclosing_region_and_offset(BOXES_LAYER, index).map(|(bi, bo)|
            AtIndex {
                val: Some(P::Value::nth(bo)),
                region_index: Some(bi),
            }
        ).unwrap_or(empty),
        ViewBy::Cell => AtIndex {
            val: state.solver.state().get([r, c]),
            region_index: None,
        },
    }
}

fn gen_val<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    grid_type: GridType<P>,
) -> [[Option<P::Value>; M]; N] {
    let mut vm = [[None; M]; N];
    if let GridType::HighlightPossible(v, _, _) = grid_type {
        for r in 0..N {
            for c in 0..M {
                if let Some(v) = state.solver.state().get([r, c]) {
                    vm[r][c] = Some(v);
                } else if let Some(ranking) = &state.solver.ranking_info() {
                    if ranking.cells().get([r, c]).0.contains(&v) {
                        vm[r][c] = Some(v)
                    }
                }
            }
        }
    } else {
        for r in 0..N {
            for c in 0..M {
                vm[r][c] = val_at_index::<P, N, M>(state, grid_type.view_by(), [r, c]).val;
            }
        }
    }
    vm
}

fn gen_possibilities_color<'a, P: PuzzleSetter>(n_possibilities: usize) -> Color {
    let max = P::Value::cardinality();
    let ratio = n_possibilities as f32 / max as f32;
    let (r, g, b) = color_lerp((0, 255, 0), (255, 0, 0), ratio);
    Color::Rgb(r, g, b)
}

fn gen_score_color<'a, P: PuzzleSetter>(score: f64, score_range: (f64, f64)) -> Color {
    let ratio = ((score - score_range.0) / (score_range.1 - score_range.0)) as f32;
    let (r, g, b) = color_lerp((0, 0, 255), (255, 255, 0), ratio);
    Color::Rgb(r, g, b)
}

fn gen_fg<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    grid_type: GridType<P>,
) -> [[Option<Color>; M]; N] {
    let mut hm = [[None; M]; N];
    let overlay = state.solver.state().overlay();
    if let Some((i, v)) = state.solver.most_recent_action() {
        let ord = v.ordinal();
        if let Some([r, c]) = match grid_type {
            // We don't actually need a StdOverlay per se, so long as the overlay
            // has the relevant ROWS/COLS/BOXES layer.
            GridType::Heatmap(ViewBy::Row, _) => overlay.enclosing_region_and_offset(ROWS_LAYER, i).and_then(|(ri, _)|
                overlay.nth_in_region(ROWS_LAYER, ri, ord)
            ),
            GridType::Heatmap(ViewBy::Col, _) => overlay.enclosing_region_and_offset(COLS_LAYER, i).and_then(|(ci, _)|
                overlay.nth_in_region(COLS_LAYER, ci, ord)
            ),
            GridType::Heatmap(ViewBy::Box, _) => overlay.enclosing_region_and_offset(BOXES_LAYER, i).and_then(|(bi, _)|
                overlay.nth_in_region(BOXES_LAYER, bi, ord)
            ),
            _ => Some(i),
        } {
            hm[r][c] = Some(if state.solver.is_valid() { Color::Green } else { Color::Red });
        }
    }
    hm
}

fn gen_bg<'a, P: PuzzleSetter, const N: usize, const M: usize, CS: ConstraintSplitter<P>>(
    state: &TuiState<'a, P>,
    grid_type: GridType<P>,
) -> [[Option<Color>; M]; N] {
    let mut hm = [[None; M]; N];
    let overlay = state.solver.state().overlay();
    let (layer, cb) = match grid_type {
        // These first several types return early.
        GridType::Puzzle => return hm,
        GridType::Constraints => {
            for r in 0..N {
                for c in 0..M {
                    let hl = if let Some((m, i)) = CS::as_multi(state.solver.constraint()).zip(state.constraint_index) {
                        m.constraint(i).debug_highlight(state.solver.state(), [r, c])
                    } else {
                        state.solver.constraint().debug_highlight(state.solver.state(), [r, c])
                    };
                    hm[r][c] = hl.map(|(r, g, b)| Color::Rgb(r, g, b));
                }
            }
            return hm;
        },
        GridType::HighlightPossible(v, layer, p) => {
            if let Some(ranking) = state.solver.ranking_info().as_ref() {
                for r in 0..N {
                    for c in 0..M {
                        if ranking.cells().get([r, c]).0.contains(&v) {
                            hm[r][c] = Some(
                                if overlay.enclosing_region_and_offset(layer, [r, c])
                                    .map_or(false, |(ep, _)| ep == p) {
                                    Color::Blue
                                } else {
                                    Color::Cyan
                                }
                            );
                        }
                    }
                }
            }
            return hm;
        },
        GridType::Heatmap(ViewBy::Cell, cb) => {
            if let Some(ranking) = state.solver.ranking_info() {
                let mut ranking = ranking.clone();
                state.solver.ranker().ensure_scored(&mut ranking, state.solver.state());
                let score_range = if let Some(r) = ranking.score_range() {
                    r
                } else {
                    // The puzzle must be full if this is the case
                    return hm;
                };
                for r in 0..N {
                    for c in 0..M {
                        if state.solver.state().get([r, c]).is_some() {
                            continue;
                        }
                        let g = ranking.cells().get([r, c]);
                        hm[r][c] = match cb {
                            ColorBy::Possibilities => {
                                Some(gen_possibilities_color::<P>(g.0.len()))
                            },
                            ColorBy::Score => {
                                g.1.try_scored().ok().map(|scored|
                                    gen_score_color::<P>(scored.score(), score_range)
                                )
                            },
                        };
                    }
                }
                return hm;
            } else {
                return hm;
            }
        },
        // These 3 cases are all handled the same, indexed by layer.
        GridType::Heatmap(vb, cb) => (vb.region_layer().unwrap(), cb),
    };
    if let Some(ranking) = state.solver.ranking_info() {
        let mut ranking = ranking.clone();
        state.solver.ranker().ensure_scored(&mut ranking, state.solver.state());
        let score_range = if let Some(r) = ranking.score_range() {
            r
        } else {
            // The puzzle must be full if this is the case
            return hm;
        };
        // Note: We are assuming that all rows have the same size (and same for cols, boxes).
        let mut infos = vec![];
        for p in 0..overlay.regions_in_layer(layer) {
            infos.push(ranking.region_info(state.solver.state(), layer, p));
        }
        for r in 0..N {
            for c in 0..M {
                let (p, _) = overlay.enclosing_region_and_offset(layer, [r, c]).unwrap();
                let v = val_at_index::<P, N, M>(state, grid_type.view_by(), [r, c]).val.unwrap();
                if infos[p].filled.contains(&v) {
                    continue;
                }
                hm[r][c] = Some(match cb {
                    ColorBy::Possibilities => {
                        let n_possibilities = infos[p].cell_choices.get(&v).len();
                        gen_possibilities_color::<P>(n_possibilities)
                    },
                    ColorBy::Score => {
                        state.solver.ranker().score_region_info(&mut infos[p]);
                        gen_score_color::<P>(
                            infos[p].feature_vecs.get(&v).try_scored().unwrap().score(),
                            score_range,
                        )
                    },
                });
            }
        }
    }
    hm
}

pub fn grid_styled_values<'a, P: PuzzleSetter, const N: usize, const M: usize, CS: ConstraintSplitter<P>>(
    state: &TuiState<'a, P>,
    grid_type: GridType<P>,
) -> GridStyledValues<P, N, M> {
    GridStyledValues::<P, N, M> {
        cursor: match grid_type {
            GridType::Heatmap(vb, _) => {
                let (cursor_pos, cursor_vb) = heatmap_cursor::<N, M>(state.grid_pos);
                if vb == cursor_vb {
                    Some(cursor_pos)
                } else {
                    None
                }
            },
            GridType::HighlightPossible(_, _, _) => None,
            GridType::Constraints => Some(state.grid_pos),
            GridType::Puzzle => Some(state.grid_pos),
        },
        vals: gen_val(state, grid_type),
        fg: gen_fg(state, grid_type),
        bg: gen_bg::<P, N, M, CS>(state, grid_type),
    }
}

pub fn grid_top<P: PuzzleSetter, const N: usize, const M: usize>(so: &Option<StdOverlay<N, M>>) -> Line<'static> {
    if let Some(overlay) = so {
        let (bc, bw) = (overlay.box_cols(), overlay.box_width());
        let seg = "─".repeat(bw*3);
        let mut pieces: Vec<Span<'_>> = vec![]; 
        pieces.push("┌".into());
        for i in 0..bc {
            pieces.push(seg.clone().into());
            if i+1 < bc {
                pieces.push("┬".into());
            }
        }
        pieces.push("┐".into());
        Line::from(pieces)
    } else {
        Line::from("┌".to_string() + &*"─".repeat(M*3) + "┐")
    }
}

pub fn grid_bottom<P: PuzzleSetter, const N: usize, const M: usize>(so: &Option<StdOverlay<N, M>>) -> Line<'static> {
    if let Some(overlay) = so {
        let (bc, bw) = (overlay.box_cols(), overlay.box_width());
        let seg = "─".repeat(bw*3);
        let mut pieces: Vec<Span<'_>> = vec![]; 
        pieces.push("└".into());
        for i in 0..bc {
            pieces.push(seg.clone().into());
            if i+1 < bc {
                pieces.push("┴".into());
            }
        }
        pieces.push("┘".into());
        Line::from(pieces)
    } else {
        Line::from("└".to_string() + &*"─".repeat(M*3) + "┘")
    }
}

pub fn grid_crossbar<P: PuzzleSetter, const N: usize, const M: usize>(overlay: &StdOverlay<N, M>) -> Line<'static> {
    let (bc, bw) = (overlay.box_cols(), overlay.box_width());
    let seg = "─".repeat(bw*3);
    let mut pieces: Vec<Span<'_>> = vec![]; 
    pieces.push("├".into());
    for i in 0..bc {
        pieces.push(seg.clone().into());
        if i+1 < bc {
            pieces.push("┼".into());
        }
    }
    pieces.push("┤".into());
    Line::from(pieces)
}

fn grid_cell<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    svals: &GridStyledValues<P, N, M>,
    index: Index,
) -> Span<'static> {
    let val = svals.vals[index[0]][index[1]];
    let mut s: Span<'_> = if let Some(v) = val {
        if svals.cursor.is_some() && index == svals.cursor.unwrap() {
            format!("[{}]", v).bold()
        } else {
            format!(" {} ", v).into()
        }
    } else {
        if svals.cursor.is_some() && index == svals.cursor.unwrap() {
            "[ ]".bold()
        } else {
            "   ".into()
        }
    };
    s = if let Some(c) = svals.fg[index[0]][index[1]] {
        s.fg(c)
    } else {
        s
    };
    s = if let Some(c) = svals.bg[index[0]][index[1]] {
        s.bg(c)
    } else {
        s
    };
    s
}

fn grid_line<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    so: &Option<StdOverlay<N, M>>,
    svals: &GridStyledValues<P, N, M>,
    row: usize,
) -> Line<'static> {
    let mut spans: Vec<Span<'_>> = vec![];
    spans.push("│".into());
    if let Some(overlay) = so {
        let (bc, bw) = (overlay.box_cols(), overlay.box_width());
        for i in 0..bc {
            for c in 0..bw {
                spans.push(grid_cell(svals, [row, i*bw + c]))
            }
            if i+1 < bc {
                spans.push("│".into());
            }
        } 
    } else {
        for col in 0..M {
            spans.push(grid_cell(svals, [row, col]));
        }
    }
    spans.push("│".into());
    Line::from(spans)
}

pub fn grid_text<'a, P: PuzzleSetter, const N: usize, const M: usize, CS: ConstraintSplitter<P>>(
    state: &TuiState<'a, P>,
    so: &Option<StdOverlay<N, M>>,
    grid_type: GridType<P>,
) -> Text<'static> {
    let svals = grid_styled_values::<P, N, M, CS>(state, grid_type);
    let mut lines = vec![];
    lines.push(grid_top::<P, N, M>(so));
    if let Some(overlay) = so {
        let (br, bh) = (overlay.box_rows(), overlay.box_height());
        for i in 0..br {
            for r in 0..bh {
                lines.push(grid_line(so, &svals, r+i*bh));
            }
            if i+1 < br {
                lines.push(grid_crossbar::<P, N, M>(overlay));
            }
        }
    } else {
        for row in 0..N {
            lines.push(grid_line(so, &svals, row));
        }
    }
    lines.push(grid_bottom::<P, N, M>(so));
    Text::from(lines)
}

pub fn grid_dims<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
) -> [usize; 2] {
    let overlay = state.solver.state().overlay();
    let (rows, cols) = overlay.grid_dims();
    let layers = overlay.region_layers();
    let supports_rcb = layers.contains(&ROWS_LAYER) && layers.contains(&COLS_LAYER) && layers.contains(&BOXES_LAYER);
    match state.mode {
        Mode::PossibilityHeatmap if supports_rcb => [rows*2, cols*2],
        Mode::ScoreHeatmap if supports_rcb => [rows*2, cols*2],
        _ => [rows, cols],
    }
}

fn upper_left_type<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>, cb: ColorBy,
) -> GridType<P> {
    let fallback = GridType::Heatmap(ViewBy::Cell, cb);
    let (cursor, vb) = heatmap_cursor::<N, M>(state.grid_pos);
    if vb == ViewBy::Cell {
        return fallback;
    }
    vb.region_layer().and_then(|layer| {
        let at = val_at_index::<P, N, M>(state, vb, cursor);
        at.val.and_then(|v|
            at.region_index.and_then(|p|
                Some(GridType::HighlightPossible(v, layer, p))
            )
        )
    }).unwrap_or(fallback)
}

pub fn draw_grid<'a, P: PuzzleSetter, const N: usize, const M: usize, CS: ConstraintSplitter<P>>(
    state: &TuiState<'a, P>,
    so: &Option<StdOverlay<N, M>>,
    frame: &mut Frame,
    area: Rect,
) {
    let is_active = state.active == Pane::Grid;
    let title_text = match state.mode {
        Mode::Constraints => "Focus Cell to See Constraints",
        Mode::PossibilityHeatmap => "Heatmap of Possible Values",
        Mode::ScoreHeatmap => "Heatmap of Scores",
        _ => "Puzzle State",
    };
    let title = Line::from(if is_active {
        title_text.bold()
    } else {
        title_text.gray()
    });
    let block = Block::bordered()
        .title(title.centered())
        .border_set(if is_active { border::DOUBLE } else { border::PLAIN });
    match state.mode {
        Mode::PossibilityHeatmap | Mode::ScoreHeatmap => {},
        Mode::Constraints => {
            frame.render_widget(
                Paragraph::new(
                    grid_text::<P, N, M, CS>(state, so, GridType::Constraints)
                ).centered().block(block),
                area,
            );
            return;
        },
        _ => {
            frame.render_widget(
                Paragraph::new(
                    grid_text::<P, N, M, CS>(state, so, GridType::Puzzle)
                ).centered().block(block),
                area,
            );
            return;
        },
    };
    // Otherwise do a 2x2 block of grids
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let cb = match &state.mode {
        Mode::ScoreHeatmap => ColorBy::Score,
        Mode::PossibilityHeatmap => ColorBy::Possibilities,
        m => panic!("upper_left_type doesn't make sense in mode: {:?}", m),
    };
    let (ul, ur, ll, lr) = (
        grid_text::<P, N, M, CS>(state, so, upper_left_type::<P, N, M>(state, cb)),
        grid_text::<P, N, M, CS>(state, so, GridType::Heatmap(ViewBy::Row, cb)),
        grid_text::<P, N, M, CS>(state, so, GridType::Heatmap(ViewBy::Col, cb)),
        grid_text::<P, N, M, CS>(state, so, GridType::Heatmap(ViewBy::Box, cb)),
    );
    let grid_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([layout::Constraint::Length(ul.height() as u16), layout::Constraint::Min(0)])
        .split(inner);
    let grid_upper = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([layout::Constraint::Percentage(50), layout::Constraint::Percentage(50)])
        .split(grid_rows[0]);
    let grid_lower = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([layout::Constraint::Percentage(50), layout::Constraint::Percentage(50)])
        .split(grid_rows[1]);
    frame.render_widget(Paragraph::new(ul).right_aligned(), grid_upper[0]);
    frame.render_widget(Paragraph::new(ur).left_aligned(), grid_upper[1]);
    frame.render_widget(Paragraph::new(ll).right_aligned(), grid_lower[0]);
    frame.render_widget(Paragraph::new(lr).left_aligned(), grid_lower[1]);
}

pub fn draw_text_area<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
    let is_active = state.active == Pane::TextArea;
    let title_text = match state.mode {
        Mode::Readme => "Debugger Hotkeys",
        Mode::Stack => "Decision Stack",
        Mode::Constraints => "Constraints for Cell",
        Mode::ConstraintsRaw => "Full Constraint Dump",
        Mode::PossibilityHeatmap => "Possible Vals/Cells",
        Mode::ScoreHeatmap => "Ranker Scores",
    };
    let title = Line::from(if is_active {
        title_text.bold()
    } else {
        title_text.gray()
    });
    let block = Block::bordered()
        .title(title.centered())
        .padding(Padding::left(2))
        .border_set(if is_active { border::DOUBLE } else { border::PLAIN })
        .border_style(if is_active { Style::new() } else { Style::new().gray() });
    let text = Text::from(
        state.scroll_lines[state.scroll_pos..state.scroll_lines.len()]
        .iter().map(|line| Line::from(line.clone())).collect::<Vec<_>>()
    );
    frame.render_widget(
        Paragraph::new(text).left_aligned().block(block),
        area,
    );
}
