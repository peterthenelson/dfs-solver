use std::marker::PhantomData;

/// Tui implementations can be implemented however they choose, but these are
/// some generic implementations of a collection of useful tasks. They are used
/// to implement the standard Tui impls.
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::Rect, style::{Color, Style, Stylize}, symbols::border, text::{Line, Span, Text}, widgets::{Block, Padding, Paragraph}, Frame
};
use crate::{
    core::{empty_map, empty_set, unpack_values, BranchOver, Index, State, Value},
    solver::{DfsSolverView, PuzzleSetter},
    sudoku::{Overlay, StandardSudokuOverlay},
    tui::{Mode, Pane, TuiState},
};

pub struct GridConfig<P: PuzzleSetter, const N: usize, const M: usize> {
    pub overlay: StandardSudokuOverlay<N, M>,
    _p_p: PhantomData<P>,
}

impl <P: PuzzleSetter, const N: usize, const M: usize> GridConfig<P, N, M> {
    pub fn new(overlay: StandardSudokuOverlay<N, M>) -> Self {
        Self { overlay, _p_p: PhantomData }
    }
    pub fn to_ord(&self, v: &P::Value) -> usize { v.ordinal() }
    pub fn nth(&self, ord: usize) -> P::Value { P::Value::nth(ord) }
}

pub fn grid_wasd<'a, P: PuzzleSetter>(state: &mut TuiState<'a, P>, key_event: KeyEvent) -> bool {
    let [r, c] = state.grid_pos;
    match key_event.code {
        KeyCode::Char('w') => if r > 0 {
            state.grid_pos = [r-1, c];
        },
        KeyCode::Char('s') => if r+1 < P::State::ROWS {
            state.grid_pos = [r+1, c];
        },
        KeyCode::Char('a') => if c > 0 {
            state.grid_pos = [r, c-1];
        },
        KeyCode::Char('d') => if c+1 < P::State::COLS {
            state.grid_pos = [r, c+1];
        },
        _ => return false,
    }
    true
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
        Line::from(vec!["[N]".blue(), "ext -- Move forward one step".into()]),
        Line::from(vec!["Ctrl+Z".blue(), " -- Undo and/or pop decision".into()]),
        Line::from(vec!["Ctrl+R".blue(), " -- Reset puzzle".into()]),
        // TODO: "[P]lay -- Move forward until no longer certain".into(),
        // TODO: Manual step, force backtrack
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
        let attr: Span<'_> = format!("({})", bp.branch_attribution.get_name()).cyan();
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

pub fn constraint_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<Line<'static>> {
    if let Some(s) = state.solver.get_constraint().debug_at(state.solver.get_state(), state.grid_pos) {
        s.lines()
            .map(|line| Line::from(if line.chars().nth(0) == Some(' ') {
                    line.to_string().into()
                } else {
                    line.to_string().cyan()
                }))
            .collect::<Vec<Line>>()
    } else {
        vec!["No constraint information for this cell".italic().into()]
    }
}

pub fn constraint_raw_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<Line<'static>> {
    to_lines(&*format!("{:?}", state.solver.get_constraint()))
}

pub fn possible_value_lines<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>, _: &GridConfig<P, N, M>,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>>;
    match state.mode {
        Mode::GridRows | Mode::GridCols | Mode::GridBoxes => {
            return vec!["TODO -- not yet implemented".bold().italic().into()];
        }
        _ => {},
    };
    if let Some(g) = state.solver.decision_grid() {
        let vals = unpack_values::<P::Value>(&g.get(state.grid_pos).0)
            .into_iter().map(|v| v.to_string()).collect::<Vec<String>>().join(", ");
        lines = vec![
            Line::from(vec!["Possible values in cell ".italic(), format!("{:?}:", state.grid_pos).blue()]),
            format!("{{{}}}", vals).green().into(),
            "".into(),
            "Features:".italic().into(),
        ];
        lines.extend(to_lines(&*format!("{}", g.get(state.grid_pos).1)).into_iter().map(|l| l.cyan()))
    } else {
        lines = vec!["No Decision Grid Available".italic().into()];
    }
    lines
}

pub fn scroll_lines<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>, cfg: &GridConfig<P, N, M>,
) -> Vec<Line<'static>> {
    match state.mode {
        Mode::Readme => readme_lines(),
        Mode::Stack => stack_lines(state),
        Mode::GridCells | Mode::GridRows | Mode::GridCols | Mode::GridBoxes => {
            possible_value_lines(state, cfg)
        },
        Mode::Constraints => constraint_lines::<P>(state),
        Mode::ConstraintsRaw => constraint_raw_lines::<P>(state),
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

/// The relevant grid partition types
#[derive(Debug, Clone, Copy)]
pub enum ViewBy { Cell, Row, Col, Box }

/// The different types of grid display you can render.
#[derive(Debug)]
pub enum GridType<P: PuzzleSetter> {
    // Just show the puzzle itself, like normal.
    Puzzle,
    // Highlight cells which could be a particular value.
    // Note: ViewBy::Cell not allowed
    HighlightPossible(P::Value, ViewBy),
    // Heatmap of possibilities.
    PossibilityHeatmap(ViewBy),
    // TODO: Add RankHeatmap which works the same but with a different color
    // scheme
}
// Why the hell did derive not succeed in doing this for me?
impl <P: PuzzleSetter> Clone for GridType<P> { fn clone(&self) -> Self { *self } }
impl <P: PuzzleSetter> Copy for GridType<P> {}

fn grid_val_for_index<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    grid_type: GridType<P>,
    index: Index,
) -> Option<P::Value> {
    let [r, c] = index;
    match grid_type {
        GridType::PossibilityHeatmap(ViewBy::Row) => Some(cfg.nth(c)),
        GridType::PossibilityHeatmap(ViewBy::Col) => Some(cfg.nth(r)),
        GridType::PossibilityHeatmap(ViewBy::Box) => {
            let (_, [br, bc]) = cfg.overlay.to_box_coords([r, c]);
            Some(cfg.nth(br*cfg.overlay.box_width() + bc))
        },
        GridType::HighlightPossible(_, _) => panic!("TODO: HighlightPossible not implemented!"),
        GridType::Puzzle | GridType::PossibilityHeatmap(ViewBy::Cell) => state.solver.get_state().get([r, c]),
    }
}

fn gen_val<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    grid_type: GridType<P>,
) -> [[Option<P::Value>; M]; N] {
    let mut vm = [[None; M]; N];
    for r in 0..N {
        for c in 0..M {
            vm[r][c] = grid_val_for_index(state, cfg, grid_type, [r, c]);
        }
    }
    vm
}

fn gen_heatmap_color<'a, P: PuzzleSetter>(n_possibilities: usize) -> Color {
    let max = P::Value::cardinality();
    let ratio = n_possibilities as f32 / max as f32;
    let red = (255.0 * ratio) as u8;
    let green = (255.0 * (1.0 - ratio)) as u8;
    Color::Rgb(red, green, 0)
}

fn gen_fg<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    grid_type: GridType<P>,
) -> [[Option<Color>; M]; N] {
    let mut hm = [[None; M]; N];
    if let GridType::HighlightPossible(_, _) = grid_type {
        panic!("TODO: HighlightPossible not implemented!");
    }
    if let Some((i, v)) = state.solver.most_recent_action() {
        let ord = v.ordinal();
        let [r, c] = match state.mode {
            Mode::GridRows => [i[0], ord],
            Mode::GridCols => [ord, i[1]],
            Mode::GridBoxes => {
                let (b, _) = cfg.overlay.to_box_coords(i);
                let bw = cfg.overlay.box_width();
                cfg.overlay.from_box_coords(b, [ord/bw, ord%bw])
            },
            _ => i,
        };
        hm[r][c] = Some(if state.solver.is_valid() { Color::Green } else { Color::Red });
    }
    hm
}

fn gen_bg<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    grid_type: GridType<P>,
) -> [[Option<Color>; M]; N] {
    let mut hm = [[None; M]; N];
    let dim = match grid_type {
        // These 3 cases are all handled the same, indexed by dim.
        GridType::PossibilityHeatmap(ViewBy::Row) => 0,
        GridType::PossibilityHeatmap(ViewBy::Col) => 1,
        GridType::PossibilityHeatmap(ViewBy::Box) => 2,
        // The remaining types return early.
        GridType::Puzzle => return hm,
        GridType::HighlightPossible(_, _) => panic!("TODO: HightlightPossible not implemented!"),
        GridType::PossibilityHeatmap(ViewBy::Cell) => if let Some(grid) = state.solver.decision_grid() {
            for r in 0..N {
                for c in 0..M {
                    if state.solver.get_state().get([r, c]).is_some() {
                        continue;
                    }
                    hm[r][c] = Some(gen_heatmap_color::<P>(grid.get([r, c]).0.len()))
                }
            }
            return hm;
        } else {
            return hm;
        },
    };
    if let Some(grid) = state.solver.decision_grid() {
        // Assumes that all rows have the same size (and same for cols, boxes).
        let partition_size = cfg.overlay.partition_size(dim, 0);
        let mut filled = vec![empty_set::<P::Value>(); partition_size];
        let mut alternatives = vec![empty_map::<P::Value, Vec<_>>(); partition_size];
        for p in 0..cfg.overlay.n_partitions(dim) {
            for index in cfg.overlay.partition_iter(dim, p) {
                if let Some(val) = state.solver.get_state().get(index) {
                    filled[p].insert(val.to_uval());
                    continue;
                }
                let g = grid.get(index);
                for uv in g.0.iter() {
                    alternatives[p].get_mut(uv).push(index);
                }
            }
        }
        for r in 0..N {
            for c in 0..M {
                let p = cfg.overlay.enclosing_partition([r, c], dim).unwrap();
                let v = grid_val_for_index(state, cfg, grid_type, [r, c]).unwrap();
                let uv = v.to_uval();
                if filled[p].contains(uv) {
                    continue;
                }
                hm[r][c] = Some(gen_heatmap_color::<P>(alternatives[p].get(uv).len()));
            }
        }
    }
    hm
}

pub fn grid_styled_values<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    grid_type: GridType<P>,
) -> GridStyledValues<P, N, M> {
    GridStyledValues::<P, N, M> {
        // TODO: Make the cursor work with 4-way grid display
        cursor: Some(state.grid_pos),
        vals: gen_val(state, cfg, grid_type),
        fg: gen_fg(state, cfg, grid_type),
        bg: gen_bg(state, cfg, grid_type),
    }
}

pub fn grid_top<P: PuzzleSetter, const N: usize, const M: usize>(cfg: &GridConfig<P, N, M>) -> Line<'static> {
    let (bc, bw) = (cfg.overlay.box_cols(), cfg.overlay.box_width());
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
}

pub fn grid_bottom<P: PuzzleSetter, const N: usize, const M: usize>(cfg: &GridConfig<P, N, M>) -> Line<'static> {
    let (bc, bw) = (cfg.overlay.box_cols(), cfg.overlay.box_width());
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
}

pub fn grid_crossbar<P: PuzzleSetter, const N: usize, const M: usize>(cfg: &GridConfig<P, N, M>) -> Line<'static> {
    let (bc, bw) = (cfg.overlay.box_cols(), cfg.overlay.box_width());
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
    cfg: &GridConfig<P, N, M>,
    svals: &GridStyledValues<P, N, M>,
    row: usize,
) -> Line<'static> {
    let mut spans: Vec<Span<'_>> = vec![];
    spans.push("│".into());
    let (bc, bw) = (cfg.overlay.box_cols(), cfg.overlay.box_width());
    for i in 0..bc {
        for c in 0..bw {
            spans.push(grid_cell(svals, [row, i*bw + c]))
        }
        if i+1 < bc {
            spans.push("│".into());
        }
    } 
    spans.push("│".into());
    Line::from(spans)
}

pub fn grid_text<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    grid_type: GridType<P>,
) -> Text<'static> {
    let svals = grid_styled_values(state, cfg, grid_type);
    let mut lines = vec![];
    lines.push(grid_top::<P, N, M>(cfg));
    let (br, bh) = (cfg.overlay.box_rows(), cfg.overlay.box_height());
    for i in 0..br {
        for r in 0..bh {
            lines.push(grid_line(cfg, &svals, r+i*bh));
        }
        if i+1 < br {
            lines.push(grid_crossbar::<P, N, M>(cfg));
        }
    }
    lines.push(grid_bottom::<P, N, M>(cfg));
    Text::from(lines)
}

pub fn draw_grid<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    frame: &mut Frame,
    area: Rect,
) {
    let is_active = state.active == Pane::Grid;
    let title_text = match state.mode {
        Mode::Constraints => "Focus Cell to See Constraints",
        Mode::GridCells => "Cell Possibility Heatmap",
        Mode::GridRows => "Row Value Possibility Heatmap",
        Mode::GridCols => "Col Value Possibility Heatmap",
        Mode::GridBoxes => "Box Value Possibility Heatmap",
        _ => "Puzzle State",
    };
    let grid_type = match state.mode {
        Mode::GridCells => GridType::PossibilityHeatmap(ViewBy::Cell),
        Mode::GridRows => GridType::PossibilityHeatmap(ViewBy::Row),
        Mode::GridCols => GridType::PossibilityHeatmap(ViewBy::Col),
        Mode::GridBoxes => GridType::PossibilityHeatmap(ViewBy::Box),
        _ => GridType::Puzzle,
    };
    let title = Line::from(if is_active {
        title_text.bold()
    } else {
        title_text.gray()
    });
    let block = Block::bordered()
        .title(title.centered())
        .border_set(if is_active { border::DOUBLE } else { border::PLAIN });
    frame.render_widget(
        Paragraph::new(grid_text(state, cfg, grid_type)).centered().block(block),
        area,
    );
}

pub fn draw_text_area<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
    let is_active = state.active == Pane::TextArea;
    let title_text = match state.mode {
        Mode::Readme => "Debugger Hotkeys",
        Mode::Stack => "Decision Stack",
        Mode::Constraints => "Constraints for Cell",
        Mode::ConstraintsRaw => "Full Constraint Dump",
        Mode::GridCells => "Possible Vals for Cell",
        Mode::GridRows => "Possible Cells for Row/Val",
        Mode::GridCols => "Possible Cells for Col/Val",
        Mode::GridBoxes => "Possible Cells for Box/Val",
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
