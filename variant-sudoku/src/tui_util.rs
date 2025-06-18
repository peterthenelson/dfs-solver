use std::marker::PhantomData;

/// Tui implementations can be implemented however they choose, but these are
/// some generic implementations of a collection of useful tasks. They are used
/// to implement the standard Tui impls.
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{self, Direction, Layout, Rect}, style::{Color, Style, Stylize}, symbols::border, text::{Line, Span, Text}, widgets::{Block, Padding, Paragraph}, Frame
};
use crate::{
    constraint::Constraint, core::{BranchOver, Index, Overlay, State, Value}, ranker::Ranker, solver::{DfsSolverView, PuzzleSetter}, sudoku::StdOverlay, tui::{Mode, Pane, TuiState}
};

// TODO: Can we switch to some sort of TryInto thing? All we really need is the
// ability to conditionally do stuff based on whether the overlay can be
// cast to a standard overlay.
pub struct GridConfig<P: PuzzleSetter, const N: usize, const M: usize> {
    pub overlay: StdOverlay<N, M>,
    _marker: PhantomData<P>,
}

impl <P: PuzzleSetter, const N: usize, const M: usize> GridConfig<P, N, M> {
    pub fn new(overlay: StdOverlay<N, M>) -> Self {
        Self { overlay, _marker: PhantomData }
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

pub fn constraint_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<Line<'static>> {
    if let Some(s) = state.solver.constraint().debug_at(state.solver.state(), state.grid_pos) {
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
    to_lines(&*format!("{:?}", state.solver.constraint()))
}

pub fn possible_value_lines<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    _: &TuiState<'a, P>, _: &GridConfig<P, N, M>,
) -> Vec<Line<'static>> {
    vec!["TODO -- not yet implemented".bold().italic().into()]
    // TODO: take a grid_type and do something with it; this is something
    // that works for the GridType::PossibilityHeatmap(ViewBy::Cells) case.
    /*
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
    */
}

pub fn scroll_lines<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>, cfg: &GridConfig<P, N, M>,
) -> Vec<Line<'static>> {
    match state.mode {
        Mode::Readme => readme_lines(),
        Mode::Stack => stack_lines(state),
        Mode::PossibilityHeatmap => possible_value_lines(state, cfg),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewBy { Cell, Row, Col, Box }

/// The relevant heatmap types
/// TODO: Add Score to this
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorBy { Possibilities }

/// The different types of grid display you can render.
#[derive(Debug, PartialEq, Eq)]
pub enum GridType<P: PuzzleSetter> {
    // Just show the puzzle itself, like normal.
    Puzzle,
    // Highlight cells which could be a particular value.
    // Note: ViewBy::Cell not allowed
    HighlightPossible(P::Value, ViewBy),
    // Heatmap of possibilities.
    Heatmap(ViewBy, ColorBy),
}
// Why the hell did derive not succeed in doing this for me?
impl <P: PuzzleSetter> Clone for GridType<P> { fn clone(&self) -> Self { *self } }
impl <P: PuzzleSetter> Copy for GridType<P> {}

fn heatmap_cursor<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    cfg: &GridConfig<P, N, M>,
    grid_pos: Index
) -> (Index, ViewBy) {
    let (mut left, mut upper) = (true, true);
    let [mut r, mut c] = grid_pos;
    if r >= cfg.overlay.rows() {
        upper = false;
        r -= cfg.overlay.rows();
    }
    if c >= cfg.overlay.cols() {
        left = false;
        c -= cfg.overlay.cols();
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

fn grid_val_for_index<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
    grid_type: GridType<P>,
    index: Index,
) -> Option<P::Value> {
    let [r, c] = index;
    match grid_type {
        GridType::Heatmap(ViewBy::Row, _) => Some(P::Value::nth(c)),
        GridType::Heatmap(ViewBy::Col, _) => Some(P::Value::nth(r)),
        GridType::Heatmap(ViewBy::Box, _) => {
            let (_, [br, bc]) = cfg.overlay.to_box_coords([r, c]);
            Some(P::Value::nth(br*cfg.overlay.box_width() + bc))
        },
        GridType::HighlightPossible(_, _) => panic!("TODO: HighlightPossible not implemented!"),
        GridType::Puzzle | GridType::Heatmap(ViewBy::Cell, _) => state.solver.state().get([r, c]),
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
        let [r, c] = match grid_type {
            GridType::Heatmap(ViewBy::Row, _) => [i[0], ord],
            GridType::Heatmap(ViewBy::Col, _) => [ord, i[1]],
            GridType::Heatmap(ViewBy::Box, _) => {
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
        GridType::Heatmap(ViewBy::Row, ColorBy::Possibilities) => 0,
        GridType::Heatmap(ViewBy::Col, ColorBy::Possibilities) => 1,
        GridType::Heatmap(ViewBy::Box, ColorBy::Possibilities) => 2,
        // The remaining types return early.
        GridType::Puzzle => return hm,
        GridType::HighlightPossible(_, _) => panic!("TODO: HightlightPossible not implemented!"),
        GridType::Heatmap(ViewBy::Cell, ColorBy::Possibilities) => {
            if let Some(grid) = state.solver.decision_grid() {
                for r in 0..N {
                    for c in 0..M {
                        if state.solver.state().get([r, c]).is_some() {
                            continue;
                        }
                        hm[r][c] = Some(gen_heatmap_color::<P>(grid.get([r, c]).0.len()))
                    }
                }
                return hm;
            } else {
                return hm;
            }
        },
    };
    if let Some(grid) = state.solver.decision_grid() {
        // Note: We are assuming that all rows have the same size (and same for cols, boxes).
        let mut infos = vec![];
        for p in 0..cfg.overlay.n_partitions(dim) {
            infos.push(state.solver.ranker().region_info(&grid, state.solver.state(), dim, p).unwrap());
        }
        for r in 0..N {
            for c in 0..M {
                let p = cfg.overlay.enclosing_partition([r, c], dim).unwrap();
                let v = grid_val_for_index(state, cfg, grid_type, [r, c]).unwrap();
                let uv = v.to_uval();
                if infos[p].filled.contains(uv) {
                    continue;
                }
                hm[r][c] = Some(gen_heatmap_color::<P>(infos[p].cell_choices.get(uv).len()));
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
        cursor: match grid_type {
            GridType::Heatmap(vb, _) => {
                let (cursor_pos, cursor_vb) = heatmap_cursor(cfg, state.grid_pos);
                if vb == cursor_vb {
                    Some(cursor_pos)
                } else {
                    None
                }
            },
            GridType::HighlightPossible(_, _) => panic!("TODO: HightlightPossible not implemented"),
            GridType::Puzzle => Some(state.grid_pos),
        },
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

pub fn grid_dims<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<P, N, M>,
) -> [usize; 2] {
    match state.mode {
        Mode::PossibilityHeatmap => [cfg.overlay.rows()*2, cfg.overlay.cols()*2],
        _ => [P::State::ROWS, P::State::COLS],
    }
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
        Mode::PossibilityHeatmap => "Heatmap of Possible Values",
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
        Mode::PossibilityHeatmap => {},
        _ => {
            frame.render_widget(
                Paragraph::new(grid_text(state, cfg, GridType::Puzzle)).centered().block(block),
                area,
            );
            return;
        },
    };
    // Otherwise do a 2x2 block of grids
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let (ul, ur, ll, lr) = (
        grid_text(state, cfg, GridType::Heatmap(ViewBy::Cell, ColorBy::Possibilities)),
        grid_text(state, cfg, GridType::Heatmap(ViewBy::Row, ColorBy::Possibilities)),
        grid_text(state, cfg, GridType::Heatmap(ViewBy::Col, ColorBy::Possibilities)),
        grid_text(state, cfg, GridType::Heatmap(ViewBy::Box, ColorBy::Possibilities)),
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
