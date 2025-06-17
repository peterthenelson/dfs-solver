use std::marker::PhantomData;

/// Tui implementations can be implemented however they choose, but these are
/// some generic implementations of a collection of useful tasks. They are used
/// to implement the standard Tui impls.
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::Rect, style::{Style, Stylize}, symbols::border, text::{Line, Span, Text}, widgets::{Block, Padding, Paragraph}, Frame
};
use crate::{
    core::{unpack_values, BranchOver, Index, State, UInt, Value},
    solver::{DfsSolverView, PuzzleSetter},
    sudoku::StandardSudokuOverlay,
    tui::{Mode, Pane, TuiState},
};

pub struct GridConfig<const N: usize, const M: usize, U: UInt, V: Value<U>> {
    pub overlay: StandardSudokuOverlay<N, M>,
    _p_u: PhantomData<U>,
    _p_v: PhantomData<V>,
}

impl <const N: usize, const M: usize, U: UInt, V: Value<U>> GridConfig<N, M, U, V> {
    pub fn new(overlay: StandardSudokuOverlay<N, M>) -> Self {
        Self { overlay, _p_u: PhantomData, _p_v: PhantomData }
    }
    pub fn to_ord(&self, v: &V) -> usize { v.ordinal() }
    pub fn nth(&self, ord: usize) -> V { V::nth(ord) }
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

pub fn possible_value_lines<'a, P: PuzzleSetter<U = u8>, const N: usize, const M: usize>(
    state: &TuiState<'a, P>, _: &GridConfig<N, M, P::U, P::Value>,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>>;
    match state.mode {
        Mode::GridRows | Mode::GridCols | Mode::GridBoxes => {
            return vec!["TODO -- not yet implemented".bold().italic().into()];
        }
        _ => {},
    };
    if let Some(g) = state.solver.decision_grid() {
        let vals = unpack_values::<u8, <<P as PuzzleSetter>::State as State<u8>>::Value>(&g.get(state.grid_pos).0)
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

pub fn scroll_lines<'a, P: PuzzleSetter<U = u8>, const N: usize, const M: usize>(
    state: &TuiState<'a, P>, cfg: &GridConfig<N, M, P::U, P::Value>,
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

pub fn grid_top<P: PuzzleSetter, const N: usize, const M: usize>(cfg: &GridConfig<N, M, P::U, P::Value>) -> Line<'static> {
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

pub fn grid_bottom<P: PuzzleSetter, const N: usize, const M: usize>(cfg: &GridConfig<N, M, P::U, P::Value>) -> Line<'static> {
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

pub fn grid_crossbar<P: PuzzleSetter, const N: usize, const M: usize>(cfg: &GridConfig<N, M, P::U, P::Value>) -> Line<'static> {
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

pub fn grid_cell<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<N, M, P::U, P::Value>,
    index: Index,
    cursor: Index,
    most_recent: Option<Index>,
) -> Span<'static> {
    let val = match state.mode {
        Mode::GridRows => Some(format!("{}", cfg.nth(index[1]))),
        Mode::GridCols => Some(format!("{}", cfg.nth(index[0]))),
        Mode::GridBoxes => {
            let (_, [r, c]) = cfg.overlay.to_box_coords(index);
            Some(format!("{}", cfg.nth(r*cfg.overlay.box_width() + c)))
        },
        _ => state.solver.get_state().get(index).map(|v| format!("{}", v)),
    };
    // TODO: Add heatmap data
    let mut s: Span<'_> = if let Some(v) = val {
        if index == cursor {
            format!("[{}]", v).bold()
        } else {
            format!(" {} ", v).into()
        }
    } else {
        if index == cursor {
            "[ ]".bold()
        } else {
            "   ".into()
        }
    };
    let local_mr = most_recent.map(|[r, c]| {
        let ord = state.solver.get_state().get([r, c]).unwrap().ordinal();
        match state.mode {
            Mode::GridRows => [r, ord],
            Mode::GridCols => [ord, c],
            Mode::GridBoxes => {
                let (b, _) = cfg.overlay.to_box_coords([r, c]);
                let bw = cfg.overlay.box_width();
                cfg.overlay.from_box_coords(b, [ord/bw, ord%bw])
            },
            _ => [r, c],
        }
    });
    s = match local_mr {
        Some(mr) if mr == index => if state.solver.is_valid() {
            s.green()
        } else {
            s.red()
        },
        _ => s,
    };
    s
}

pub fn grid_line<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<N, M, P::U, P::Value>,
    row: usize,
    cursor: Index,
    most_recent: Option<Index>,
) -> Line<'static> {
    let mut spans: Vec<Span<'_>> = vec![];
    spans.push("│".into());
    let (bc, bw) = (cfg.overlay.box_cols(), cfg.overlay.box_width());
    for i in 0..bc {
        for c in 0..bw {
            spans.push(grid_cell(state, cfg, [row, i*bw + c], cursor, most_recent))
        }
        if i+1 < bc {
            spans.push("│".into());
        }
    } 
    spans.push("│".into());
    Line::from(spans)
}

pub fn draw_grid<'a, P: PuzzleSetter, const N: usize, const M: usize>(
    state: &TuiState<'a, P>,
    cfg: &GridConfig<N, M, P::U, P::Value>,
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
    let title = Line::from(if is_active {
        title_text.bold()
    } else {
        title_text.gray()
    });
    let block = Block::bordered()
        .title(title.centered())
        .border_set(if is_active { border::DOUBLE } else { border::PLAIN });
    let mr = state.solver.most_recent_action().map(|(i, _)| i);
    let mut grid_lines = vec![];
    grid_lines.push(grid_top::<P, N, M>(cfg));
    let (br, bh) = (cfg.overlay.box_rows(), cfg.overlay.box_height());
    for i in 0..br {
        for r in 0..bh {
            grid_lines.push(grid_line(state, cfg, r+i*bh, state.grid_pos, mr))
        }
        if i+1 < br {
            grid_lines.push(grid_crossbar::<P, N, M>(cfg));
        }
    }
    grid_lines.push(grid_bottom::<P, N, M>(cfg));
    frame.render_widget(
        Paragraph::new(Text::from(grid_lines)).centered().block(block),
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
