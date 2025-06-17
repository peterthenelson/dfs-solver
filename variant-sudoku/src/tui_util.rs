/// Tui implementations can be implemented however they choose, but these are
/// some generic implementations of a collection of useful tasks. They are used
/// to implement the standard Tui impls.
use crossterm::event::{KeyCode, KeyEvent};
use ratatui::{
    layout::{Rect},
    style::{Style, Stylize},
    symbols::border,
    text::{Line, Span, Text},
    widgets::{Block, Padding, Paragraph},
    Frame,
};
use crate::{
    core::{BranchOver, Index, State},
    solver::{DfsSolverView, PuzzleSetter},
    sudoku::{unpack_sval_vals},
    tui::{Mode, Pane, TuiState},
};

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

pub fn to_lines(s: &str) -> Vec<String> {
    s.lines().map(|line| line.to_string()).collect()
}

pub fn readme_lines() -> Vec<String> {
    vec![
        "[N]ext -- Move forward one step".into(),
        "Ctrl+Z -- Undo and/or pop decision".into(),
        "Ctrl+R -- Reset puzzle".into(),
        // TODO: "[P]lay -- Move forward until no longer certain".into(),
        // TODO: Manual step, force backtrack
    ]
}

pub fn stack_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<String> {
    let mut lines = vec![];
    for bp in state.solver.stack() {
        let leader = format!("@step#{} ({})", bp.branch_step, bp.branch_attribution.get_name());
        let mut alt2 = None;
        let alt1 = match &bp.choices {
            BranchOver::Empty => "EMPTY".to_string(),
            BranchOver::Cell(cell, vals, i) => {
                let start = format!("{:?}: ", cell);
                if vals.len() == 1 {
                    format!("{}[{}]", start, vals[0])
                } else {
                    let parts = vals.iter().map(|v| format!("{}", v)).collect::<Vec<_>>();
                    let mut underline = " ".repeat(start.len()+1);
                    let underlen = parts[0..*i].iter().map(|s| s.len()+2).fold(0, |a, b| a+b);
                    underline.push_str(&*"-".repeat(underlen));
                    underline.push_str("^");
                    alt2 = Some(underline);
                    format!("{}[{}]", start, parts.join(", "))
                }
            },
            BranchOver::Value(val, cells, i) => {
                let start = format!("{}: ", val);
                if cells.len() == 1 {
                    format!("{}[{:?}]", start, cells[0])
                } else {
                    let parts = cells.iter().map(|c| format!("{:?}", c)).collect::<Vec<_>>();
                    let mut underline = " ".repeat(start.len()+1);
                    let underlen = parts[0..*i].iter().map(|s| s.len()+2).fold(0, |a, b| a+b);
                    underline.push_str(&*"-".repeat(underlen));
                    underline.push_str("^");
                    alt2 = Some(underline);
                    format!("{}[{}]", start, parts.join(", "))
                }
            },
        };
        lines.push(format!("{} -- {}", leader, alt1));
        if let Some(s) = alt2 {
            lines.push(" ".repeat(leader.len()+4) + &s);
        }
    }
    lines
}

pub fn constraint_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<String> {
    if let Some(s) = state.solver.get_constraint().debug_at(state.solver.get_state(), state.grid_pos) {
        s.lines().map(|line| line.to_string()).collect()
    } else {
        vec!["No constraint information for this cell".to_string()]
    }
}

pub fn constraint_raw_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<String> {
    to_lines(&*format!("{:?}", state.solver.get_constraint()))
}

pub fn possible_value_lines<'a, P: PuzzleSetter<U = u8>, const MIN: u8, const MAX: u8>(state: &TuiState<'a, P>) -> Vec<String> {
    let mut lines;
    if let Some(g) = state.solver.decision_grid() {
        lines = vec![
            format!("Possible Values in cell {:?}:", state.grid_pos),
            format!("{:?}", unpack_sval_vals::<MIN, MAX>(&g.get(state.grid_pos).0)),
            "".to_string(),
            "Features:".to_string(),
        ];
        lines.extend(format!("{}", g.get(state.grid_pos).1)
            .lines()
            .map(|line| line.to_string()));
    } else {
        lines = vec!["No Decision Grid Available".to_string()];
    }
    lines
}

pub fn scroll_lines_generic<'a, P: PuzzleSetter<U = u8>, const MIN: u8, const MAX: u8>(state: &TuiState<'a, P>) -> Vec<String> {
    match state.mode {
        Mode::Readme => readme_lines(),
        Mode::Stack => stack_lines(state),
        Mode::GridCells => possible_value_lines::<P, MIN, MAX>(state),
        Mode::Constraints => constraint_lines::<P>(state),
        Mode::ConstraintsRaw => constraint_raw_lines::<P>(state),
    }
}

pub fn grid_top(seg_len: usize, segs: usize) -> Line<'static> {
    let seg = "─".repeat(seg_len*3);
    let mut pieces: Vec<Span<'_>> = vec![]; 
    pieces.push("┌".into());
    for i in 0..segs {
        pieces.push(seg.clone().into());
        if i+1 < segs {
            pieces.push("┬".into());
        }
    }
    pieces.push("┐".into());
    Line::from(pieces)
}

pub fn grid_bottom(seg_len: usize, segs: usize) -> Line<'static> {
    let seg = "─".repeat(seg_len*3);
    let mut pieces: Vec<Span<'_>> = vec![]; 
    pieces.push("└".into());
    for i in 0..segs {
        pieces.push(seg.clone().into());
        if i+1 < segs {
            pieces.push("┴".into());
        }
    }
    pieces.push("┘".into());
    Line::from(pieces)
}

pub fn grid_crossbar(seg_len: usize, segs: usize) -> Line<'static> {
    let seg = "─".repeat(seg_len*3);
    let mut pieces: Vec<Span<'_>> = vec![]; 
    pieces.push("├".into());
    for i in 0..segs {
        pieces.push(seg.clone().into());
        if i+1 < segs {
            pieces.push("┼".into());
        }
    }
    pieces.push("┤".into());
    Line::from(pieces)
}

pub fn grid_cell<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, index: Index, cursor: Index, most_recent: Option<Index>) -> Span<'static> {
    let mut s: Span<'_> = if let Some(v) = state.solver.get_state().get(index) {
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
    s = match most_recent {
        Some(mr) if mr == index => if state.solver.is_valid() {
            s.green()
        } else {
            s.red()
        },
        _ => s,
    };
    s
}

pub fn grid_line<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, row: usize, seg_len: usize, segs: usize, cursor: Index, most_recent: Option<Index>) -> Line<'static> {
    let mut spans: Vec<Span<'_>> = vec![];
    spans.push("│".into());
    for i in 0..segs {
        for c in 0..seg_len {
            spans.push(grid_cell(state, [row, i*seg_len + c], cursor, most_recent))
        }
        if i+1 < segs {
            spans.push("│".into());
        }
    } 
    spans.push("│".into());
    Line::from(spans)
}

pub fn draw_grid<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, v_seg_len: usize, v_segs: usize, h_seg_len: usize, h_segs: usize, frame: &mut Frame, area: Rect) {
    let is_active = state.active == Pane::Grid;
    let title_text = "Grid";
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
    grid_lines.push(grid_top(h_seg_len, h_segs));
    for i in 0..v_segs {
        for r in 0..v_seg_len {
            grid_lines.push(grid_line(state, r+i*v_seg_len, h_seg_len, h_segs, state.grid_pos, mr))
        }
        if i+1 < v_segs {
            grid_lines.push(grid_crossbar(h_seg_len, h_segs));
        }
    }
    grid_lines.push(grid_bottom(h_seg_len, h_segs));
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
        Mode::GridCells => "Possible Cell Vals",
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
