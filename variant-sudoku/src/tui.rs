use std::{env, io, time::Duration};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::{Direction, Layout, Rect}, style::{Style, Stylize}, symbols::border, text::{Line, Span, Text}, widgets::{Block, Padding, Paragraph}, DefaultTerminal, Frame
};
use strum::EnumCount;
use crate::{
    core::{ConstraintResult, Index, State}, debug::{DbgObserver, Sample}, solver::{DfsSolver, DfsSolverState, DfsSolverView, FindFirstSolution, PuzzleSetter, StepObserver}, sudoku::{unpack_sval_vals, EightStd, FourStd, NineStd, SixStd}
};

/// Solves the puzzle in command-line mode. No interactivity, but a StepObserver
/// can be passed in to periodically print out or save debug information. The
/// solution will be printed at the end.
pub fn solve_cli<P: PuzzleSetter, D: StepObserver<P::U, P::State>>(mut observer: D) {
    let (mut s, r, mut c) = P::setup();
    let mut finder = FindFirstSolution::new(&mut s, &r, &mut c, Some(&mut observer));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{:?}", maybe_solution.expect("No solution found!").get_state());
}

/// Solves the puzzle in the interactive debugger. This 
pub fn solve_interactive<P: PuzzleSetter, T: Tui<P>>() -> io::Result<()> {
    let mut terminal = ratatui::init();
    let (mut s, r, mut c) = P::setup();
    let mut ts = TuiState::<P>::new(&mut s, &r, &mut c);
    let app_result = tui_run::<P, T>(&mut ts, &mut terminal);
    ratatui::restore();
    app_result
}

/// Provides a convenient wrapper around solve_cli and solve_interactive that
/// does the appropriate thing based on flags:
///  - By default runs silently and dumps stats every 30s
///  - Can be configured to .sample_print()
///    --sample_secs=10 <-- Sample::time(Duration::from_secs(10))
///    --sample_every=10000 <-- Sample::every_n(10000)
///  - Can run in interactive mode instead:
///    --interactive
pub fn solve_main<P: PuzzleSetter, T: Tui<P>>(stats_file: &str) {
    let flag = parse_main_args();
    if let MainFlags::Interactive = &flag {
        solve_interactive::<P, T>().unwrap();
        return;
    }
    let mut dbg = DbgObserver::new();
    dbg.sample_stats(stats_file, Sample::time(Duration::from_secs(30)));
    match flag {
        MainFlags::Default => {
            dbg.sample_print(Sample::never());
        },
        MainFlags::SampleEvery(n) => {
            dbg.sample_print(Sample::every_n(n));
        },
        MainFlags::SampleSecs(s) => {
            dbg.sample_print(Sample::time(Duration::from_secs(s)));
        },
        _ => {},
    };
    solve_cli::<P, _>(dbg);
}

#[cfg(any(test, feature = "test-util"))]
pub mod test_util {
    use super::*;
    /// Solves the puzzle (silently, unless a StepObserver is used), replacing the
    /// real givens with the provided ones. This is useful for testing.
    pub fn solve_with_given<P: PuzzleSetter, D: StepObserver<P::U, P::State>>(
        given: P::State,
        mut observer: D,
    ) {
        let (mut s, r, mut c) = P::setup_with_givens(given);
        let mut finder = FindFirstSolution::new(&mut s, &r, &mut c, Some(&mut observer));
        let _ = finder.solve().expect("Puzzle solver returned an error:");
    }
}

enum MainFlags {
    Default,
    SampleSecs(u64),
    SampleEvery(usize),
    Interactive,
}

fn parse_main_args() -> MainFlags {
    let mut args = vec![];
    for arg in env::args().skip(1) {
        if let Some((x, y)) = arg.split_once("=") {
            args.push(x.to_string());
            args.push(y.to_string());
        } else {
            args.push(arg);
        }
    }
    let mut flag: Option<MainFlags> = None;
    let only_one = "You may only specify one of --sample_secs, --sample_n, and --interactive";
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--sample_secs" => {
                if flag.is_some() {
                    panic!("{}", only_one);
                }
                let val = iter
                    .next().expect("--sample_secs requires a value")
                    .parse::<u64>()
                    .expect("Invalid value for --sample_secs. Must be an unsigned integer.");
                flag = Some(MainFlags::SampleSecs(val))
            },
            "--sample_every" => {
                if flag.is_some() {
                    panic!("{}", only_one);
                }
                let val = iter
                    .next().expect("--sample_every requires a value")
                    .parse::<usize>()
                    .expect("Invalid value for --sample_every. Must be an unsigned integer.");
                flag = Some(MainFlags::SampleEvery(val))
            },
            "--interactive" => {
                if flag.is_some() {
                    panic!("{}", only_one);
                }
                flag = Some(MainFlags::Interactive)
            },
            _ => panic!("Unknown flag: {}", arg),
        }
    }
    if let Some(f) = flag {
        f
    } else {
        MainFlags::Default
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    Ok,
    Err(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pane {
    Grid,
    TextArea,
}

#[derive(Debug, Clone, PartialEq, Eq, IntoPrimitive, TryFromPrimitive, strum_macros::EnumCount)]
#[repr(u8)]
pub enum Mode {
    GridCells = 1,
    /* TODO: Add these
    StackIntro,
    GridRows,
    GridCols,
    GridBoxes,
    Stats,
    */
    Constraints,
    ConstraintsRaw,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TuiStateEvent {
    Ignore,
    PaneSwitch,
    ModeUpdate,
    Step,
    Delegate(KeyEvent),
    Exit,
}

pub struct TuiState<'a, P: PuzzleSetter> {
    pub solver: DfsSolver<'a, P::U, P::State, P::Ranker, P::Constraint>,
    pub grid_pos: Index,
    pub scroll_pos: usize,
    pub scroll_lines: Vec<String>,
    pub mode: Mode,
    pub active: Pane,
    pub exit: Option<Status>,
}

/// Support for integrating a particular State implementation with the generic
/// Tui code. This library provides a working integration for any PuzzleSetter
/// with State = NineStd. TODO -- support the other standard ones.
pub trait Tui<P: PuzzleSetter> {
    fn init<'a>(state: &mut TuiState<'a, P>);
    fn on_mode_change<'a>(state: &mut TuiState<'a, P>);
    fn update<'a>(state: &mut TuiState<'a, P>);
    fn on_grid_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent);
    fn on_text_area_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent);
    fn draw_grid<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect);
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect);
}

impl <'a, P: PuzzleSetter> TuiState<'a, P> {
    pub fn new(puzzle: &'a mut P::State, ranker: &'a P::Ranker, constraint: &'a mut P::Constraint) -> Self {
        Self {
            solver: DfsSolver::new(puzzle, ranker, constraint),
            grid_pos: [0, 0],
            scroll_pos: 0,
            scroll_lines: Vec::new(),
            mode: Mode::GridCells,
            active: Pane::Grid,
            exit: None,
        }
    }

    pub fn step(&mut self) {
        if !self.solver.is_done() {
            let result = self.solver.step();
            if let Err(e) = result {
                self.exit(Status::Err(format!("{:?}", e)));
            }
        }
    }

    pub fn exit(&mut self, status: Status) {
        self.exit = Some(status);
    }
}

fn tui_run<'a, P: PuzzleSetter, T: Tui<P>>(state: &mut TuiState<'a, P>, terminal: &mut DefaultTerminal) -> io::Result<()> {
    T::init(state);
    while state.exit.is_none() {
        terminal.draw(|frame| {
            let (g, ta) = tui_draw::<P, T>(state, frame);
            T::draw_grid(state, frame, g);
            T::draw_text_area(state, frame, ta);
        })?;
        match tui_handle_events::<P, T>(state)? {
            TuiStateEvent::ModeUpdate => T::on_mode_change(state),
            TuiStateEvent::Delegate(ke) => match state.active.clone() {
                Pane::Grid => T::on_grid_event(state, ke),
                Pane::TextArea => T::on_text_area_event(state, ke),
            },
            _ => {},
        }
        T::update(state);
    }
    match state.exit.clone().unwrap() {
        Status::Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
        Status::Ok => Ok(()),
    }
}

fn tui_draw<'a, P: PuzzleSetter, T: Tui<P>>(state: &TuiState<'a, P>, frame: &mut Frame) -> (Rect, Rect) {
    let size = frame.area();
    let vertical_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(4),  // Header
            ratatui::layout::Constraint::Min(0),     // Body (fills remaining space)
            ratatui::layout::Constraint::Length(1),  // Footer
        ])
        .split(size);
    let header_area = vertical_chunks[0];
    let body_area = vertical_chunks[1];
    let footer_area = vertical_chunks[2];
    let horizontal_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            ratatui::layout::Constraint::Min(25),  // Grid on left
            ratatui::layout::Constraint::Min(25),  // Text area on right
        ])
        .split(body_area);
    let grid_area = horizontal_chunks[0];
    let text_area = horizontal_chunks[1];
    let title = Line::from(" Sudoku Debugger ".bold());
    let block = Block::bordered()
        .title(title.centered())
        .border_set(border::PLAIN);
    let solver_state: Span<'_> = match state.solver.solver_state() {
        DfsSolverState::Initializing(_) => "Initializing".yellow(),
        DfsSolverState::Advancing(_) => "Advancing".green(),
        DfsSolverState::Backtracking(_) => "Backtracking".red(),
        DfsSolverState::Exhausted => "Exhausted".magenta(),
        DfsSolverState::Solved => "Solved".blue(),
    };
    let header_lines = vec![
        Line::from(vec![
            "State: ".into(), solver_state,
            " Steps: ".into(), state.solver.step_count().to_string().yellow(),
            " Mode: ".into(), format!("{:?}", state.mode).yellow(),
        ]),
        if let DfsSolverState::Initializing(_) = state.solver.solver_state() {
            Line::from("Replaying given cells...")
        } else {
            match state.solver.constraint_result() {
                ConstraintResult::Contradiction(a) => {
                    Line::from(format!("Contradiction: ({})", a.get_name()).red())
                },
                ConstraintResult::Certainty(cd, a) => {
                    Line::from(format!("Certainty: {:?}={} ({})", cd.index, cd.value, a.get_name()).green())
                },
                _ => Line::from(""),
            }
        },
    ];
    frame.render_widget(
        Paragraph::new(Text::from(header_lines))
            .centered()
            .block(block),
        header_area,
    );
    let instructions = Line::from(vec![
        " Move ".into(),
        "W/A/S/D".blue().bold(),
        " Panes ".into(),
        "Ctrl+A/D".blue().bold(),
        " Modes ".into(),
        "Tab/Shift+Tab".blue().bold(),
        " Advance ".into(),
        "N".blue().bold(),
        " Quit ".into(),
        "Ctrl+C ".blue().bold(),
    ]);
    frame.render_widget(
        Paragraph::new(instructions).centered(),
        footer_area,
    );
    (grid_area, text_area)
}

fn tui_handle_events<'a, P: PuzzleSetter, T: Tui<P>>(state: &mut TuiState<'a, P>) -> io::Result<TuiStateEvent> {
    Ok(match event::read()? {
        Event::Key(key_event) if key_event.kind == KeyEventKind::Press => match key_event.code {
            KeyCode::Char('c') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                state.exit(Status::Ok);
                TuiStateEvent::Exit
            },
            KeyCode::BackTab => {
                let mut m: u8 = state.mode.clone().into();
                if m == 1 {
                    m = Mode::COUNT as u8;
                } else {
                    m -= 1;
                }
                state.mode = m.try_into().unwrap();
                TuiStateEvent::ModeUpdate
            },
            KeyCode::Tab => {
                let mut m: u8 = state.mode.clone().into();
                if m == Mode::COUNT as u8 {
                    m = 1;
                } else {
                    m += 1;
                }
                state.mode = m.try_into().unwrap();
                TuiStateEvent::ModeUpdate
            },
            KeyCode::Char('n') => {
                state.step();
                TuiStateEvent::Step
            },
            KeyCode::Char('a') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                if state.active == Pane::TextArea {
                    state.active = Pane::Grid;
                }
                TuiStateEvent::PaneSwitch
            },
            KeyCode::Char('d') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                if state.active == Pane::Grid {
                    state.active = Pane::TextArea;
                }
                TuiStateEvent::PaneSwitch
            },
            _ => { TuiStateEvent::Delegate(key_event) },
        },
        _ => TuiStateEvent::Ignore,
    })
}

/// Generic grid handling
fn grid_wasd<'a, P: PuzzleSetter>(state: &mut TuiState<'a, P>, key_event: KeyEvent) -> bool {
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

/// Generic scrolling handling
fn text_area_ws<'a, P: PuzzleSetter>(state: &mut TuiState<'a, P>, key_event: KeyEvent) -> bool {
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

/// Generic dumping of constraints
fn constraint_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<String> {
    if let Some(s) = state.solver.get_constraint().debug_at(state.solver.get_state(), state.grid_pos) {
        s.lines().map(|line| line.to_string()).collect()
    } else {
        vec!["No constraint information for this cell".to_string()]
    }
}

/// Generic dumping of constraints
fn constraint_raw_lines<'a, P: PuzzleSetter>(state: &TuiState<'a, P>) -> Vec<String> {
    format!("{:?}", state.solver.get_constraint())
        .lines()
        .map(|line| line.to_string())
        .collect()
}

/// Generic dumping of possible values
fn possible_value_lines<'a, P: PuzzleSetter<U = u8>, const MIN: u8, const MAX: u8>(state: &TuiState<'a, P>) -> Vec<String> {
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

/// Generic rendering of the top of the grid
fn grid_top(seg_len: usize, segs: usize) -> Line<'static> {
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

/// Generic rendering of the bottom of the grid
fn grid_bottom(seg_len: usize, segs: usize) -> Line<'static> {
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

/// Generic rendering of crossbars in the grid
fn grid_crossbar(seg_len: usize, segs: usize) -> Line<'static> {
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

// Generic rendering of cells in the grid
fn grid_cell<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, index: Index, cursor: Index, most_recent: Option<Index>) -> Span<'static> {
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

fn grid_line<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, row: usize, seg_len: usize, segs: usize, cursor: Index, most_recent: Option<Index>) -> Line<'static> {
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

fn draw_grid<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, v_seg_len: usize, v_segs: usize, h_seg_len: usize, h_segs: usize, frame: &mut Frame, area: Rect) {
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

fn draw_text_area<'a, P: PuzzleSetter>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
    let is_active = state.active == Pane::TextArea;
    let title_text = match state.mode {
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

pub struct NineStdTui;
impl <P: PuzzleSetter<U = u8, State = NineStd>> Tui<P> for NineStdTui {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = match state.mode {
            Mode::GridCells => possible_value_lines::<P, 1, 9>(state),
            Mode::Constraints => constraint_lines::<P>(state),
            Mode::ConstraintsRaw => constraint_raw_lines::<P>(state),
        };
    }
    fn on_mode_change<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_pos = 0;
        Self::update(state);
    }
    fn on_grid_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = grid_wasd(state, key_event);
    }
    fn on_text_area_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = text_area_ws(state, key_event);
    }
    fn draw_grid<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_grid(state, 3, 3, 3, 3, frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct EightStdTui;
impl <P: PuzzleSetter<U = u8, State = EightStd>> Tui<P> for EightStdTui {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = match state.mode {
            Mode::GridCells => possible_value_lines::<P, 1, 8>(state),
            Mode::Constraints => constraint_lines::<P>(state),
            Mode::ConstraintsRaw => constraint_raw_lines::<P>(state),
        };
    }
    fn on_mode_change<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_pos = 0;
        Self::update(state);
    }
    fn on_grid_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = grid_wasd(state, key_event);
    }
    fn on_text_area_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = text_area_ws(state, key_event);
    }
    fn draw_grid<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_grid(state, 2, 4, 2, 4, frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct SixStdTui;
impl <P: PuzzleSetter<U = u8, State = SixStd>> Tui<P> for SixStdTui {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = match state.mode {
            Mode::GridCells => possible_value_lines::<P, 1, 6>(state),
            Mode::Constraints => constraint_lines::<P>(state),
            Mode::ConstraintsRaw => constraint_raw_lines::<P>(state),
        };
    }
    fn on_mode_change<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_pos = 0;
        Self::update(state);
    }
    fn on_grid_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = grid_wasd(state, key_event);
    }
    fn on_text_area_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = text_area_ws(state, key_event);
    }
    fn draw_grid<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_grid(state, 2, 3, 3, 2, frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct FourStdTui;
impl <P: PuzzleSetter<U = u8, State = FourStd>> Tui<P> for FourStdTui {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = match state.mode {
            Mode::GridCells => possible_value_lines::<P, 1, 4>(state),
            Mode::Constraints => constraint_lines::<P>(state),
            Mode::ConstraintsRaw => constraint_raw_lines::<P>(state),
        };
    }
    fn on_mode_change<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_pos = 0;
        Self::update(state);
    }
    fn on_grid_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = grid_wasd(state, key_event);
    }
    fn on_text_area_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        let _ = text_area_ws(state, key_event);
    }
    fn draw_grid<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_grid(state, 2, 2, 2, 2, frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}