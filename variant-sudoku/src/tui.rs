use std::{env, io, sync::Mutex, time::Duration};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    layout::{Direction, Layout, Rect},
    style::{Stylize},
    symbols::border,
    text::{Line, Span, Text},
    widgets::{Block, Paragraph},
    DefaultTerminal,
    Frame,
};
use strum::EnumCount;
use crate::{
    core::{ConstraintResult, Index, Overlay, State},
    debug::{DbgObserver, Sample},
    solver::{DfsSolver, DfsSolverState, DfsSolverView, FindFirstSolution, PuzzleSetter, StepObserver},
};

/// Solves the puzzle in command-line mode. No interactivity, but a StepObserver
/// can be passed in to periodically print out or save debug information. The
/// solution will be printed at the end.
pub fn solve_cli<P: PuzzleSetter, D: StepObserver<P::Value, P::Overlay, P::Ranker, P::Constraint>>(mut observer: D) {
    let (mut s, r, mut c) = P::setup();
    let mut finder = FindFirstSolution::new(&mut s, &r, &mut c, Some(&mut observer));
    let maybe_solution = finder.solve().expect("Puzzle solver returned an error:");
    println!("Solution:\n{:?}", maybe_solution.expect("No solution found!").state());
}

/// Solves the puzzle in the interactive debugger.
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
pub fn solve_main<P: PuzzleSetter, T: Tui<P>>() {
    let flag = parse_main_args();
    if let MainFlags::Interactive = &flag {
        solve_interactive::<P, T>().unwrap();
        return;
    }
    let mut dbg = DbgObserver::new();
    if let Some(name) = P::name() {
        let figure = format!("stats/{}.png", name);
        let json = format!("stats/{}.json", name);
        dbg.sample_stats(figure, json, Sample::time(Duration::from_secs(30)));
    } else {
        println!("Warning: Puzzle does not have a name; no stats will be saved.")
    }
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
    /// real givens with the provided ones. This works similar to solve_cli, but
    /// it's useful in testing situations.
    pub fn solve_with_given<P: PuzzleSetter, D: StepObserver<P::Value, P::Overlay, P::Ranker, P::Constraint>>(
        given: State<P::Value, P::Overlay>,
        mut observer: D,
    ) {
        let (mut s, r, mut c) = P::setup_with_givens(given);
        let mut finder = FindFirstSolution::new(&mut s, &r, &mut c, Some(&mut observer));
        let _ = finder.solve()
            .expect("Puzzle solver returned an error:")
            .expect("Puzzle solver didn't find a solution");
    }

    /// You can use the interactive debugger in your tests to debug problems.
    /// Usage notes:
    /// - If you don't have a PuzzleSetter already, you can probably use
    ///   FakeSetter to magic one up that's sufficient for using this function.
    /// - If you have a common puzzle setup, you may be able to do something
    ///   like this:
    ///   debug_std!(NineStdTui, &mut puzzle, &ranker, &mut constraint);
    /// - You should run just the relevant test case and not capture output:
    ///   $ cargo test my_test_case -- --nocapture
    pub fn interactive_debug<P: PuzzleSetter, T: Tui<P>>(state: &mut State<P::Value, P::Overlay>, ranker: &P::Ranker, constraint: &mut P::Constraint) {
        let mut terminal = ratatui::init();
        let mut ts = TuiState::<P>::new(state, ranker, constraint);
        let app_result = tui_run::<P, T>(&mut ts, &mut terminal);
        ratatui::restore();
        app_result.unwrap();
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

lazy_static::lazy_static! {
    static ref DEBUG_TEXT: Mutex<Option<String>> = Mutex::new(None);
}

pub fn tui_debug(s: String) {
    let mut lock = DEBUG_TEXT.lock().unwrap();
    *lock = Some(s);
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
    Readme = 1,
    PossibilityHeatmap,
    ScoreHeatmap,
    Stack,
    Constraints,
    ConstraintsRaw,
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Modal {
    ManualEntry(Index),
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TuiStateEvent {
    Ignore,
    PaneSwitch,
    ModeUpdate,
    Step,
    Jump,
    PlayPause,
    Reset,
    Undo,
    Delegate(KeyEvent),
    Exit,
}

pub struct TuiState<'a, P: PuzzleSetter> {
    pub solver: DfsSolver<'a, P::Value, P::Overlay, P::Ranker, P::Constraint>,
    pub grid_pos: Index,
    pub grid_dims: [usize; 2],
    pub constraint_index: Option<usize>,
    pub scroll_pos: usize,
    pub scroll_lines: Vec<Line<'a>>,
    pub paused: bool,
    pub mode: Mode,
    pub active: Pane,
    pub exit: Option<Status>,
    pub modal: Option<Modal>,
}

/// Support for integrating a particular State implementation with the generic
/// Tui code. See tui_std for working integration for any PuzzleSetter
/// whose State is NineStd, EightStd, SixStd, or FourStd, as well as partial
/// functionality for any non-standard puzzles.
pub trait Tui<P: PuzzleSetter> {
    fn init<'a>(state: &mut TuiState<'a, P>);
    fn on_mode_change<'a>(state: &mut TuiState<'a, P>);
    fn update<'a>(state: &mut TuiState<'a, P>);
    fn on_grid_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent);
    fn on_text_area_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent);
    fn on_modal_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent);
    fn draw_grid<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect);
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect);
    fn draw_modal<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect);
}

impl <'a, P: PuzzleSetter> TuiState<'a, P> {
    pub fn new(puzzle: &'a mut State<P::Value, P::Overlay>, ranker: &'a P::Ranker, constraint: &'a mut P::Constraint) -> Self {
        let (n, m) = puzzle.overlay().grid_dims();
        Self {
            solver: DfsSolver::new(puzzle, ranker, constraint),
            grid_pos: [0, 0],
            grid_dims: [n, m],
            constraint_index: None,
            scroll_pos: 0,
            scroll_lines: Vec::new(),
            paused: true,
            mode: Mode::Readme,
            active: Pane::Grid,
            exit: None,
            modal: None,
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

    pub fn reset(&mut self) {
        self.solver.reset();
    }

    pub fn undo(&mut self) {
        let result = self.solver.retreat();
        if let Err(e) = result {
            self.exit(Status::Err(format!("{:?}", e)));
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
            let (g, ta, m) = tui_draw::<P, T>(state, frame);
            T::draw_grid(state, frame, g);
            T::draw_text_area(state, frame, ta);
            T::draw_modal(state, frame, m);
        })?;
        match tui_handle_events::<P, T>(state)? {
            Some(TuiStateEvent::ModeUpdate) => T::on_mode_change(state),
            Some(TuiStateEvent::Delegate(ke)) => match &state.modal {
                Some(_) => T::on_modal_event(state, ke),
                None => match &state.active {
                    Pane::Grid => T::on_grid_event(state, ke),
                    Pane::TextArea => T::on_text_area_event(state, ke),
                },
            },
            _ => {},
        }
        if !state.solver.is_done() && !state.paused {
            state.step()
        }
        T::update(state);
    }
    match state.exit.clone().unwrap() {
        Status::Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
        Status::Ok => Ok(()),
    }
}

fn modal_area(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    let vertical = Layout::vertical([
        ratatui::layout::Constraint::Percentage(percent_y)
    ]).flex(ratatui::layout::Flex::Center);
    let horizontal = Layout::horizontal([
        ratatui::layout::Constraint::Percentage(percent_x)
    ]).flex(ratatui::layout::Flex::Center);
    let [area] = vertical.areas(area);
    let [area] = horizontal.areas(area);
    area
}

fn tui_draw<'a, P: PuzzleSetter, T: Tui<P>>(state: &TuiState<'a, P>, frame: &mut Frame) -> (Rect, Rect, Rect) {
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
        DfsSolverState::InitializationFailed => "Initialization Failed".magenta(),
        DfsSolverState::Advancing(_) => "Advancing".green(),
        DfsSolverState::Backtracking(_) => "Backtracking".red(),
        DfsSolverState::Exhausted => "Exhausted".magenta(),
        DfsSolverState::Solved => "Solved".blue(),
    };
    let mut first_line = vec![
        "State: ".into(), solver_state,
        " Steps: ".into(), state.solver.step_count().to_string().yellow(),
        " Mode: ".into(), format!("{:?}", state.mode).yellow(),
    ];
    {
        let lock = DEBUG_TEXT.lock().unwrap();
        if let Some(dt) = &*lock {
            first_line.push(" -- ".into());
            first_line.push(dt.clone().magenta());
        };
    }
    let header_lines = vec![
        Line::from(first_line),
        if let DfsSolverState::Initializing(_) = state.solver.solver_state() {
            Line::from("Replaying given cells...")
        } else {
            match state.solver.constraint_result() {
                ConstraintResult::Contradiction(a) => Line::from(vec![
                    "Contradiction: ".red(),
                    format!("({})", a.name()).cyan(),
                ]),
                ConstraintResult::Certainty(cd, a) => Line::from(vec![
                    "Certainty: ".green(),
                    format!("{:?}", cd.index).blue(),
                    " = ".into(),
                    format!("{} ", cd.value).green(),
                    format!("({})", a.name()).cyan(),
                ]),
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
        "Space".blue().bold(),
        " Modes ".into(),
        "Tab/Shift+Tab".blue().bold(),
        " Quit ".into(),
        "Ctrl+C ".blue().bold(),
    ]);
    frame.render_widget(
        Paragraph::new(instructions).centered(),
        footer_area,
    );
    (grid_area, text_area, modal_area(frame.area(), 30, 20))
}

fn tui_handle_events<'a, P: PuzzleSetter, T: Tui<P>>(state: &mut TuiState<'a, P>) -> io::Result<Option<TuiStateEvent>> {
    let has_event = event::poll(Duration::from_secs(0))?;
    if !has_event {
        return Ok(None);
    }
    let ev = event::read()?;
    if state.modal.is_some() {
        match &ev {
            Event::Key(ke) if ke.kind == KeyEventKind::Press => {
                return Ok(Some(TuiStateEvent::Delegate(*ke)));
            },
            _ => {},
        }
    }
    Ok(Some(match ev {
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
            KeyCode::Char('z') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                state.undo();
                TuiStateEvent::Undo
            },
            KeyCode::Char('j') => {
                state.step();
                while !state.solver.is_done() {
                    if let ConstraintResult::Ok = state.solver.constraint_result() {
                        break;
                    }
                    state.step();
                }
                TuiStateEvent::Jump
            },
            KeyCode::Char('p') => {
                state.paused = !state.paused;
                TuiStateEvent::PlayPause
            },
            KeyCode::Char('r') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                state.reset();
                TuiStateEvent::Reset
            },
            KeyCode::Char(' ') => {
                state.active = match state.active {
                    Pane::TextArea => Pane::Grid,
                    Pane::Grid => Pane::TextArea,
                };
                TuiStateEvent::PaneSwitch
            },
            _ => { TuiStateEvent::Delegate(key_event) },
        },
        _ => TuiStateEvent::Ignore,
    }))
}
