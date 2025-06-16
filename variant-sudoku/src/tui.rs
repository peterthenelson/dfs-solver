use std::io;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    buffer::Buffer,
    layout::{Direction, Layout},
    style::{Style, Stylize},
    text::{Line, Span, Text},
    widgets::Paragraph,
    DefaultTerminal,
    Frame,
};
use strum::EnumCount;
use crate::{core::Index, solver::{DfsSolver, DfsSolverView, FindFirstSolution, PuzzleSetter, StepObserver}, sudoku::NineStd};

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
pub fn solve_interactive<P: PuzzleSetter<U = u8, State = NineStd>>() -> io::Result<()> {
    let mut terminal = ratatui::init();
    let (mut s, r, mut c) = P::setup();
    let mut app = TuiState::<P>::new(&mut s, &r, &mut c);
    let app_result = app.run(&mut terminal);
    ratatui::restore();
    app_result
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

#[derive(Debug, Clone, PartialEq, Eq)]
enum Status {
    Ok,
    Err(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pane {
    Left,
    Right,
}

#[derive(Debug, Clone, PartialEq, Eq, IntoPrimitive, TryFromPrimitive, strum_macros::EnumCount)]
#[repr(u8)]
enum Mode {
    GridCells = 1,
    /* TODO: Add these
    GridRows,
    GridCols,
    GridBoxes,
    Stats,
    */
    Contraints,
}

struct TuiState<'a, P: PuzzleSetter<U = u8, State = NineStd>> {
    solver: DfsSolver<'a, P::U, P::State, P::Ranker, P::Constraint>,
    grid_pos: Index,
    scroll_pos: usize,
    scroll_lines: Vec<String>,
    mode: Mode,
    active: Pane,
    exit: Option<Status>,
}


impl <'a, P: PuzzleSetter<U = u8, State = NineStd>> TuiState<'a, P> {
    pub fn new(puzzle: &'a mut P::State, ranker: &'a P::Ranker, constraint: &'a mut P::Constraint) -> Self {
        let mut t = Self {
            solver: DfsSolver::new(puzzle, ranker, constraint),
            grid_pos: [0, 0],
            scroll_pos: 0,
            scroll_lines: Vec::new(),
            mode: Mode::GridCells,
            active: Pane::Left,
            exit: None,
        };
        t.init();
        t
    }

    fn init(&mut self) { /* TODO */ }

    fn step(&mut self) {
        if !self.solver.is_done() {
            let result = self.solver.step();
            if let Err(e) = result {
                self.exit(Status::Err(format!("{:?}", e)));
            }
        }
    }

    pub fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while self.exit.is_none() {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events()?;
        }
        match self.exit.clone().unwrap() {
            Status::Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
            Status::Ok => Ok(()),
        }
    }

    fn draw(&self, frame: &mut Frame) {
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
        // TODO
        frame.render_widget(Paragraph::new(Text::from("header")), header_area);
        frame.render_widget(
            Paragraph::new(instructions).centered(),
            footer_area,
        );
        frame.render_widget(Paragraph::new(Text::from("grid")), grid_area);
        frame.render_widget(Paragraph::new(Text::from("header")), text_area);
    }

    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        // TODO
        Ok(())
    }

    fn update_mode(&mut self, mode: Mode) {
        self.mode = mode;
        let new_mode = self.mode.clone();
        // TODO
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('c') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                self.exit(Status::Ok);
            },
            KeyCode::BackTab => {
                let mut m: u8 = self.mode.clone().into();
                if m == 1 {
                    m = Mode::COUNT as u8;
                } else {
                    m -= 1;
                }
                self.update_mode(m.try_into().unwrap());
            },
            KeyCode::Tab => {
                let mut m: u8 = self.mode.clone().into();
                if m == Mode::COUNT as u8 {
                    m = 1;
                } else {
                    m += 1;
                }
                self.update_mode(m.try_into().unwrap());
            },
            KeyCode::Char('n') => self.step(),
            KeyCode::Char('a') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.active == Pane::Right {
                    self.active = Pane::Left;
                }
            },
            KeyCode::Char('d') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                if self.active == Pane::Left {
                    self.active = Pane::Right;
                }
            },
            _ => { /* TODO */ },
        }
    }

    fn exit(&mut self, status: Status) {
        self.exit = Some(status);
    }
}