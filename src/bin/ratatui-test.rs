use std::io;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use strum::EnumCount;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    buffer::Buffer,
    layout::{Direction, Layout, Rect},
    style::{Style, Stylize},
    symbols::border,
    text::{Line, Span, Text},
    widgets::{Block, Padding, Paragraph, Widget},
    DefaultTerminal, Frame,
};
use variant_sudoku_dfs::{
    cages::{CageBuilder, CageChecker, CAGE_FEATURE},
    constraint::MultiConstraint,
    core::{ConstraintResult, FeatureVec, Index, State},
    kropki::{KropkiBuilder, KropkiChecker, KROPKI_BLACK_FEATURE},
    ranker::{OverlaySensitiveLinearRanker, NUM_POSSIBLE_FEATURE}, solver::{DfsSolver, DfsSolverState, DfsSolverView},
    sudoku::{nine_standard_overlay, unpack_sval_vals, NineStd, StandardSudokuChecker}
};

fn build_puzzle() -> NineStd {
    NineStd::new(nine_standard_overlay())
}

fn build_constraints(puzzle: &NineStd) -> MultiConstraint<u8, NineStd> {
    let cb = CageBuilder::new(true, puzzle.get_overlay());
    let cages = vec![
        cb.across(15, [0, 0], 3),
        cb.nosum(vec![[0, 3], [0, 4], [0, 5]]),
        cb.nosum(vec![[0, 6], [0, 7], [0, 8]]),
        cb.nosum(vec![[1, 0], [1, 1], [1, 2]]),
        cb.nosum(vec![[1, 3], [1, 4], [1, 5]]),
        cb.across(19, [1, 6], 3),
        cb.across(18, [2, 0], 3),
        cb.across(17, [2, 3], 3),
        cb.nosum(vec![[2, 6], [2, 7], [2, 8]]),
        cb.nosum(vec![[4, 0], [3, 0], [3, 1]]),
        cb.sum(10, vec![[4, 1], [4, 2], [3, 2]]),
        cb.sum(13, vec![[4, 3], [3, 3], [3, 4]]),
        cb.sum(16, vec![[4, 4], [4, 5], [3, 5]]),
        cb.nosum(vec![[4, 6], [3, 6], [3, 7]]),
        cb.sum(11, vec![[4, 7], [4, 8], [3, 8]]),
        cb.sum(14, vec![[5, 0], [6, 0], [6, 1]]),
        cb.nosum(vec![[5, 1], [5, 2], [5, 3]]),
        cb.nosum(vec![[5, 4], [5, 5], [5, 6]]),
        cb.nosum(vec![[5, 7], [5, 8], [6, 8]]),
        cb.nosum(vec![[6, 2], [6, 3], [6, 4]]),
        cb.nosum(vec![[6, 5], [6, 6], [6, 7]]),
        cb.sum(20, vec![[8, 0], [7, 0], [7, 1]]),
        cb.sum(12, vec![[8, 1], [8, 2], [7, 2]]),
        cb.nosum(vec![[8, 3], [7, 3], [7, 4]]),
        cb.sum(16, vec![[8, 4], [8, 5], [7, 5]]),
        cb.nosum(vec![[8, 6], [7, 6], [7, 7]]),
        cb.sum(16, vec![[8, 7], [8, 8], [7, 8]]),
    ];
    let kb = KropkiBuilder::new(puzzle.get_overlay());
    let kropkis = vec![
        kb.b_chain(vec![[0, 0], [1, 0], [2, 0]]),
        kb.b_chain(vec![[1, 2], [0, 2], [0, 3]]),
        kb.b_chain(vec![[2, 2], [2, 3], [1, 3]]),
        kb.b_chain(vec![[1, 5], [0, 5], [0, 6]]),
        kb.b_chain(vec![[2, 5], [2, 6], [1, 6]]),
        kb.b_down([0, 8]),
        kb.b_across([3, 1]),
        kb.b_across([3, 4]),
        kb.b_chain(vec![[3, 7], [3, 8], [2, 8]]),
        kb.b_down([4, 0]),
        kb.b_down([4, 1]),
        kb.b_down([4, 3]),
        kb.b_down([4, 4]),
        kb.b_down([4, 6]),
        kb.b_down([4, 7]),
        kb.b_down([6, 1]),
        kb.b_down([6, 2]),
        kb.b_down([6, 4]),
        kb.b_down([6, 5]),
        kb.b_down([6, 7]),
        kb.b_down([6, 8]),
        kb.b_across([8, 0]),
        kb.b_across([8, 3]),
        kb.b_across([8, 6]),
    ];
    return MultiConstraint::new(vec_box::vec_box![
        StandardSudokuChecker::new(&puzzle),
        CageChecker::new(cages),
        KropkiChecker::new(kropkis),
    ]);
}

fn build_ranker() -> OverlaySensitiveLinearRanker {
    OverlaySensitiveLinearRanker::new(FeatureVec::from_pairs(vec![
        (NUM_POSSIBLE_FEATURE, -100.0),
        (CAGE_FEATURE, 1.0),
        (KROPKI_BLACK_FEATURE, 1.0),
    ]), |_, x, y| x+y)
}

fn main() -> io::Result<()> {
    let mut terminal = ratatui::init();
    let mut puzzle = build_puzzle();
    let mut constraint = build_constraints(&puzzle);
    let ranker = build_ranker();
    let mut app = App::new(&mut puzzle, &ranker, &mut constraint);
    let app_result = app.run(&mut terminal);
    ratatui::restore();
    app_result
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
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
pub enum Mode {
    GridCells = 1,
    /* TODO: Add these
    GridRows,
    GridCols,
    GridBoxes,
    Stats,
    */
    Contraints,
}

pub struct App<'a> {
    solver: DfsSolver<'a, u8, NineStd, OverlaySensitiveLinearRanker, MultiConstraint<u8, NineStd>>,
    grid_pos: Index,
    scroll_pos: usize,
    scroll_lines: Vec<String>,
    mode: Mode,
    active: Pane,
    exit: Option<Status>,
}

pub struct HeaderWidget<'a>(&'a App<'a>);
impl <'a> Widget for HeaderWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let title = Line::from(" Sudoku Debugger ".bold());
        let block = Block::bordered()
            .title(title.centered())
            .border_set(border::PLAIN);
        let solver_state: Span<'_> = match self.0.solver.solver_state() {
            DfsSolverState::Initializing(_) => "Initializing".yellow(),
            DfsSolverState::Advancing(_) => "Advancing".green(),
            DfsSolverState::Backtracking(_) => "Backtracking".red(),
            DfsSolverState::Exhausted => "Exhausted".magenta(),
            DfsSolverState::Solved => "Solved".blue(),
        };
        let header_lines = vec![
            Line::from(vec![
                "State: ".into(), solver_state,
                " Steps: ".into(), self.0.solver.step_count().to_string().yellow(),
                " Mode: ".into(), format!("{:?}", self.0.mode).yellow(),
            ]),
            if let DfsSolverState::Initializing(_) = self.0.solver.solver_state() {
                Line::from("Replaying given cells...")
            } else {
                match self.0.solver.constraint_result() {
                    ConstraintResult::Contradiction(a) => {
                        Line::from(format!("Contradiction: ({})", a.get_name()).red())
                    },
                    ConstraintResult::Certainty(cd, a) => {
                        Line::from(format!("Certainty: {:?}={} ({})", cd.index, cd.value.val(), a.get_name()).green())
                    },
                    _ => Line::from(""),
                }

            }
        ];
        Paragraph::new(Text::from(header_lines))
            .centered()
            .block(block)
            .render(area, buf);
    }
}

pub struct GridUpdate<'a, 'b>(&'a mut App<'b>);
impl <'a, 'b> GridUpdate<'a, 'b> {
    fn update(&mut self) {
    }

    fn on_mode(&mut self, _: Mode) {
    }

    fn on_event(&mut self, key_event: KeyEvent) {
        let [r, c] = self.0.grid_pos;
        match key_event.code {
            KeyCode::Char('w') => if r > 0 {
                self.0.grid_pos = [r-1, c];
            },
            KeyCode::Char('s') => if r < 8 {
                self.0.grid_pos = [r+1, c];
            },
            KeyCode::Char('a') => if c > 0 {
                self.0.grid_pos = [r, c-1];
            },
            KeyCode::Char('d') => if c < 8 {
                self.0.grid_pos = [r, c+1];
            },
            _ => {},
        }
    }
}

pub struct GridWidget<'a>(&'a App<'a>);
impl <'a> GridWidget<'a> {
    fn cell(&self, index: Index, cursor: Index, most_recent: Option<Index>) -> Span<'_> {
        let mut s: Span<'_> = if let Some(v) = self.0.solver.get_state().get(index) {
            if index == cursor {
                format!("[{}]", v.val()).into()
            } else {
                format!(" {} ", v.val()).into()
            }
        } else {
            if index == cursor {
                "[ ]".into()
            } else {
                "   ".into()
            }
        };
        s = match most_recent {
            Some(mr) if mr == index => if self.0.solver.is_valid() {
                s.green()
            } else {
                s.red()
            },
            _ => s,
        };
        s
    }

    fn line(&self, row: usize, cursor: Index, most_recent: Option<Index>) -> Line<'_> {
        Line::from(vec![
            self.cell([row, 0], cursor, most_recent),
            self.cell([row, 1], cursor, most_recent),
            self.cell([row, 2], cursor, most_recent),
            "│".into(),
            self.cell([row, 3], cursor, most_recent),
            self.cell([row, 4], cursor, most_recent),
            self.cell([row, 5], cursor, most_recent),
            "│".into(),
            self.cell([row, 6], cursor, most_recent),
            self.cell([row, 7], cursor, most_recent),
            self.cell([row, 8], cursor, most_recent),
        ])
    }
}

impl <'a> Widget for GridWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let is_active = self.0.active == Pane::Left;
        let title_text = "Grid";
        let title = Line::from(if is_active {
            title_text.bold()
        } else {
            title_text.gray()
        });
        let block = Block::bordered()
            .title(title.centered())
            .border_set(if is_active { border::DOUBLE } else { border::PLAIN });
        let mr = self.0.solver.most_recent_action().map(|(i, _)| i);
        let grid_lines = vec![
            self.line(0, self.0.grid_pos, mr),
            self.line(1, self.0.grid_pos, mr),
            self.line(2, self.0.grid_pos, mr),
            Line::from("─────────┼─────────┼─────────"),
            self.line(3, self.0.grid_pos, mr),
            self.line(4, self.0.grid_pos, mr),
            self.line(5, self.0.grid_pos, mr),
            Line::from("─────────┼─────────┼─────────"),
            self.line(6, self.0.grid_pos, mr),
            self.line(7, self.0.grid_pos, mr),
            self.line(8, self.0.grid_pos, mr),
        ];
        Paragraph::new(Text::from(grid_lines))
            .centered()
            .block(block)
            .render(area, buf);
    }
}

pub struct TextAreaUpdate<'a, 'b>(&'a mut App<'b>);
impl <'a, 'b> TextAreaUpdate<'a, 'b> {
    fn constraint_lines(&self) -> Vec<String> {
        format!("{:?}", self.0.solver.get_constraint())
            .lines()
            .map(|line| line.to_string())
            .collect()
    }
    fn possible_value_lines(&self) -> Vec<String> {
        let mut lines;
        if let Some(g) = self.0.solver.decision_grid() {
            lines = vec![
                format!("Possible Values in cell {:?}:", self.0.grid_pos),
                format!("{:?}", unpack_sval_vals::<1, 9>(&g.get(self.0.grid_pos).0)),
                "".to_string(),
                "Features:".to_string(),
            ];
            lines.extend(format!("{:?}", g.get(self.0.grid_pos).1)
                .lines()
                .map(|line| line.to_string()));
        } else {
            lines = vec!["No Decision Grid Available".to_string()];
        }
        lines
    }
    fn update(&mut self) {
        self.0.scroll_lines = match self.0.mode {
            Mode::GridCells => self.possible_value_lines(),
            Mode::Contraints => self.constraint_lines(),
        }
    }
    fn on_mode(&mut self, mode: Mode) {
        self.0.scroll_pos = 0;
        self.0.scroll_lines = match mode {
            Mode::GridCells => self.possible_value_lines(),
            Mode::Contraints => self.constraint_lines(),
        };
    }
    fn on_event(&mut self, key_event: KeyEvent) {
        match key_event.code {
            KeyCode::Char('w') => if self.0.scroll_pos > 0 {
                self.0.scroll_pos -= 1;
            },
            KeyCode::Char('s') => if self.0.scroll_pos+1 < self.0.scroll_lines.len() {
                self.0.scroll_pos += 1;
            },
            _ => {},
        }
    }
} 

pub struct TextAreaWidget<'a>(&'a App<'a>);
impl <'a> Widget for TextAreaWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let is_active = self.0.active == Pane::Right;
        let title_text = match self.0.mode {
            Mode::Contraints => "Constraints",
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
            self.0.scroll_lines[self.0.scroll_pos..self.0.scroll_lines.len()]
            .iter().map(|line| Line::from(line.clone())).collect::<Vec<_>>()
        );
        Paragraph::new(text)
            .left_aligned()
            .block(block)
            .render(area, buf);
    }
}

impl <'a> App<'a> {
    pub fn new(puzzle: &'a mut NineStd, ranker: &'a OverlaySensitiveLinearRanker, constraint: &'a mut MultiConstraint<u8, NineStd>) -> Self {
        let mut app = Self {
            solver: DfsSolver::new(puzzle, ranker, constraint),
            grid_pos: [0, 0],
            scroll_pos: 0,
            scroll_lines: Vec::new(),
            mode: Mode::GridCells,
            active: Pane::Left,
            exit: None,
        };
        app.init();
        app
    }

    fn init(&mut self) {
        {
            let mode = self.mode.clone();
            let mut u = GridUpdate(self);
            u.on_mode(mode);
        }
        {
            let mode = self.mode.clone();
            let mut u = TextAreaUpdate(self);
            u.on_mode(mode);
        }
    }

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
        frame.render_widget(HeaderWidget(self), header_area);
        frame.render_widget(
            Paragraph::new(instructions).centered(),
            footer_area,
        );
        frame.render_widget(GridWidget(self), grid_area);
        frame.render_widget(TextAreaWidget(self), text_area);
    }

    fn handle_events(&mut self) -> io::Result<()> {
        match event::read()? {
            Event::Key(key_event) if key_event.kind == KeyEventKind::Press => {
                self.handle_key_event(key_event)
            }
            _ => {}
        };
        {
            let mut u = GridUpdate(self);
            u.update();
        }
        {
            let mut u = TextAreaUpdate(self);
            u.update();
        }
        Ok(())
    }

    fn update_mode(&mut self, mode: Mode) {
        self.mode = mode;
        let new_mode = self.mode.clone();
        match self.active {
            Pane::Left => {
                let mut u = GridUpdate(self);
                u.on_mode(new_mode);
            },
            Pane::Right => {
                let mut u = TextAreaUpdate(self);
                u.on_mode(new_mode);
            },
        }
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
            _ => match self.active {
                Pane::Left => {
                    let mut u = GridUpdate(self);
                    u.on_event(key_event);
                },
                Pane::Right => {
                    let mut u = TextAreaUpdate(self);
                    u.on_event(key_event);
                },
            },
        }
    }

    fn exit(&mut self, status: Status) {
        self.exit = Some(status);
    }
}
