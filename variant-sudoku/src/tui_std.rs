use std::marker::PhantomData;

/// Tui implementations for any PuzzleSetters that use NineStd, EightStd,
/// SixStd, or FourStd for their states. These are probably the only Tui
/// implementations you will need 90% of the time.
use crossterm::event::KeyEvent;
use ratatui::{layout::Rect, text::Line, Frame};
use crate::{
    core::Index, solver::PuzzleSetter, sudoku::{eight_standard_overlay, four_standard_overlay, nine_standard_overlay, six_standard_overlay, EightStd, FourStd, NineStd, SVal, SixStd}, tui::{Tui, TuiState}, tui_util::{
        draw_grid,
        draw_text_area, 
        grid_wasd,
        grid_dims,
        scroll_lines,
        text_area_ws, GridConfig,
    }
};

fn adjust_len(i: usize, v: &Vec<Line>) -> usize {
    if v.len() == 0 {
        0
    } else if i+1 > v.len() {
        v.len() - 1
    } else {
        i
    }
}

fn adjust_pos(i: Index, dims: [usize; 2]) -> Index {
    let [r, c] = i;
    [std::cmp::min(r, dims[0]), std::cmp::min(c, dims[1])]
}

pub struct NineStdTui<P: PuzzleSetter<Value = SVal<1, 9>, State = NineStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = SVal<1, 9>, State = NineStd>> NineStdTui<P> {
    fn grid_cfg() -> GridConfig<P, 9, 9> {
        GridConfig::new(nine_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = SVal<1, 9>, State = NineStd>> Tui<P> for NineStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::grid_cfg());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::grid_cfg());
        state.grid_pos = adjust_pos(state.grid_pos, state.grid_dims);
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
        draw_grid(state, &Self::grid_cfg(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct EightStdTui<P: PuzzleSetter<Value = SVal<1, 8>, State = EightStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = SVal<1, 8>, State = EightStd>> EightStdTui<P> {
    fn grid_cfg() -> GridConfig<P, 8, 8> {
        GridConfig::new(eight_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = SVal<1, 8>, State = EightStd>> Tui<P> for EightStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::grid_cfg());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::grid_cfg());
        state.grid_pos = adjust_pos(state.grid_pos, state.grid_dims);
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
        draw_grid(state, &Self::grid_cfg(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct SixStdTui<P: PuzzleSetter<Value = SVal<1, 6>, State = SixStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = SVal<1, 6>, State = SixStd>> SixStdTui<P> {
    fn grid_cfg() -> GridConfig<P, 6, 6> {
        GridConfig::new(six_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = SVal<1, 6>, State = SixStd>> Tui<P> for SixStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::grid_cfg());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::grid_cfg());
        state.grid_pos = adjust_pos(state.grid_pos, state.grid_dims);
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
        draw_grid(state, &Self::grid_cfg(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct FourStdTui<P: PuzzleSetter<Value = SVal<1, 4>, State = FourStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = SVal<1, 4>, State = FourStd>> FourStdTui<P> {
    fn grid_cfg() -> GridConfig<P, 4, 4> {
        GridConfig::new(four_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = SVal<1, 4>, State = FourStd>> Tui<P> for FourStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::grid_cfg());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::grid_cfg());
        state.grid_pos = adjust_pos(state.grid_pos, state.grid_dims);
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
        draw_grid(state, &Self::grid_cfg(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}
