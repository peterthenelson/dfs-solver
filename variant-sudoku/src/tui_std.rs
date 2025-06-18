use std::marker::PhantomData;

/// Tui implementations for any PuzzleSetters that use NineStd, EightStd,
/// SixStd, or FourStd for their states. These are probably the only Tui
/// implementations you will need 90% of the time.
use crossterm::event::KeyEvent;
use ratatui::{layout::Rect, text::Line, Frame};
use crate::{
    core::Index,
    solver::PuzzleSetter,
    sudoku::{
        eight_standard_overlay,
        four_standard_overlay,
        nine_standard_overlay,
        six_standard_overlay,
        EightStd,
        EightStdOverlay,
        EightStdVal,
        FourStd,
        FourStdOverlay,
        FourStdVal,
        NineStd,
        NineStdOverlay,
        NineStdVal,
        SixStd,
        SixStdOverlay,
        SixStdVal, StdOverlay,
    },
    tui::{Tui, TuiState},
    tui_util::{
        draw_grid, draw_text_area, grid_dims, grid_wasd, scroll_lines, text_area_ws,
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

pub struct NineStdTui<P: PuzzleSetter<Value = NineStdVal, Overlay = NineStdOverlay, State = NineStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = NineStdVal, Overlay = NineStdOverlay, State = NineStd>> NineStdTui<P> {
    fn overlay() -> Option<StdOverlay<9, 9>> {
        Some(nine_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = NineStdVal, Overlay = NineStdOverlay, State = NineStd>> Tui<P> for NineStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::overlay());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::overlay());
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
        draw_grid(state, &Self::overlay(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct EightStdTui<P: PuzzleSetter<Value = EightStdVal, Overlay = EightStdOverlay, State = EightStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = EightStdVal, Overlay = EightStdOverlay, State = EightStd>> EightStdTui<P> {
    fn overlay() -> Option<StdOverlay<8, 8>> {
        Some(eight_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = EightStdVal, Overlay = EightStdOverlay, State = EightStd>> Tui<P> for EightStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::overlay());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::overlay());
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
        draw_grid(state, &Self::overlay(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct SixStdTui<P: PuzzleSetter<Value = SixStdVal, Overlay = SixStdOverlay, State = SixStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = SixStdVal, Overlay = SixStdOverlay, State = SixStd>> SixStdTui<P> {
    fn overlay() -> Option<StdOverlay<6, 6>> {
        Some(six_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = SixStdVal, Overlay = SixStdOverlay, State = SixStd>> Tui<P> for SixStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::overlay());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::overlay());
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
        draw_grid(state, &Self::overlay(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}

pub struct FourStdTui<P: PuzzleSetter<Value = FourStdVal, Overlay = FourStdOverlay, State = FourStd>>(PhantomData<P>);
impl <P: PuzzleSetter<Value = FourStdVal, Overlay = FourStdOverlay, State = FourStd>> FourStdTui<P> {
    fn overlay() -> Option<StdOverlay<4, 4>> {
        Some(four_standard_overlay())
    }
}
impl <P: PuzzleSetter<Value = FourStdVal, Overlay = FourStdOverlay, State = FourStd>> Tui<P> for FourStdTui<P> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines(state, &Self::overlay());
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &Self::overlay());
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
        draw_grid(state, &Self::overlay(), frame, area);
    }
    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}
