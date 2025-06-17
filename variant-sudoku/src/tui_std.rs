/// Tui implementations for any PuzzleSetters that use NineStd, EightStd,
/// SixStd, or FourStd for their states. These are probably the only Tui
/// implementations you will need 90% of the time.
use crossterm::event::KeyEvent;
use ratatui::{layout::Rect, Frame};
use crate::{
    solver::PuzzleSetter,
    sudoku::{EightStd, FourStd, NineStd, SixStd},
    tui::{Tui, TuiState},
    tui_util::{
        draw_grid,
        draw_text_area, 
        grid_wasd,
        scroll_lines_generic,
        text_area_ws,
    },
};

pub struct NineStdTui;
impl <P: PuzzleSetter<U = u8, State = NineStd>> Tui<P> for NineStdTui {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        Self::on_mode_change(state)
    }
    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines_generic::<P, 1, 9>(state);
        if state.scroll_lines.len() == 0 {
            state.scroll_pos = 0;
        } else if state.scroll_pos+1 > state.scroll_lines.len() {
            state.scroll_pos = state.scroll_lines.len() - 1;
        }
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
        state.scroll_lines = scroll_lines_generic::<P, 1, 8>(state);
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
        state.scroll_lines = scroll_lines_generic::<P, 1, 6>(state);
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
        state.scroll_lines = scroll_lines_generic::<P, 1, 4>(state);
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
