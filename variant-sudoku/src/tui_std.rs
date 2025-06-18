use std::marker::PhantomData;

/// Tui implementations for any PuzzleSetters that use NineStd, EightStd,
/// SixStd, or FourStd for their states. These are probably the only Tui
/// implementations you will need 90% of the time. You can also use the
/// DefaultTui with arbitrary PuzzleSetters, but any features relying on
/// a StdOverlay will be disabled.
use crossterm::event::KeyEvent;
use ratatui::{layout::Rect, text::Line, Frame};
use crate::{
    core::{Index, State},
    solver::{DfsSolverView, PuzzleSetter},
    sudoku::{
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

pub trait OverlayStandardizer<P: PuzzleSetter, const N: usize, const M: usize> {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<N, M>>;
}

pub struct NullOverlayStandardizer<const N: usize, const M: usize>;
impl <P: PuzzleSetter, const N: usize, const M: usize> OverlayStandardizer<P, N, M> for NullOverlayStandardizer<N, M> {
    fn to_std(_: &<P as PuzzleSetter>::Overlay) -> Option<StdOverlay<N, M>> { None }
}
impl <P: PuzzleSetter<Value = NineStdVal, Overlay = NineStdOverlay, State = NineStd>> OverlayStandardizer<P, 9, 9> for NineStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<9, 9>> { Some(overlay.clone()) }
}
impl <P: PuzzleSetter<Value = EightStdVal, Overlay = EightStdOverlay, State = EightStd>> OverlayStandardizer<P, 8, 8> for EightStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<8, 8>> { Some(overlay.clone()) }
}
impl <P: PuzzleSetter<Value = SixStdVal, Overlay = SixStdOverlay, State = SixStd>> OverlayStandardizer<P, 6, 6> for SixStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<6, 6>> { Some(overlay.clone()) }
}
impl <P: PuzzleSetter<Value = FourStdVal, Overlay = FourStdOverlay, State = FourStd>> OverlayStandardizer<P, 4, 4> for FourStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<4, 4>> { Some(overlay.clone()) }
}

pub struct DefaultTui<P: PuzzleSetter, const N: usize, const M: usize, OS: OverlayStandardizer<P, N, M>>(PhantomData<(P, OS)>);
pub type NineStdTui<P> = DefaultTui<P, 9, 9, NineStd>;
pub type EightStdTui<P> = DefaultTui<P, 8, 8, EightStd>;
pub type SixStdTui<P> = DefaultTui<P, 6, 6, SixStd>;
pub type FourStdTui<P> = DefaultTui<P, 4, 4, FourStd>;

impl <P: PuzzleSetter, const N: usize, const M: usize, OS: OverlayStandardizer<P, N, M>> DefaultTui<P, N, M, OS> {
    pub fn new() -> Self { Self(PhantomData) }
}

impl <P: PuzzleSetter, const N: usize, const M: usize, OS: OverlayStandardizer<P, N, M>>
Tui<P> for DefaultTui<P, N, M, OS> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        // Wish this could be static
        assert_eq!(P::State::ROWS, N);
        assert_eq!(P::State::COLS, M);
        Self::on_mode_change(state)
    }

    fn update<'a>(state: &mut TuiState<'a, P>) {
        let so = OS::to_std(state.solver.state().overlay());
        state.scroll_lines = scroll_lines(state, &so);
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims(state, &so);
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
        let so = OS::to_std(state.solver.state().overlay());
        draw_grid(state, &so, frame, area);
    }

    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }
}
