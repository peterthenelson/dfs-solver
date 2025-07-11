use std::marker::PhantomData;

/// Tui implementations for any PuzzleSetters that use NineStd, EightStd,
/// SixStd, or FourStd for their states. These are probably the only Tui
/// implementations you will need 90% of the time. You can also use the
/// DefaultTui with arbitrary PuzzleSetters, but any features relying on
/// a StdOverlay will be disabled.
use crossterm::event::KeyEvent;
use ratatui::{layout::Rect, text::Line, Frame};
use crate::{
    core::{Index, Overlay},
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
        constraint_c,
        draw_grid,
        draw_modal,
        draw_text_area,
        grid_dims,
        grid_m,
        grid_wasd,
        modal_event,
        scroll_lines,
        text_area_ws,
        ConstraintSplitter
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
impl <P: PuzzleSetter<Value = NineStdVal, Overlay = NineStdOverlay>> OverlayStandardizer<P, 9, 9> for NineStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<9, 9>> { Some(overlay.clone()) }
}
impl <P: PuzzleSetter<Value = EightStdVal, Overlay = EightStdOverlay>> OverlayStandardizer<P, 8, 8> for EightStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<8, 8>> { Some(overlay.clone()) }
}
impl <P: PuzzleSetter<Value = SixStdVal, Overlay = SixStdOverlay>> OverlayStandardizer<P, 6, 6> for SixStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<6, 6>> { Some(overlay.clone()) }
}
impl <P: PuzzleSetter<Value = FourStdVal, Overlay = FourStdOverlay>> OverlayStandardizer<P, 4, 4> for FourStd {
    fn to_std(overlay: &P::Overlay) -> Option<StdOverlay<4, 4>> { Some(overlay.clone()) }
}

pub struct DefaultTui<P: PuzzleSetter, const N: usize, const M: usize, OS: OverlayStandardizer<P, N, M>, CS: ConstraintSplitter<P>>(PhantomData<(P, OS, CS)>);
pub type NineStdTui<P> = DefaultTui<P, 9, 9, NineStd, P>;
pub type EightStdTui<P> = DefaultTui<P, 8, 8, EightStd, P>;
pub type SixStdTui<P> = DefaultTui<P, 6, 6, SixStd, P>;
pub type FourStdTui<P> = DefaultTui<P, 4, 4, FourStd, P>;

#[cfg(any(test, feature = "test-util"))]
pub mod test_util {
    #[macro_export]
    macro_rules! debug_std {
        ($stdtui:ident, $puzzle:expr, $ranker:expr, $constraint:expr) => {{
            use crate::tui::test_util::interactive_debug;
            use crate::solver::test_util::FakeSetter;
            let puzzle_ref = $puzzle;
            let ranker_ref = $ranker;
            let constraint_ref = $constraint;
            interactive_debug::<
                FakeSetter<_, _, _, _>,
                $stdtui<FakeSetter<_, _, _, _>>
            >(puzzle_ref, ranker_ref, constraint_ref)
        }};
    }
}

impl <P: PuzzleSetter, const N: usize, const M: usize, OS: OverlayStandardizer<P, N, M>, CS: ConstraintSplitter<P>>
DefaultTui<P, N, M, OS, CS> {
    pub fn new() -> Self { Self(PhantomData) }
}

impl <P: PuzzleSetter, const N: usize, const M: usize, OS: OverlayStandardizer<P, N, M>, CS: ConstraintSplitter<P>>
Tui<P> for DefaultTui<P, N, M, OS, CS> {
    fn init<'a>(state: &mut TuiState<'a, P>) {
        // Wish this could be static
        let (n, m) = state.solver.state().overlay().grid_dims();
        assert_eq!(n, N);
        assert_eq!(m, M);
        Self::on_mode_change(state)
    }

    fn update<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_lines = scroll_lines::<P, N, M, CS>(state);
        state.scroll_pos = adjust_len(state.scroll_pos, &state.scroll_lines);
        state.grid_dims = grid_dims::<P, N, M>(state);
        state.grid_pos = adjust_pos(state.grid_pos, state.grid_dims);
    }

    fn on_mode_change<'a>(state: &mut TuiState<'a, P>) {
        state.scroll_pos = 0;
        Self::update(state);
    }

    fn on_grid_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        if grid_m(state, key_event) { return }
        if grid_wasd(state, key_event) { return }
        if constraint_c::<P, CS>(state, key_event) { return }
    }

    fn on_text_area_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        if text_area_ws(state, key_event) { return }
        if constraint_c::<P, CS>(state, key_event) { return }
    }

    fn on_modal_event<'a>(state: &mut TuiState<'a, P>, key_event: KeyEvent) {
        if modal_event(state, key_event) { return }
    }

    fn draw_grid<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        let so = OS::to_std(state.solver.state().overlay());
        draw_grid::<P, N, M, CS>(state, &so, frame, area);
    }

    fn draw_text_area<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_text_area(state, frame, area);
    }

    fn draw_modal<'a>(state: &TuiState<'a, P>, frame: &mut Frame, area: Rect) {
        draw_modal(state, frame, area);
    }
}
