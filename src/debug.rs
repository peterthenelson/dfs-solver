use std::{collections::HashMap, time::{Duration, SystemTime}};

use rand::{distr::{Bernoulli, Distribution}, rng, rngs::ThreadRng};
use crate::core::{State, UInt};
use crate::constraint::{ConstraintResult, Possibilities};
use crate::solver::{DfsSolverState, DfsSolverView, StepObserver};
use plotters::{chart::ChartBuilder, coord::Shift, prelude::{BitMapBackend, Circle, DrawResult, DrawingArea, DrawingBackend, IntoDrawingArea, IntoLogRange, IntoSegmentedCoord, MultiLineText, Rectangle, SegmentValue}, style::{Color, IntoFont, BLUE, RED, WHITE}};

fn short_result<U: UInt, S: State<U>>(result: &ConstraintResult<U, S::Value>, puzzle: &S) -> String {
    match result {
        ConstraintResult::Contradiction => "Contradiction".to_string(),
        ConstraintResult::Certainty(d) => {
            format!("Certainty({:?}, {:?})", d.index, d.value).to_string()
        },
        ConstraintResult::Other(Possibilities::Any) => "Any".to_string(),
        ConstraintResult::Other(Possibilities::Grid(_)) => {
            if result.has_contradiction(puzzle) {
                "Grid with Contradiction".to_string()
            } else if let Some(d) = result.has_certainty(puzzle) {
                format!("Grid with Certainty({:?}, {:?})", d.index, d.value).to_string()
            } else {
                "Grid[...]".to_string()
            }
        }
    }
}

fn bar_chart<'a, DB: DrawingBackend>(area: &DrawingArea<DB, Shift>, histogram: &Histogram, bar_margin: u32) -> DrawResult<(), DB> {
    let mut chart_builder = ChartBuilder::on(area);
    chart_builder.margin(5).set_left_and_bottom_label_area_size(20);
    let mut chart_context = chart_builder.build_cartesian_2d(
        (0..histogram.max).into_segmented(),
        0..histogram.max_count)?;
    chart_context.configure_mesh().draw()?;
    chart_context.draw_series(histogram.value_counts.iter().map(|(k, v)| {
        let x0 = SegmentValue::Exact(*k as i32);
        let x1 = SegmentValue::Exact((*k+1) as i32);
        let mut bar = Rectangle::new(
            [(x0, 0), (x1, *v as i32)],
            BLUE.filled(),
        );
        bar.set_margin(0, 0, bar_margin, bar_margin);
        bar
    }))?;
    Ok(())
}

fn ccdf<'a, DB: DrawingBackend>(area: &DrawingArea<DB, Shift>, histogram: &Histogram) -> DrawResult<(), DB> {
    let mut vals: Vec<_> = histogram.value_counts.iter().collect();
    vals.sort_by_key(|&(val, _)| *val);
    let mut ccdf_points = Vec::new();
    let mut cumulative = 0;
    for &(val, count) in vals.iter().rev() {
        cumulative += count;
        let prob = (cumulative as f64) / (histogram.count as f64);
        ccdf_points.push((*val, prob));
    }
    ccdf_points.reverse();
    let mut chart_builder = ChartBuilder::on(area);
    chart_builder.margin(5).set_left_and_bottom_label_area_size(20);
    let mut chart_context = chart_builder.build_cartesian_2d(
        (1..histogram.max).log_scale(),
        0.0..1.0)?;
    chart_context.configure_mesh().draw()?;
    chart_context.draw_series(
        ccdf_points
            .into_iter()
            .map(|(x, y)| Circle::new((x as i32, y), 3, RED.filled())),
    )?;
    Ok(())
}

#[derive(PartialEq, Clone, Debug)]
pub struct Histogram {
    pub value_counts: HashMap<usize, usize>,
    pub total: i32,
    pub count: i32,
    pub max: i32,
    pub max_count: i32,
    pub mean: f32,
    pub median: f32,
}

impl Histogram {
    pub fn from_value_counts(value_to_count: &HashMap<usize, usize>) -> Histogram {
        let mut val_counts = value_to_count.iter().map(|(v, c)| (*v as i32, *c as i32)).collect::<Vec<_>>();
        val_counts.sort();
        let total = val_counts.iter().fold(0, |n, (v, c)| n + v*c);
        let count = val_counts.iter().fold(0, |n, (_, c)| n + c);
        let max = val_counts.iter().fold(0, |n, (v, _)| std::cmp::max(*v, n));
        let max_count = val_counts.iter().fold(0, |n, (_, c)| std::cmp::max(*c, n));
        let mean = (total as f32)/(count as f32);
        let median_lo_index = (count - 1) / 2;
        let median_hi_index = count / 2;
        let mut median_lo = None;
        let mut median_hi = None;
        let mut n = 0;
        for (v, c) in val_counts {
            let next_n = n + c;
            if median_lo.is_none() && median_lo_index < next_n {
                median_lo = Some(v);
            }
            if median_hi.is_none() && median_hi_index < next_n {
                median_hi = Some(v);
            }
            n = next_n;
            if median_lo.is_some() && median_hi.is_some() {
                break;
            }
        }
        let median = (median_lo.unwrap_or(0) as f32 + median_hi.unwrap_or(0) as f32)/2.0;
        Histogram { value_counts: value_to_count.clone(), total, count, max, max_count, mean, median }
    }
}

enum SampleState {
    Never,
    AtEnd,
    EveryN(usize, usize),
    Probability(Bernoulli, ThreadRng),
    Time(Duration, SystemTime),
}

pub struct Sample {
    state: SampleState,
}

impl Sample {
    pub fn never() -> Self {
        Self { state: SampleState::Never }
    }

    pub fn at_end() -> Self {
        Self { state: SampleState::AtEnd }
    }

    pub fn every_n(n: usize) -> Self {
        Self { state: SampleState::EveryN(n, 0) }
    }
    pub fn probability(p: f64) -> Self {
        Self {
            state: SampleState::Probability(Bernoulli::new(p).unwrap(), rng())
        }
    }

    pub fn time(every: Duration) -> Self {
        Self { state: SampleState::Time(every, SystemTime::now()) }
    }

    pub fn sample<U: UInt, S: State<U>>(&mut self, solver: &dyn DfsSolverView<U, S>) -> bool {
        match &mut self.state {
            SampleState::Never => false,
            SampleState::AtEnd => {
                solver.is_done()
            },
            SampleState::EveryN(n, count) => {
                *count += 1;
                if count >= n || solver.is_done() {
                    *count = 0;
                    true
                } else {
                    false
                }
            },
            SampleState::Probability(d, rng) => {
                d.sample(rng) || solver.is_done()
            },
            SampleState::Time(duration, last) => {
                let now = SystemTime::now();
                let elapsed = now.duration_since(*last).expect("Time went backwards!");
                if elapsed >= *duration || solver.is_done() {
                    *last = now;
                    true
                } else {
                    false
                }
            },
        }
    }
}

pub struct DbgObserver<U: UInt, S: State<U>> {
    print_sample: Sample,
    stat: Option<(String, Sample)>,
    certainty_streak: usize,
    certainty_hist: HashMap<usize, usize>,
    advance_hist: HashMap<usize, usize>,
    width_hist: HashMap<usize, usize>,
    backtrack_hist: HashMap<usize, usize>,
    filled_hist: HashMap<usize, usize>,
    steps: usize,
    _marker: std::marker::PhantomData<(U, S)>,
}

// TODO: Add correction_delay_hist
impl <U: UInt, S: State<U>> DbgObserver<U, S> {
    pub fn new() -> Self {
        DbgObserver {
            print_sample: Sample::every_n(1),
            stat: None,
            certainty_streak: 0,
            certainty_hist: HashMap::new(),
            advance_hist: HashMap::new(),
            width_hist: HashMap::new(),
            backtrack_hist: HashMap::new(),
            filled_hist: HashMap::new(),
            steps: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn sample_print(&mut self, sample: Sample) -> &mut Self {
        self.print_sample = sample;
        self
    }

    pub fn sample_stats<Str: Into<String>>(&mut self, filename: Str, sample: Sample) -> &mut Self {
        self.stat = Some((filename.into(), sample));
        self
    }

    fn update_stats(&mut self, solver: &dyn DfsSolverView<U, S>) {
        match solver.solver_state() {
            DfsSolverState::Advancing((n, w)) => {
                *self.advance_hist.entry(n).or_default() += 1;
                *self.width_hist.entry(w).or_default() += 1;
                if solver.constraint_result().has_certainty(solver.get_state()).is_some() {
                    self.certainty_streak += 1;
                }
            },
            DfsSolverState::Backtracking(n) => {
                *self.backtrack_hist.entry(n).or_default() += 1;
                if self.certainty_streak > 0 {
                    *self.certainty_hist.entry(self.certainty_streak).or_default() += 1;
                    self.certainty_streak = 0
                }
            },
            DfsSolverState::Solved => {
                if self.certainty_streak > 0 {
                    *self.certainty_hist.entry(self.certainty_streak + 1).or_default() += 1;
                    self.certainty_streak = 0;
                }
            },
            _ => {},
        }
        let mut filled = 0;
        for r in 0..S::ROWS {
            for c in 0..S::COLS {
                if solver.get_state().get([r, c]).is_some() {
                    filled += 1;
                }
            }
        }
        *self.filled_hist.entry(filled).or_default() += 1;
        self.steps += 1;
    }

    fn stats_figure<'a, DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>) -> DrawResult<(), DB> {
        area.fill(&WHITE)?;
        let (top, bottom) = area.split_vertically(50);
        let mut top_caption = MultiLineText::<_, String>::new((15, 15), ("sans-serif", 24).into_font());
        top_caption.push_line(format!("Steps: {}", self.steps));
        top.draw(&top_caption)?;
        let areas = bottom.split_evenly((3, 2));
        for (i, caption, value_counts, bar_margin) in vec![
            (0, "Num. choices at each advance", &self.width_hist, 5),
            (1, "Num. steps with N filled-in cells", &self.filled_hist, 0),
            (2, "Length of uninterrupted advances", &self.advance_hist, 1),
            (3, "Length of backtracks", &self.backtrack_hist, 1),
            (4, "Length of streaks of certainty", &self.certainty_hist, 1),
        ] {
            let hist = Histogram::from_value_counts(&value_counts);
            let (upper, lower) = areas[i].split_vertically(areas[i].relative_to_height(0.18));
            let mut extended_caption = MultiLineText::<_, String>::new((5, 5), ("sans-serif", 14).into_font());
            extended_caption.push_line(caption);
            extended_caption.push_line(format!(
                "E = {:.3}, lg2(E) = {:.3}, med = {:.1}, max = {}",
                hist.mean, hist.mean.log2(), hist.median, hist.max,
            ));
            upper.draw(&extended_caption)?;
            let (left, right) = lower.split_horizontally(lower.relative_to_width(0.5));
            bar_chart(&left, &hist, bar_margin)?;
            ccdf(&right, &hist)?;
        }
        Ok(())
    }

    pub fn dump_stats(&self, hist_filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        print!("Steps: {}\n", self.steps);
        let n_decisions = self.width_hist.iter().fold(0, |n, (_, count)| n+count);
        let total_choices = self.width_hist.iter().fold(0, |n, (w, count)| n+w*count);
        print!("Average Decision Width: {}\n", (total_choices as f64)/(n_decisions as f64));
        let area = BitMapBackend::new(hist_filename, (800, 1000)).into_drawing_area();
        self.stats_figure(&area).map_err(|e| {
            Box::new(e) as Box<dyn std::error::Error>
        })
    }

    pub fn print(&self, solver: &dyn DfsSolverView<U, S>) {
        let state = solver.get_state();
        if solver.is_initializing() {
            print!(
                "INITIALIZING: {:?}\n{:?}{:?}{}\n",
                solver.most_recent_action(), state, solver.get_constraint(),
                short_result::<U, S>(&solver.constraint_result(), state),
            );
        } else if solver.is_done() {
            if solver.is_valid() {
                print!(
                    "SOLVED: {:?}\n{:?}{:?}{}\n",
                    solver.most_recent_action(), state, solver.get_constraint(),
                    short_result::<U, S>(&solver.constraint_result(), state),
                );
            } else {
                print!("UNSOLVABLE");
            }
        } else {
            print!(
                "STEP: {:?}\n{:?}{:?}{}\n",
                solver.most_recent_action(), state, solver.get_constraint(),
                short_result::<U, S>(&solver.constraint_result(), state),
            );
        }
    }
}

impl <U: UInt, S: State<U>> StepObserver<U, S> for DbgObserver<U, S> {
    fn after_step(&mut self, solver: &dyn DfsSolverView<U, S>) {
        self.update_stats(solver);
        if self.print_sample.sample(solver) {
            self.print(solver);
        }
        if let Some((f, s)) = &mut self.stat {
            let filename = f.clone();
            if s.sample(solver) {
                self.dump_stats(&filename)
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to dump stats: {}\n", e)
                    });
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn to_counter(vals: Vec<usize>) -> HashMap<usize, usize> {
        let mut counter = HashMap::new();
        for v in vals {
            *counter.entry(v).or_default() += 1;
        }
        counter
    }

    #[test]
    fn test_dist_stat() {
        for hist in vec![
            Histogram {
                value_counts: to_counter(vec![2, 2, 3, 4, 4]),
                total: 15,
                count: 5,
                max: 4,
                max_count: 2,
                mean: 3.0,
                median: 3.0,
            },
            Histogram {
                value_counts: to_counter(vec![2, 2, 3, 3, 3, 4]),
                total: 17,
                count: 6,
                max: 4,
                max_count: 3,
                mean: 17.0/6.0,
                median: 3.0,
            },
            Histogram {
                value_counts: to_counter(vec![2, 3, 3, 4, 4, 4]),
                total: 20,
                count: 6,
                max: 4,
                max_count: 3,
                mean: 20.0/6.0,
                median: 3.5,
            },
        ] {
            let actual = Histogram::from_value_counts(&hist.value_counts);
            assert_eq!(actual, hist);
        }
    }
}