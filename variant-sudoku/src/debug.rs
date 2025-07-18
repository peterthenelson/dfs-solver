use std::{collections::HashMap, fs::File, io::Write, time::{Duration, SystemTime}};
use rand::{distr::{Bernoulli, Distribution}, rng, rngs::ThreadRng};
use crate::{constraint::Constraint, core::{ConstraintResult, Overlay, Value}, ranker::Ranker};
use crate::solver::{DfsSolverState, DfsSolverView, StepObserver};
use plotters::{chart::ChartBuilder, coord::Shift, prelude::{BitMapBackend, Circle, DrawResult, DrawingArea, DrawingBackend, IntoDrawingArea, IntoLogRange, IntoSegmentedCoord, MultiLineText, Rectangle, SegmentValue}, style::{Color, IntoFont, BLUE, RED, WHITE}};
use serde_derive::{Serialize, Deserialize};

pub struct NullObserver;

impl <V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O>>
StepObserver<V, O, R, C> for NullObserver {
    fn after_step(&mut self, _solver: &dyn DfsSolverView<V, O, R, C>) {}
}

fn short_result<V: Value, O: Overlay>(result: &ConstraintResult<V>) -> String {
    match result {
        ConstraintResult::Contradiction(a) => {
            format!("Contradiction({})", a.name())
        },
        ConstraintResult::Certainty(d, a) => {
            format!("Certainty({:?}, {:?}, {})", d.index, d.value, a.name())
        },
        ConstraintResult::Ok => "Ok".to_string(),
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

enum TimerState {
    Init,
    // With the time it was started
    Running(SystemTime),
    // With the duration from start to end
    Ended(Duration),
}

impl TimerState {
    fn new() -> Self { Self::Init }

    fn start(&mut self) {
        if let TimerState::Init = self {
            *self = TimerState::Running(SystemTime::now());
        } else {
            panic!("TimerState cannot be started if not in Init state.")
        }
    }

    fn end(&mut self) {
        if let TimerState::Running(s) = self {
            *self = TimerState::Ended(
                SystemTime::now().duration_since(*s).expect("Time went backwards!")
            );
        } else {
            panic!("TimerState cannot be ended if not in Running state.")
        }
    }

    fn to_duration(&self) -> Duration {
        match self {
            TimerState::Init => Duration::new(0, 0),
            TimerState::Running(s) => SystemTime::now().duration_since(*s).expect("Time went backwards!"),
            TimerState::Ended(d) => *d,
        }
    }
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

#[derive(Serialize, Deserialize)]
pub struct HistogramSummary {
    pub total: i32,
    pub count: i32,
    pub max: i32,
    pub max_count: i32,
    pub mean: f32,
    pub median: f32,
}

impl HistogramSummary {
    pub fn default() -> Self {
        Self {
            total: 0,
            count: 0,
            max: 0,
            max_count: 0,
            mean: 0.0,
            median: 0.0,
        }
    }

    pub fn delta_from(&self, base: &HistogramSummary) -> HistogramSummary {
        HistogramSummary {
            total: self.total - base.total,
            count: self.total - base.total,
            max: self.max - base.max,
            max_count: self.max_count - base.max_count,
            mean: self.mean - base.mean,
            median: self.median - base.median,
        }
    }
}

impl Histogram {
    pub fn from_value_counts(value_to_count: &HashMap<usize, usize>) -> Histogram {
        let mut val_counts = value_to_count.iter().map(|(v, c)| (*v as i32, *c as i32)).collect::<Vec<_>>();
        val_counts.sort();
        let total = val_counts.iter().fold(0, |n, (v, c)| n + v*c);
        let count = val_counts.iter().fold(0, |n, (_, c)| n + c);
        let max = val_counts.iter().fold(0, |n, (v, _)| std::cmp::max(*v, n));
        let max_count = val_counts.iter().fold(0, |n, (_, c)| std::cmp::max(*c, n));
        let mean = if count > 0 { (total as f32)/(count as f32) } else { 0.0 };
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

    fn summary(&self) -> HistogramSummary {
        HistogramSummary {
            total: self.total,
            count: self.count,
            max: self.max,
            max_count: self.max_count,
            mean: self.mean,
            median: self.median,
        }
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

    pub fn sample<V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O>>(
        &mut self, solver: &dyn DfsSolverView<V, O, R, C>,
    ) -> bool {
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

pub struct DbgObserver<V: Value, O: Overlay> {
    timer: TimerState,
    print_sample: Sample,
    stat: Option<(String, String, Sample)>,
    certainty_streak: usize,
    certainty_streak_hist: HashMap<usize, usize>,
    advance_streak_hist: HashMap<usize, usize>,
    decision_width_hist: HashMap<usize, usize>,
    backtrack_streak_hist: HashMap<usize, usize>,
    mistake_delay_hist: HashMap<usize, usize>,
    n_filled_hist: HashMap<usize, usize>,
    prev_state: Option<DfsSolverState>,
    streak: usize,
    steps: usize,
    _marker: std::marker::PhantomData<(V, O)>,
}

#[derive(Serialize, Deserialize)]
pub struct StatsSummary {
    steps: i32,
    seconds: f64,
    decision_width: HistogramSummary,
    mistake_delay: HistogramSummary,
    n_filled: HistogramSummary,
    certainty_streak: HistogramSummary,
    advance_streak: HistogramSummary,
    backtrack_streak: HistogramSummary,
}

impl StatsSummary {
    pub fn default() -> Self {
        Self {
            steps: 0,
            seconds: 0.0,
            decision_width: HistogramSummary::default(),
            mistake_delay: HistogramSummary::default(),
            n_filled: HistogramSummary::default(),
            certainty_streak: HistogramSummary::default(),
            advance_streak: HistogramSummary::default(),
            backtrack_streak: HistogramSummary::default(),
        }
    }

    pub fn delta_from(&self, base: &StatsSummary) -> StatsSummary {
        StatsSummary {
            steps: self.steps - base.steps,
            seconds: self.seconds - base.seconds,
            decision_width: self.decision_width.delta_from(&base.decision_width),
            mistake_delay: self.mistake_delay.delta_from(&base.mistake_delay),
            n_filled: self.n_filled.delta_from(&base.n_filled),
            certainty_streak: self.certainty_streak.delta_from(&base.certainty_streak),
            advance_streak: self.advance_streak.delta_from(&base.advance_streak),
            backtrack_streak: self.backtrack_streak.delta_from(&base.backtrack_streak),
        }
    }
}

impl <V: Value, O: Overlay> DbgObserver<V, O> {
    pub fn new() -> Self {
        DbgObserver {
            timer: TimerState::new(),
            print_sample: Sample::every_n(1),
            stat: None,
            certainty_streak: 0,
            certainty_streak_hist: HashMap::new(),
            advance_streak_hist: HashMap::new(),
            decision_width_hist: HashMap::new(),
            backtrack_streak_hist: HashMap::new(),
            mistake_delay_hist: HashMap::new(),
            n_filled_hist: HashMap::new(),
            prev_state: None,
            streak: 0,
            steps: 0,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn sample_print(&mut self, sample: Sample) -> &mut Self {
        self.print_sample = sample;
        self
    }

    pub fn sample_stats<Str: Into<String>>(&mut self, figure_file: Str, json_file: Str, sample: Sample) -> &mut Self {
        self.stat = Some((figure_file.into(), json_file.into(), sample));
        self
    }

    fn update_stats<R: Ranker<V, O>, C: Constraint<V, O>>(&mut self, solver: &dyn DfsSolverView<V, O, R, C>) {
        match solver.solver_state() {
            DfsSolverState::Advancing(state) => {
                if let Some(DfsSolverState::Advancing(_)) = self.prev_state {
                    self.streak += 1;
                } else {
                    self.streak = 1;
                }
                *self.advance_streak_hist.entry(self.streak).or_default() += 1;
                *self.decision_width_hist.entry(state.possibilities).or_default() += 1;
                if let ConstraintResult::Certainty(_, _) = solver.constraint_result() {
                    self.certainty_streak += 1;
                }
            },
            DfsSolverState::Backtracking(_) => {
                if let Some(DfsSolverState::Backtracking(_)) = self.prev_state {
                    self.streak += 1;
                } else {
                    self.streak = 1;
                }
                *self.backtrack_streak_hist.entry(self.streak).or_default() += 1;
                if self.certainty_streak > 0 {
                    *self.certainty_streak_hist.entry(self.certainty_streak).or_default() += 1;
                    self.certainty_streak = 0
                }
            },
            DfsSolverState::Solved => {
                if self.certainty_streak > 0 {
                    *self.certainty_streak_hist.entry(self.certainty_streak + 1).or_default() += 1;
                    self.certainty_streak = 0;
                }
            },
            _ => {},
        }
        self.prev_state = Some(solver.solver_state());
        if let Some(backtracked_steps) = solver.backtracked_steps() {
            *self.mistake_delay_hist.entry(backtracked_steps).or_default() += 1;
        }
        let mut filled = 0;
        let (n, m) = solver.state().overlay().grid_dims();
        for r in 0..n {
            for c in 0..m {
                if solver.state().get([r, c]).is_some() {
                    filled += 1;
                }
            }
        }
        *self.n_filled_hist.entry(filled).or_default() += 1;
        self.steps += 1;
    }

    fn stats_json(&self, json_filename: &str, summary: &StatsSummary) -> Result<(), std::io::Error> {
        let mut f = File::create(json_filename)?;
        let json_data = serde_json::to_string_pretty(&summary)?;
        f.write_all(json_data.as_bytes())?;
        Ok(())
    }

    fn stats_figure<'a, DB: DrawingBackend>(&self, area: &DrawingArea<DB, Shift>) -> DrawResult<(), DB> {
        area.fill(&WHITE)?;
        let (top, bottom) = area.split_vertically(50);
        let mut top_caption = MultiLineText::<_, String>::new((15, 15), ("sans-serif", 24).into_font());
        top_caption.push_line(format!(
            "Steps: {}; Seconds elapsed: {}", self.steps,
            self.timer.to_duration().as_secs_f64(),
        ));
        top.draw(&top_caption)?;
        let areas = bottom.split_evenly((3, 2));
        for (i, caption, value_counts, bar_margin) in vec![
            (0, "Num. choices at each advance", &self.decision_width_hist, 5),
            (1, "Mistake/backtrack delay", &self.mistake_delay_hist, 1),
            (2, "Num. steps with N filled-in cells", &self.n_filled_hist, 0),
            (3, "Certainty streaks", &self.certainty_streak_hist, 1),
            (4, "Advance streaks", &self.advance_streak_hist, 1),
            (5, "Backtrack streaks", &self.backtrack_streak_hist, 1),
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

    pub fn stats_summary(&self) -> StatsSummary {
        StatsSummary {
            steps: self.steps as i32,
            seconds: self.timer.to_duration().as_secs_f64(),
            decision_width: Histogram::from_value_counts(&self.decision_width_hist).summary(),
            n_filled: Histogram::from_value_counts(&self.n_filled_hist).summary(),
            advance_streak: Histogram::from_value_counts(&self.advance_streak_hist).summary(),
            certainty_streak: Histogram::from_value_counts(&self.certainty_streak_hist).summary(),
            backtrack_streak: Histogram::from_value_counts(&self.backtrack_streak_hist).summary(),
            mistake_delay: Histogram::from_value_counts(&self.mistake_delay_hist).summary(),
        }
    }

    pub fn dump_stats(&self, figure_filename: &str, json_filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let stats = self.stats_summary();
        print!("Steps: {}\n", stats.steps);
        print!("Time elapsed: {}\n", stats.seconds);
        print!("Average Decision Width: {}\n", stats.decision_width.mean);
        let area = BitMapBackend::new(figure_filename, (800, 1000)).into_drawing_area();
        self.stats_json(json_filename, &stats).map_err(|s| {
            let e: Box<dyn std::error::Error> = s.into();
            e
        })?;
        self.stats_figure(&area).map_err(|de| {
            let e: Box<dyn std::error::Error> = Box::new(de);
            e
        })
    }

    pub fn print<R: Ranker<V, O>, C: Constraint<V, O>>(&self, solver: &dyn DfsSolverView<V, O, R, C>) {
        let state = solver.state();
        if solver.is_initializing() {
            print!(
                "INITIALIZING: {:?}; {} elapsed\n{:?}{:?}{}\n",
                solver.most_recent_action(), self.timer.to_duration().as_secs_f64(),
                state, solver.constraint(),
                short_result::<V, O>(&solver.constraint_result()),
            );
        } else if solver.is_done() {
            match solver.solver_state() {
                DfsSolverState::Solved => {
                    print!(
                        "SOLVED: {:?}; {} elapsed\n{:?}{:?}{}\n",
                        solver.most_recent_action(), self.timer.to_duration().as_secs_f64(),
                        state, solver.constraint(),
                        short_result::<V, O>(&solver.constraint_result()),
                    );
                },
                DfsSolverState::Exhausted => print!("EXHAUSTED\n"),
                DfsSolverState::InitializationFailed => print!("INITIALIZATION FAILED\n"),
                _ => panic!("This should be unreachable!"),
            };
        } else {
            print!(
                "STEP: {:?}; {} elapsed\n{:?}{:?}{}\n",
                solver.most_recent_action(), self.timer.to_duration().as_secs_f64(),
                state, solver.constraint(),
                short_result::<V, O>(&solver.constraint_result()),
            );
        }
    }
}

impl <V: Value, O: Overlay, R: Ranker<V, O>, C: Constraint<V, O>>
StepObserver<V, O, R, C> for DbgObserver<V, O> {
    fn after_step(&mut self, solver: &dyn DfsSolverView<V, O, R, C>) {
        if let TimerState::Init = self.timer {
            self.timer.start();
        }
        if solver.is_done() {
            self.timer.end();
        }
        self.update_stats(solver);
        if self.print_sample.sample(solver) {
            self.print(solver);
        }
        if let Some((f, j, s)) = &mut self.stat {
            let (figure_filename, json_filename) = (f.clone(), j.clone());
            if s.sample(solver) {
                self.dump_stats(&figure_filename, &json_filename)
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