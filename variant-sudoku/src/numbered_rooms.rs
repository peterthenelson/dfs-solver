use std::fmt::Debug;
use crate::{constraint::Constraint, core::{Attribution, CertainDecision, ConstraintResult, Error, Feature, Index, Key, State, Stateful, VBitSet, VSet, VSetMut, Value}, ranker::RankingInfo, sudoku::{StdOverlay, StdVal}};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumberedRoomDirection {
    // Row, going right (starting from the left)
    RR,
    // Row, going left (starting from the right)
    RL,
    // Row, double ended
    RB,
    // Column, going down (starting from the top)
    CD,
    // Column, going up (starting from the bottom)
    CU,
    // Column, double ended
    CB,
}

impl NumberedRoomDirection {
    pub fn is_row(&self) -> bool {
        match self {
            NumberedRoomDirection::RR | NumberedRoomDirection::RL | NumberedRoomDirection::RB => true,
            _ => false,
        }
    }

    pub fn is_col(&self) -> bool {
        match self {
            NumberedRoomDirection::CD | NumberedRoomDirection::CU | NumberedRoomDirection::CB => true,
            _ => false,
        }
    }

    pub fn is_two_sided(&self) -> bool {
        match self {
            NumberedRoomDirection::RB | NumberedRoomDirection::CB => true,
            _ => false,
        }
    }
}

#[derive(Clone)]
pub struct NumberedRoom<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    number: u8,
    index: usize,
    direction: NumberedRoomDirection,
    room_indices: Vec<Index>,
    possible_room_values: VBitSet<StdVal<MIN, MAX>>,
    impossible_positions: Vec<Index>,
    possibilities: Vec<Vec<(Index, StdVal<MIN, MAX>)>>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> Debug for NumberedRoom<MIN, MAX, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NumberedRoom({}, {:?}, {})", self.number, self.index, match self.direction {
            NumberedRoomDirection::RR => "RR",
            NumberedRoomDirection::RL => "RL",
            NumberedRoomDirection::RB => "RB",
            NumberedRoomDirection::CD => "CD",
            NumberedRoomDirection::CU => "CU",
            NumberedRoomDirection::CB => "CB",
        })
    }
}

// Iterates over non-room indices of the row/column.
pub struct NumberedRoomIter<'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    room: &'a NumberedRoom<MIN, MAX, N, M>,
    i: usize,
}

impl <'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize>
NumberedRoomIter<'a, MIN, MAX, N, M> {
    pub fn new(room: &'a NumberedRoom<MIN, MAX, N, M>) -> Self {
        NumberedRoomIter { room, i: 0 }
    }
}

impl <'a, const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Iterator for NumberedRoomIter<'a, MIN, MAX, N, M> {
    type Item = Index;

    fn next(&mut self) -> Option<Self::Item> {
        match self.room.direction {
            NumberedRoomDirection::RR | NumberedRoomDirection::RL | NumberedRoomDirection::RB => {
                if self.i >= M {
                    return None;
                }
                let ret = [self.room.index, self.i];
                self.i += 1;
                if self.room.room_indices.contains(&ret) {
                    return self.next();
                }
                return Some(ret);
            },
            NumberedRoomDirection::CD | NumberedRoomDirection::CU | NumberedRoomDirection::CB => {
                if self.i >= N {
                    return None;
                }
                let ret = [self.i, self.room.index];
                self.i += 1;
                if self.room.room_indices.contains(&ret) {
                    return self.next();
                }
                return Some(ret);
            },
        }
    }
}

fn possible_room_values_raw<const MIN: u8, const MAX: u8, const N: usize, const M: usize>(
    number: u8,
    direction: NumberedRoomDirection,
) -> VBitSet<StdVal<MIN, MAX>> {
    if number == 1 {
        return VBitSet::<StdVal<MIN, MAX>>::full();
    }
    let full = StdVal::<MIN, MAX>::possibilities();
    let pos_vals = match direction {
        NumberedRoomDirection::RR | NumberedRoomDirection::CD |
        NumberedRoomDirection::RL | NumberedRoomDirection::CU => {
            vec![StdVal::new(MIN)]
        },
        NumberedRoomDirection::RB | NumberedRoomDirection::CB => vec![
            StdVal::new(MIN),
            StdVal::new(MAX),
        ],
    };
    let self_vals = match direction {
        NumberedRoomDirection::RR | NumberedRoomDirection::CD |
        NumberedRoomDirection::RL | NumberedRoomDirection::CU => {
            vec![StdVal::new(number)]
        },
        NumberedRoomDirection::RB | NumberedRoomDirection::CB => vec![
            StdVal::new(number),
            StdVal::new(MAX + 1 - number),
        ],
    };
    VBitSet::<StdVal<MIN, MAX>>::from_values(
        &full.into_iter().filter(|v| {
            !pos_vals.contains(v) && !self_vals.contains(v)
        }).collect()
    )
}

fn possibilities_raw<const MIN: u8, const MAX: u8, const N: usize, const M: usize>(
    number: u8,
    index: usize,
    direction: NumberedRoomDirection,
) -> Vec<Vec<(Index, StdVal<MIN, MAX>)>> {
    let mut result = Vec::new();
    for v in StdVal::<MIN, MAX>::possibilities() {
        if number != 1 && (v.val() == 1 || v.val() == number) {
            result.push(Vec::new());
            continue;
        }
        let num_val = StdVal::<MIN, MAX>::new(number);
        let conv = StdVal::<MIN, MAX>::new(MAX + 1 - v.val());
        if number != 1 && direction.is_two_sided() && (conv.val() == 1 || conv.val() == number) {
            result.push(Vec::new());
            continue;
        }
        result.push(match direction {
            NumberedRoomDirection::RR => vec![
                ([index, 0], v),
                ([index, (v.val() - 1) as usize], num_val),
            ],
            NumberedRoomDirection::RL => vec![
                ([index, M - 1], v),
                ([index, M - v.val() as usize], num_val),
            ],
            NumberedRoomDirection::RB => vec![
                ([index, 0], v),
                ([index, (v.val() - 1) as usize], num_val),
                ([index, M - 1], conv),
            ],
            NumberedRoomDirection::CD => vec![
                ([0, index], v),
                ([(v.val() - 1) as usize, index], num_val),
            ],
            NumberedRoomDirection::CU => vec![
                ([M - 1, index], v),
                ([M - v.val() as usize, index], num_val),
            ],
            NumberedRoomDirection::CB => vec![
                ([0, index], v),
                ([(v.val() - 1) as usize, index], num_val),
                ([M - 1, index], conv),
            ],
        });
    }
    result
}

fn impossible_positions_raw<const MIN: u8, const MAX: u8, const N: usize, const M: usize>(
    number: u8,
    index: usize,
    direction: NumberedRoomDirection,
) -> Vec<Index> {
    let mut impossible = Vec::new();
    if number == 1 {
        return impossible
    }
    impossible.extend(match direction {
        NumberedRoomDirection::RR => vec![[index, number as usize - 1]],
        NumberedRoomDirection::RL => vec![[index, M - number as usize]],
        NumberedRoomDirection::RB => vec![
            [index, number as usize - 1],
            [index, M - number as usize]
        ],
        NumberedRoomDirection::CD => vec![[number as usize - 1, index]],
        NumberedRoomDirection::CU => vec![[N - number as usize, index]],
        NumberedRoomDirection::CB => vec![
            [number as usize - 1, index],
            [N - number as usize, index],
        ],
    });
    impossible.extend(match direction {
        NumberedRoomDirection::RR => vec![[index, 0]],
        NumberedRoomDirection::RL => vec![[index, M - 1]],
        NumberedRoomDirection::RB => vec![[index, 0], [index, M - 1]],
        NumberedRoomDirection::CD => vec![[0, index]],
        NumberedRoomDirection::CU => vec![[N - 1, index]],
        NumberedRoomDirection::CB => vec![[0, index], [N - 1, index]],
    });
    impossible.sort();
    let mut impossible_uniq = Vec::new();
    for i in impossible {
        if !impossible_uniq.contains(&i) {
            impossible_uniq.push(i);
        }
    }
    impossible_uniq
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
NumberedRoom<MIN, MAX, N, M> {
    pub fn new(number: u8, index: usize, direction: NumberedRoomDirection) -> Self {
        // Would be nice to have this be static, but we can't.
        assert!(MIN == 1, "MIN must be 1 for some numbered room logic to make sense");
        Self {
            number, index, direction,
            room_indices: match direction {
                NumberedRoomDirection::RR => vec![[index, 0]],
                NumberedRoomDirection::RL => vec![[index, M - 1]],
                NumberedRoomDirection::RB => vec![[index, 0], [index, M - 1]],
                NumberedRoomDirection::CD => vec![[0, index]],
                NumberedRoomDirection::CU => vec![[N - 1, index]],
                NumberedRoomDirection::CB => vec![[0, index], [N - 1, index]],
            },
            possible_room_values: possible_room_values_raw::<MIN, MAX, N, M>(number, direction),
            impossible_positions: impossible_positions_raw::<MIN, MAX, N, M>(number, index, direction),
            possibilities: possibilities_raw::<MIN, MAX, N, M>(number, index, direction),
        }
    }

    pub fn contains(&self, index: Index) -> bool {
        match self.direction {
            NumberedRoomDirection::RR | NumberedRoomDirection::RL | NumberedRoomDirection::RB => {
                index[0] == self.index
            },
            NumberedRoomDirection::CD | NumberedRoomDirection::CU | NumberedRoomDirection::CB => {
                index[1] == self.index
            },
        }
    }

    pub fn relevant_possibility(&self, index: Index, value: &StdVal<MIN, MAX>) -> Option<usize> {
        if !self.contains(index) {
            return None;
        }
        if value.val() == self.number {
            return Some(match self.direction {
                NumberedRoomDirection::RR => index[1],
                NumberedRoomDirection::RL => M - index[1] - 1,
                NumberedRoomDirection::RB => index[1],
                NumberedRoomDirection::CD => index[0],
                NumberedRoomDirection::CU => N - index[0] - 1,
                NumberedRoomDirection::CB => index[0],
            });
        }
        if self.room_indices[0] == index {
            return Some((value.val() - 1) as usize);
        } else if self.direction.is_two_sided() && self.room_indices[1] == index {
            return Some(if self.direction.is_row() {
                M - value.val() as usize
            } else {
                N - value.val() as usize
            });
        }
        None
    }

    pub fn room_indices(&self) -> &Vec<Index> {
        &self.room_indices
    }

    pub fn possible_room_values(&self) -> &VBitSet<StdVal<MIN, MAX>> {
        &self.possible_room_values
    }

    pub fn impossible_positions(&self) -> &Vec<Index> {
        &self.impossible_positions
    }

    pub fn possibilities(&self) -> &Vec<Vec<(Index, StdVal<MIN, MAX>)>> {
        &self.possibilities
    }

    pub fn iter_non_room_indices(&self) -> NumberedRoomIter<'_, MIN, MAX, N, M> {
        NumberedRoomIter::new(self)
    }
}

pub const NUMBERED_ROOM_ILLEGAL_ACTION: Error = Error::new_const("A numbered room violation already exists; can't apply further actions.");
pub const NUMBERED_ROOM_UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");
pub const NUMBERED_ROOM_FEATURE: &str = "NUMBERED_ROOM";
pub const NUMBERED_ROOM_CONFLICT_ATTRIBUTION: &str = "NUMBERED_ROOM_CONFLICT";
pub const NUMBERED_ROOM_CERTAINTY_ATTRIBUTION: &str = "NUMBERED_ROOM_CERTAINTY";
pub const NUMBERED_ROOM_INFEASIBLE_ATTRIBUTION: &str = "NUMBERED_ROOM_INFEASIBLE";

pub struct NumberedRoomChecker<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    rooms: Vec<NumberedRoom<MIN, MAX, N, M>>,
    room_completion: Vec<(usize, usize)>,
    room_feature: Key<Feature>,
    room_conflict_attr: Key<Attribution>,
    room_certainty_attr: Key<Attribution>,
    room_if_attr: Key<Attribution>,
    illegal: Option<(Index, StdVal<MIN, MAX>, Key<Attribution>)>,
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> NumberedRoomChecker<MIN, MAX, N, M> {
    pub fn new(rooms: Vec<NumberedRoom<MIN, MAX, N, M>>) -> Self {
        let len = rooms.len();
        NumberedRoomChecker {
            rooms,
            room_completion: vec![(0, 0); len],
            room_feature: Key::register(NUMBERED_ROOM_FEATURE),
            room_conflict_attr: Key::register(NUMBERED_ROOM_CONFLICT_ATTRIBUTION),
            room_certainty_attr: Key::register(NUMBERED_ROOM_CERTAINTY_ATTRIBUTION),
            room_if_attr: Key::register(NUMBERED_ROOM_INFEASIBLE_ATTRIBUTION),
            illegal: None,
        }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Debug for NumberedRoomChecker<MIN, MAX, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.illegal {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.name())?;
        }
        for (i, r) in self.rooms.iter().enumerate() {
            write!(f, "  {:?}\n", r)?;
            write!(f, "  - {}\n", r.possible_room_values().to_string())?;
            let (p_i, n) = self.room_completion[i];
            if n == 0 {
                write!(f, "  - Not started\n")?;
            } else if n == r.possibilities()[p_i].len() {
                write!(f, "  - Complete\n")?;
            } else {
                write!(f, "  - Partial ({}/{})\n", n, r.possibilities()[p_i].len())?;
            }
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Stateful<StdVal<MIN, MAX>> for NumberedRoomChecker<MIN, MAX, N, M> {
    fn reset(&mut self) {
        self.room_completion = vec![(0, 0); self.rooms.len()];
    }

    fn apply(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        // In theory we could be allow multiple illegal moves and just
        // invalidate and recalculate the grid or something, but it seems hard.
        if self.illegal.is_some() {
            return Err(NUMBERED_ROOM_ILLEGAL_ACTION);
        }
        let mut completions = Vec::new();
        for (i, r) in self.rooms.iter().enumerate() {
            if let Some(p_i) = r.relevant_possibility(index, &value) {
                let p = &r.possibilities()[p_i];
                if p.is_empty() {
                    self.illegal = Some((index, value, self.room_if_attr));
                    return Ok(());
                }
                if self.room_completion[i].1 == 0 || self.room_completion[i].0 == p_i {
                    completions.push((i, p_i));
                } else {
                    self.illegal = Some((index, value, self.room_conflict_attr));
                    return Ok(());
                }
            }
        }
        for (i, p_i) in completions {
            self.room_completion[i].0 = p_i;
            self.room_completion[i].1 += 1;
        }
        Ok(())
    }

    fn undo(&mut self, index: Index, value: StdVal<MIN, MAX>) -> Result<(), Error> {
        if let Some((i, v, _)) = self.illegal {
            if i != index || v != value {
                return Err(NUMBERED_ROOM_UNDO_MISMATCH);
            } else {
                self.illegal = None;
                return Ok(());
            }
        }
        for (i, r) in self.rooms.iter().enumerate() {
            if let Some(_) = r.relevant_possibility(index, &value) {
                self.room_completion[i].1 -= 1;
                if self.room_completion[i].1 == 0 {
                    self.room_completion[i].0 = 0;
                }
            }
        }
        Ok(())
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<StdVal<MIN, MAX>, StdOverlay<N, M>> for NumberedRoomChecker<MIN, MAX, N, M> {
    fn name(&self) -> Option<String> { Some("NumberedRoomChecker".to_string()) }

    fn check(&self, puzzle: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, ranking: &mut RankingInfo<StdVal<MIN, MAX>>) -> ConstraintResult<StdVal<MIN, MAX>> {
        if let Some((_, _, a)) = &self.illegal {
            return ConstraintResult::Contradiction(*a);
        }
        let grid = ranking.cells_mut();
        for (i, room) in self.rooms.iter().enumerate() {
            let (p_i, n) = self.room_completion[i];
            if n > 0 {
                for (index, val) in &room.possibilities()[p_i] {
                    if let Some(v) = puzzle.get(*index) {
                        if v != *val {
                            return ConstraintResult::Contradiction(
                                self.room_conflict_attr,
                            );
                        }
                    } else {
                        return ConstraintResult::Certainty(
                            CertainDecision::new(*index, *val),
                            self.room_certainty_attr,
                        );
                    }
                }
            }
            for cell in room.room_indices() {
                if puzzle.get(*cell).is_some() {
                    continue;
                }
                let g = grid.get_mut(*cell);
                g.0.intersect_with(room.possible_room_values());
                g.1.add(&self.room_feature, 1.0);
            }
            let v = StdVal::<MIN, MAX>::new(room.number);
            for cell in room.impossible_positions() {
                if puzzle.get(*cell).is_some() {
                    continue;
                }
                let g = grid.get_mut(*cell);
                g.0.remove(&v);
            }
        }
        ConstraintResult::Ok
    }

    fn debug_at(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<String> {
        let header = "NumberedRoomChecker:\n";
        let mut lines = vec![];
        if let Some((i, v, a)) = &self.illegal {
            if *i == index {
                lines.push(format!("  Illegal move: {:?}={:?} ({})", i, v, a.name()));
            }
        }
        for (i, r) in self.rooms.iter().enumerate() {
            if !r.contains(index) {
                continue;
            }
            lines.push(format!("  {:?}", r));
            lines.push(format!("  - {}", r.possible_room_values().to_string()));
            let (p_i, n) = self.room_completion[i];
            if n == 0 {
                lines.push("  - Not started".to_string());
            } else if n == r.possibilities()[p_i].len() {
                lines.push("  - Complete".to_string());
            } else {
                lines.push(format!("  - Partial ({}/{})", n, r.possibilities()[p_i].len()));
            }
        }
        if lines.is_empty() {
            None
        } else {
            Some(format!("{}{}", header, lines.join("\n")))
        }
    }

    fn debug_highlight(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<(u8, u8, u8)> {
        for r in &self.rooms {
            if !r.contains(index) {
                continue;
            }
            if r.room_indices.contains(&index) {
                return Some((200, 0, 200));
            } else {
                return Some((0, 200, 0));
            }
        }
        None
    }
}

#[cfg(test)]
mod test {
    use crate::{constraint::MultiConstraint, ranker::StdRanker, solver::{FindFirstSolution, PuzzleSetter}, sudoku::{four_standard_overlay, FourStd, FourStdOverlay, FourStdVal, SixStdVal, StdChecker}};
    use super::*;

    fn assert_missing_possible_room_vals<const MIN: u8, const MAX: u8, const N: usize, const M: usize>(
        number: u8,
        direction: NumberedRoomDirection,
        expected_missing: Vec<StdVal<MIN, MAX>>,
    ) {
        let room = NumberedRoom::<MIN, MAX, N, M>::new(number, 0, direction);
        let actual = room.possible_room_values();
        let full = StdVal::<MIN, MAX>::possibilities();
        let actual_missing: Vec<StdVal<MIN, MAX>> = full.into_iter()
            .filter(|v| !actual.contains(v))
            .collect();
        assert_eq!(
            actual_missing, expected_missing,
            "Missing values for {:?} direction with number {} do not match expected: {:?} != {:?}",
            direction, number, actual_missing, expected_missing
        )
    }

    #[test]
    fn test_single_ended_possible_room_vals() {
        for direction in [
            NumberedRoomDirection::RR,
            NumberedRoomDirection::RL,
            NumberedRoomDirection::CD,
            NumberedRoomDirection::CU,
        ] {
            // 1 can be the min (unlike other numbers), as well as
            // all the other ones.
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                1, direction, vec![],
            );
            // 2 through 6 can neither be in the position index nor their own position.
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                2, direction,
                vec![StdVal::new(1), StdVal::new(2)],
            );
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                4, direction,
                vec![StdVal::new(1), StdVal::new(4)],
            );
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                6, direction,
                vec![StdVal::new(1), StdVal::new(6)],
            );
        }
    }

    #[test]
    fn test_double_ended_possible_room_vals() {
        for direction in [
            NumberedRoomDirection::RB,
            NumberedRoomDirection::CB,
        ] {
            // 1 can be in either position index (unlike other numbers), as well as
            // all the other ones.
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                1, direction, vec![],
            );
            // 2 through 6 can neither be in neither of the position indices nor
            // their own position from either end.
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                2, direction,
                vec![StdVal::new(1), StdVal::new(2), StdVal::new(5), StdVal::new(6)],
            );
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                4, direction,
                vec![StdVal::new(1), StdVal::new(3), StdVal::new(4), StdVal::new(6)],
            );
            assert_missing_possible_room_vals::<1, 6, 6, 6>(
                6, direction,
                vec![StdVal::new(1), StdVal::new(6)],
            );
        }
    }

    fn assert_impossible_positions<const MIN: u8, const MAX: u8, const N: usize, const M: usize>(
        number: u8,
        direction: NumberedRoomDirection,
        expected: Vec<Index>,
    ) {
        let room = NumberedRoom::<MIN, MAX, N, M>::new(number, 0, direction);
        let actual = room.impossible_positions();
        assert_eq!(
            *actual, expected,
            "Impossible positions for {:?} direction with number {} do not match expected: {:?} != {:?}",
            direction, number, actual, expected,
        )
    }

    #[test]
    fn test_rr_impossible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_impossible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::RR, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_impossible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::RR,
            vec![[0, 0], [0, 1]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::RR,
            vec![[0, 0], [0, 3]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::RR,
            vec![[0, 0], [0, 5]],
        );
    }

    #[test]
    fn test_rl_impossible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_impossible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::RL, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_impossible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::RL,
            vec![[0, 4], [0, 5]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::RL,
            vec![[0, 2], [0, 5]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::RL,
            vec![[0, 0], [0, 5]],
        );
    }

    #[test]
    fn test_rb_impossible_positions() {
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_impossible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::RB, vec![],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_impossible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::RB,
            vec![[0, 0], [0, 1], [0, 4], [0, 5]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::RB,
            vec![[0, 0], [0, 2], [0, 3], [0, 5]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::RB,
            vec![[0, 0], [0, 5]],
        );
    }

    #[test]
    fn test_cd_impossible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_impossible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::CD, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_impossible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::CD,
            vec![[0, 0], [1, 0]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::CD,
            vec![[0, 0], [3, 0]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::CD,
            vec![[0, 0], [5, 0]],
        );
    }

    #[test]
    fn test_cu_impossible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_impossible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::CU, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_impossible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::CU,
            vec![[4, 0], [5, 0]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::CU,
            vec![[2, 0], [5, 0]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::CU,
            vec![[0, 0], [5, 0]],
        );
    }

    #[test]
    fn test_cb_impossible_positions() {
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_impossible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::CB, vec![],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_impossible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::CB,
            vec![[0, 0], [1, 0], [4, 0], [5, 0]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::CB,
            vec![[0, 0], [2, 0], [3, 0], [5, 0]],
        );
        assert_impossible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::CB,
            vec![[0, 0], [5, 0]],
        );
    }

    fn val(i: u8) -> SixStdVal {
        SixStdVal::new(i)
    }

    #[test]
    fn test_rr_possibilities() {
        let dir = NumberedRoomDirection::RR;
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(1, 0, dir).possibilities(),
            vec![
                vec![([0, 0], val(1)), ([0, 0], val(1))],
                vec![([0, 0], val(2)), ([0, 1], val(1))],
                vec![([0, 0], val(3)), ([0, 2], val(1))],
                vec![([0, 0], val(4)), ([0, 3], val(1))],
                vec![([0, 0], val(5)), ([0, 4], val(1))],
                vec![([0, 0], val(6)), ([0, 5], val(1))],
            ],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(3, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([0, 1], val(3))],
                vec![],
                vec![([0, 0], val(4)), ([0, 3], val(3))],
                vec![([0, 0], val(5)), ([0, 4], val(3))],
                vec![([0, 0], val(6)), ([0, 5], val(3))],
            ],
        );
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(6, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([0, 1], val(6))],
                vec![([0, 0], val(3)), ([0, 2], val(6))],
                vec![([0, 0], val(4)), ([0, 3], val(6))],
                vec![([0, 0], val(5)), ([0, 4], val(6))],
                vec![],
            ],
        );
    }

    #[test]
    fn test_rl_possibilities() {
        let dir = NumberedRoomDirection::RL;
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(1, 0, dir).possibilities(),
            vec![
                vec![([0, 5], val(1)), ([0, 5], val(1))],
                vec![([0, 5], val(2)), ([0, 4], val(1))],
                vec![([0, 5], val(3)), ([0, 3], val(1))],
                vec![([0, 5], val(4)), ([0, 2], val(1))],
                vec![([0, 5], val(5)), ([0, 1], val(1))],
                vec![([0, 5], val(6)), ([0, 0], val(1))],
            ],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(3, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 5], val(2)), ([0, 4], val(3))],
                vec![],
                vec![([0, 5], val(4)), ([0, 2], val(3))],
                vec![([0, 5], val(5)), ([0, 1], val(3))],
                vec![([0, 5], val(6)), ([0, 0], val(3))],
            ],
        );
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(6, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 5], val(2)), ([0, 4], val(6))],
                vec![([0, 5], val(3)), ([0, 3], val(6))],
                vec![([0, 5], val(4)), ([0, 2], val(6))],
                vec![([0, 5], val(5)), ([0, 1], val(6))],
                vec![],
            ],
        );
    }

    #[test]
    fn test_rb_possibilities() {
        let dir = NumberedRoomDirection::RB;
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(1, 0, dir).possibilities(),
            vec![
                vec![([0, 0], val(1)), ([0, 0], val(1)), ([0, 5], val(6))],
                vec![([0, 0], val(2)), ([0, 1], val(1)), ([0, 5], val(5))],
                vec![([0, 0], val(3)), ([0, 2], val(1)), ([0, 5], val(4))],
                vec![([0, 0], val(4)), ([0, 3], val(1)), ([0, 5], val(3))],
                vec![([0, 0], val(5)), ([0, 4], val(1)), ([0, 5], val(2))],
                vec![([0, 0], val(6)), ([0, 5], val(1)), ([0, 5], val(1))],
            ],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(3, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([0, 1], val(3)), ([0, 5], val(5))],
                vec![],
                vec![],
                vec![([0, 0], val(5)), ([0, 4], val(3)), ([0, 5], val(2))],
                vec![],
            ],
        );
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(6, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([0, 1], val(6)), ([0, 5], val(5))],
                vec![([0, 0], val(3)), ([0, 2], val(6)), ([0, 5], val(4))],
                vec![([0, 0], val(4)), ([0, 3], val(6)), ([0, 5], val(3))],
                vec![([0, 0], val(5)), ([0, 4], val(6)), ([0, 5], val(2))],
                vec![],
            ],
        );
    }

    #[test]
    fn test_cd_possibilities() {
        let dir = NumberedRoomDirection::CD;
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(1, 0, dir).possibilities(),
            vec![
                vec![([0, 0], val(1)), ([0, 0], val(1))],
                vec![([0, 0], val(2)), ([1, 0], val(1))],
                vec![([0, 0], val(3)), ([2, 0], val(1))],
                vec![([0, 0], val(4)), ([3, 0], val(1))],
                vec![([0, 0], val(5)), ([4, 0], val(1))],
                vec![([0, 0], val(6)), ([5, 0], val(1))],
            ],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(3, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([1, 0], val(3))],
                vec![],
                vec![([0, 0], val(4)), ([3, 0], val(3))],
                vec![([0, 0], val(5)), ([4, 0], val(3))],
                vec![([0, 0], val(6)), ([5, 0], val(3))],
            ],
        );
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(6, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([1, 0], val(6))],
                vec![([0, 0], val(3)), ([2, 0], val(6))],
                vec![([0, 0], val(4)), ([3, 0], val(6))],
                vec![([0, 0], val(5)), ([4, 0], val(6))],
                vec![],
            ],
        );
    }

    #[test]
    fn test_cu_possibilities() {
        let dir = NumberedRoomDirection::CU;
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(1, 0, dir).possibilities(),
            vec![
                vec![([5, 0], val(1)), ([5, 0], val(1))],
                vec![([5, 0], val(2)), ([4, 0], val(1))],
                vec![([5, 0], val(3)), ([3, 0], val(1))],
                vec![([5, 0], val(4)), ([2, 0], val(1))],
                vec![([5, 0], val(5)), ([1, 0], val(1))],
                vec![([5, 0], val(6)), ([0, 0], val(1))],
            ],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(3, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([5, 0], val(2)), ([4, 0], val(3))],
                vec![],
                vec![([5, 0], val(4)), ([2, 0], val(3))],
                vec![([5, 0], val(5)), ([1, 0], val(3))],
                vec![([5, 0], val(6)), ([0, 0], val(3))],
            ],
        );
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(6, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([5, 0], val(2)), ([4, 0], val(6))],
                vec![([5, 0], val(3)), ([3, 0], val(6))],
                vec![([5, 0], val(4)), ([2, 0], val(6))],
                vec![([5, 0], val(5)), ([1, 0], val(6))],
                vec![],
            ],
        );
    }

    #[test]
    fn test_cb_possibilities() {
        let dir = NumberedRoomDirection::CB;
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(1, 0, dir).possibilities(),
            vec![
                vec![([0, 0], val(1)), ([0, 0], val(1)), ([5, 0], val(6))],
                vec![([0, 0], val(2)), ([1, 0], val(1)), ([5, 0], val(5))],
                vec![([0, 0], val(3)), ([2, 0], val(1)), ([5, 0], val(4))],
                vec![([0, 0], val(4)), ([3, 0], val(1)), ([5, 0], val(3))],
                vec![([0, 0], val(5)), ([4, 0], val(1)), ([5, 0], val(2))],
                vec![([0, 0], val(6)), ([5, 0], val(1)), ([5, 0], val(1))],
            ],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(3, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([1, 0], val(3)), ([5, 0], val(5))],
                vec![],
                vec![],
                vec![([0, 0], val(5)), ([4, 0], val(3)), ([5, 0], val(2))],
                vec![],
            ],
        );
        assert_eq!(
            *NumberedRoom::<1, 6, 6, 6>::new(6, 0, dir).possibilities(),
            vec![
                vec![],
                vec![([0, 0], val(2)), ([1, 0], val(6)), ([5, 0], val(5))],
                vec![([0, 0], val(3)), ([2, 0], val(6)), ([5, 0], val(4))],
                vec![([0, 0], val(4)), ([3, 0], val(6)), ([5, 0], val(3))],
                vec![([0, 0], val(5)), ([4, 0], val(6)), ([5, 0], val(2))],
                vec![],
            ],
        );
    }

    #[test]
    fn test_relevant_possibility() {
        // One concrete example
        assert_eq!(
            NumberedRoom::<1, 6, 6, 6>::new(3, 0, NumberedRoomDirection::RR)
                .relevant_possibility([0, 3], &val(3)),
            Some(3),
        );
        // Enumerate others based on possibilities
        for dir in [
            NumberedRoomDirection::RR,
            NumberedRoomDirection::RL,
            NumberedRoomDirection::RB,
            NumberedRoomDirection::CD,
            NumberedRoomDirection::CU,
            NumberedRoomDirection::CB,
        ] {
            for num in 1..=6 {
                let room = NumberedRoom::<1, 6, 6, 6>::new(num, 0, dir);
                for (exp, p) in room.possibilities().iter().enumerate() {
                    for (index, value) in p {
                        if let Some(act) = room.relevant_possibility(*index, value) {
                            assert_eq!(
                                exp, act,
                                "Expected NumberedRoom({}, 0, {:?}).relevant_possibility({:?}, {}) \
                                 to be {}, but got {}", num, dir, index, value, exp, act,
                            );
                        } else {
                            panic!("Expected NumberedRoom({}, 0, {:?}).relevant_possibility({:?}, {}) \
                                    to be Some({}) but got None", num, dir, index, value, exp);
                        }
                    }
                }
            }
        }
    }

    struct E2ENumberedRoom;
    impl PuzzleSetter for E2ENumberedRoom {
        type Value = FourStdVal;
        type Overlay = FourStdOverlay;
        type Ranker = StdRanker<Self::Overlay>;
        type Constraint = MultiConstraint<Self::Value, Self::Overlay>;
        fn setup() -> (FourStd, Self::Ranker, Self::Constraint) {
            Self::setup_with_givens(FourStd::new(four_standard_overlay()))
        }
        fn setup_with_givens(given: FourStd) -> (FourStd, Self::Ranker, Self::Constraint) {
            // Solution is like this. No givens, but the numbered rooms
            // determine it.
            //    1 4 4
            //    v v v
            //  1>1 2|3 4<1
            //  2>3 4|2 1<1
            //    ---+---
            //  1>2 1|4 3<1
            //  2>4 3|1 2<1
            //    ^ ^ ^
            //    1 4 1
            let rooms = vec![
                NumberedRoom::new(1, 0, NumberedRoomDirection::RB),
                NumberedRoom::new(2, 1, NumberedRoomDirection::RR),
                NumberedRoom::new(1, 1, NumberedRoomDirection::RL),
                NumberedRoom::new(1, 2, NumberedRoomDirection::RB),
                NumberedRoom::new(2, 3, NumberedRoomDirection::RR),
                NumberedRoom::new(1, 3, NumberedRoomDirection::RL),
                NumberedRoom::new(1, 0, NumberedRoomDirection::CB),
                NumberedRoom::new(4, 1, NumberedRoomDirection::CB),
                NumberedRoom::new(4, 2, NumberedRoomDirection::CD),
                NumberedRoom::new(1, 2, NumberedRoomDirection::CU),
            ];
            let constraint = MultiConstraint::new(vec_box::vec_box![
                StdChecker::new(given.overlay()),
                NumberedRoomChecker::new(rooms),
            ]);
            (given, StdRanker::default(), constraint)
        }
    }

    #[test]
    fn test_e2e_numbered_room_example() -> Result<(), Error> {
        let (mut puzzle, ranker, mut constraint) = E2ENumberedRoom::setup();
        let mut finder = FindFirstSolution::new(&mut puzzle, &ranker, &mut constraint, None);
        let maybe_solution = finder.solve()?;
        assert!(maybe_solution.is_some());
        let expected: &str = "1 2|3 4\n\
                              3 4|2 1\n\
                              ---+---\n\
                              2 1|4 3\n\
                              4 3|1 2\n";
        let solution = maybe_solution.unwrap().state();
        assert_eq!(solution.overlay().serialize_pretty(&solution), expected);
        Ok(())
    }
}