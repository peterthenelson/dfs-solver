use std::fmt::Debug;
use crate::{constraint::Constraint, core::{ConstraintResult, Index, State, Stateful, VBitSet, VSet, Value}, ranker::RankingInfo, sudoku::{StdOverlay, StdVal}};

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

#[derive(Clone)]
pub struct NumberedRoom<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    number: u8,
    index: usize,
    direction: NumberedRoomDirection,
    possible_room_values: VBitSet<StdVal<MIN, MAX>>,
    room_indices: Vec<Index>,
    possible_positions: Vec<Index>,
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

fn possible_positions_raw<const MIN: u8, const MAX: u8, const N: usize, const M: usize>(
    number: u8,
    index: usize,
    direction: NumberedRoomDirection,
) -> Vec<Index> {
    let full = match direction {
        NumberedRoomDirection::RR | NumberedRoomDirection::RL | NumberedRoomDirection::RB => {
            (0..M).map(|i| [index, i]).collect::<Vec<_>>()
        },
        NumberedRoomDirection::CD | NumberedRoomDirection::CU | NumberedRoomDirection::CB => {
            (0..N).map(|i| [i, index]).collect::<Vec<_>>()
        },
    };
    if number == 1 {
        return full;
    }
    let self_indices = match direction {
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
    };
    let room_indices = match direction {
        NumberedRoomDirection::RR => vec![[index, 0]],
        NumberedRoomDirection::RL => vec![[index, M - 1]],
        NumberedRoomDirection::RB => vec![[index, 0], [index, M - 1]],
        NumberedRoomDirection::CD => vec![[0, index]],
        NumberedRoomDirection::CU => vec![[N - 1, index]],
        NumberedRoomDirection::CB => vec![[0, index], [N - 1, index]],
    };
    full.into_iter().filter(|pos| {
        !room_indices.contains(pos) && !self_indices.contains(pos)
    }).collect()
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
            possible_positions: possible_positions_raw::<MIN, MAX, N, M>(number, index, direction),
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

    pub fn relevant_move(&self, value: &StdVal<MIN, MAX>, index: Index) -> bool {
        if !self.contains(index) {
            return false;
        }
        value.val() == self.number || self.room_indices.contains(&index)
    }

    pub fn room_indices(&self) -> &Vec<Index> {
        &self.room_indices
    }

    pub fn possible_room_values(&self) -> &VBitSet<StdVal<MIN, MAX>> {
        &self.possible_room_values
    }

    pub fn possible_positions(&self) -> &Vec<Index> {
        &self.possible_positions
    }

    pub fn iter_non_room_indices(&self) -> NumberedRoomIter<'_, MIN, MAX, N, M> {
        NumberedRoomIter::new(self)
    }
}

pub const NUMBERED_ROOM_HEAD_FEATURE: &str = "NUMBERED_ROOM_HEAD";
pub const NUMBERED_ROOM_LN_POSSIBILITIES_FEATURE: &str = "NUMBERED_ROOM_LN_POSSIBILITIES";
pub const NUMBERED_ROOM_TAIL_FEATURE: &str = "NUMBERED_ROOM_TAIL";
pub const NUMBERED_ROOM_CONFLICT_ATTRIBUTION: &str = "NUMBERED_ROOM_CONFLICT_BAD";
pub const NUMBERED_ROOM_INFEASIBLE_ATTRIBUTION: &str = "NUMBERED_ROOM_INFEASIBLE";

pub struct NumberedRoomChecker<const MIN: u8, const MAX: u8, const N: usize, const M: usize> {
    rooms: Vec<NumberedRoom<MIN, MAX, N, M>>,
    /*room_head_feature: Key<Feature>,
    room_ln_possibilities_feature: Key<Feature>,
    room_tail_feature: Key<Feature>,
    room_conflict_attr: Key<Attribution>,
    room_if_attr: Key<Attribution>,*/
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize> NumberedRoomChecker<MIN, MAX, N, M> {
    pub fn new(rooms: Vec<NumberedRoom<MIN, MAX, N, M>>) -> Self {
        NumberedRoomChecker {
            rooms,
            /*room_head_feature: Key::register(NUMBERED_ROOM_HEAD_FEATURE),
            room_ln_possibilities_feature: Key::register(NUMBERED_ROOM_LN_POSSIBILITIES_FEATURE),
            room_tail_feature: Key::register(NUMBERED_ROOM_TAIL_FEATURE),
            room_conflict_attr: Key::register(NUMBERED_ROOM_CONFLICT_ATTRIBUTION),
            room_if_attr: Key::register(NUMBERED_ROOM_INFEASIBLE_ATTRIBUTION),*/
        }
    }
}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Debug for NumberedRoomChecker<MIN, MAX, N, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in &self.rooms {
            write!(f, "  {:?}\n", r)?;
            write!(f, "  - {}\n", r.possible_room_values().to_string())?;
        }
        Ok(())
    }
}

// Trivial implementation of Statefulness
impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Stateful<StdVal<MIN, MAX>> for NumberedRoomChecker<MIN, MAX, N, M> {}

impl <const MIN: u8, const MAX: u8, const N: usize, const M: usize>
Constraint<StdVal<MIN, MAX>, StdOverlay<N, M>> for NumberedRoomChecker<MIN, MAX, N, M> {
    fn name(&self) -> Option<String> { Some("NumberedRoomChecker".to_string()) }

    fn check(&self, _puzzle: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, _ranking: &mut RankingInfo<StdVal<MIN, MAX>>) -> ConstraintResult<StdVal<MIN, MAX>> {
        todo!()
    }

    fn debug_at(&self, _: &State<StdVal<MIN, MAX>, StdOverlay<N, M>>, index: Index) -> Option<String> {
        let header = "NumberedRoomChecker:\n";
        let mut lines = vec![];
        for r in &self.rooms {
            if !r.contains(index) {
                continue;
            }
            lines.push(format!("  {:?}", r));
            lines.push(format!("  {}", r.possible_room_values().to_string()));
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

    fn assert_missing_possible_positions<const MIN: u8, const MAX: u8, const N: usize, const M: usize>(
        number: u8,
        direction: NumberedRoomDirection,
        expected_missing: Vec<Index>,
    ) {
        let room = NumberedRoom::<MIN, MAX, N, M>::new(number, 0, direction);
        let actual = room.possible_positions();
        let full = match direction {
            NumberedRoomDirection::RR | NumberedRoomDirection::RL | NumberedRoomDirection::RB => {
                (0..M).map(|i| [0, i]).collect::<Vec<_>>()
            },
            NumberedRoomDirection::CD | NumberedRoomDirection::CU | NumberedRoomDirection::CB => {
                (0..N).map(|i| [i, 0]).collect::<Vec<_>>()
            },
        };
        for pos in actual {
            assert!(full.contains(pos), "Position {:?} is not in the full set of positions for {:?} direction", pos, direction);
        }
        let actual_missing: Vec<Index> = full.into_iter()
            .filter(|pos| !actual.contains(pos))
            .collect();
        assert_eq!(
            actual_missing, expected_missing,
            "Missing positions for {:?} direction with number {} do not match expected: {:?} != {:?}",
            direction, number, actual_missing, expected_missing
        )
    }

    #[test]
    fn test_rr_possible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::RR, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::RR,
            vec![[0, 0], [0, 1]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::RR,
            vec![[0, 0], [0, 3]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::RR,
            vec![[0, 0], [0, 5]],
        );
    }

    #[test]
    fn test_rl_possible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::RL, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::RL,
            vec![[0, 4], [0, 5]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::RL,
            vec![[0, 2], [0, 5]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::RL,
            vec![[0, 0], [0, 5]],
        );
    }

    #[test]
    fn test_rb_possible_positions() {
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::RB, vec![],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::RB,
            vec![[0, 0], [0, 1], [0, 4], [0, 5]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::RB,
            vec![[0, 0], [0, 2], [0, 3], [0, 5]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::RB,
            vec![[0, 0], [0, 5]],
        );
    }

    #[test]
    fn test_cd_possible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::CD, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::CD,
            vec![[0, 0], [1, 0]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::CD,
            vec![[0, 0], [3, 0]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::CD,
            vec![[0, 0], [5, 0]],
        );
    }

    #[test]
    fn test_cu_possible_positions() {
        // 1 can be in the position index (unlike other numbers), as well as
        // all the other ones.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::CU, vec![],
        );
        // 2 through 6 can neither be in the position index nor their own position.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::CU,
            vec![[4, 0], [5, 0]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::CU,
            vec![[2, 0], [5, 0]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::CU,
            vec![[0, 0], [5, 0]],
        );
    }

    #[test]
    fn test_cb_possible_positions() {
        // 1 can be in either position index (unlike other numbers), as well as
        // all the other ones.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            1, NumberedRoomDirection::CB, vec![],
        );
        // 2 through 6 can neither be in neither of the position indices nor
        // their own position from either end.
        assert_missing_possible_positions::<1, 6, 6, 6>(
            2, NumberedRoomDirection::CB,
            vec![[0, 0], [1, 0], [4, 0], [5, 0]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            4, NumberedRoomDirection::CB,
            vec![[0, 0], [2, 0], [3, 0], [5, 0]],
        );
        assert_missing_possible_positions::<1, 6, 6, 6>(
            6, NumberedRoomDirection::CB,
            vec![[0, 0], [5, 0]],
        );
    }
}