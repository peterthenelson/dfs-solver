use crate::{core::{empty_set, unpack_first, unpack_last, UVSet, Value}, sudoku::StdVal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Range<const MIN: u8, const MAX: u8> {
    Empty,
    HalfOpen(u8, u8),
}

impl <const MIN: u8, const MAX: u8> Range<MIN, MAX> {
    pub fn from_set(set: &UVSet<u8>) -> Range<MIN, MAX> {
        if let Some(min) = unpack_first::<StdVal<MIN, MAX>>(set) {
            let max = unpack_last::<StdVal<MIN, MAX>>(set).unwrap();
            Range::HalfOpen(min.val(), max.val()+1)
        } else {
            Range::Empty
        }
    }

    pub fn intersection(&self, other: &Range<MIN, MAX>) -> Range<MIN, MAX> {
        match self {
            Range::Empty => Range::Empty,
            Range::HalfOpen(min1, max1) => match other {
                Range::Empty => Range::Empty,
                Range::HalfOpen(min2, max2) => {
                    let min3 = min1.max(min2);
                    let max3 = max1.min(max2);
                    if min3 < max3 {
                        Range::HalfOpen(*min3, *max3)
                    } else {
                        Range::Empty
                    }
                },
            },
        }
    }

    pub fn clip_min(&mut self, min: u8) {
        match self {
            Range::Empty => {},
            Range::HalfOpen(smin, smax) => {
                *smin = min.max(*smin);
                if *smin >= *smax {
                    *self = Range::Empty;
                }
            },
        }
    }

    pub fn clip_max(&mut self, max: u8) {
        match self {
            Range::Empty => {},
            Range::HalfOpen(smin, smax) => {
                *smax = max.min(*smax);
                if *smin >= *smax {
                    *self = Range::Empty;
                }
            },
        }
    }

    pub fn is_empty(&self) -> bool { if let Range::Empty = self { true } else { false } }

    pub fn to_set(&self) -> UVSet<u8> {
        let mut s = empty_set::<StdVal<MIN, MAX>>();
        match self {
            Range::Empty => {},
            Range::HalfOpen(min, max) => {
                for v in *min..*max {
                    s.insert(StdVal::<MIN, MAX>::new(v).to_uval());
                }
            }
        }
        s
    }
}

#[cfg(test)]
mod test {
    use crate::{core::pack_values, sudoku::{unpack_stdval_vals, NineStdVal}};
    use super::*;

    type NineRange = Range<1, 9>;

    #[test]
    fn test_range() {
        assert!(NineRange::Empty.is_empty());
        assert!(!NineRange::HalfOpen(1, 5).is_empty());
        assert_eq!(unpack_stdval_vals::<1, 9>(&NineRange::Empty.to_set()), Vec::<u8>::new());
        assert_eq!(
            unpack_stdval_vals::<1, 9>(&NineRange::HalfOpen(2, 5).to_set()),
            vec![2, 3, 4],
        );
        assert_eq!(
            NineRange::from_set(&empty_set::<NineStdVal>()),
            NineRange::Empty,
        );
        assert_eq!(
            NineRange::from_set(&pack_values::<NineStdVal>(&vec![
                NineStdVal::new(4), NineStdVal::new(5), NineStdVal::new(2),
            ])),
            NineRange::HalfOpen(2, 6),
        );
    }

    #[test]
    fn test_range_intersection() {
        assert_eq!(
            NineRange::Empty.intersection(&NineRange::HalfOpen(1, 5)),
            NineRange::Empty,
        );
        assert_eq!(
            NineRange::HalfOpen(1, 5).intersection(&NineRange::Empty),
            NineRange::Empty,
        );
        assert_eq!(
            NineRange::HalfOpen(1, 5).intersection(&NineRange::HalfOpen(5, 9)),
            NineRange::Empty,
        );
        assert_eq!(
            NineRange::HalfOpen(1, 5).intersection(&NineRange::HalfOpen(3, 7)),
            NineRange::HalfOpen(3, 5),
        );
    }

    #[test]
    fn test_range_clip() {
        {
            let mut r = NineRange::Empty;
            r.clip_max(9);
            assert!(r.is_empty());
        }
        {
            let mut r = NineRange::Empty;
            r.clip_min(9);
            assert!(r.is_empty());
        }
        {
            let mut r = NineRange::HalfOpen(2, 5);
            r.clip_min(3);
            assert_eq!(r, NineRange::HalfOpen(3, 5));
        }
        {
            let mut r = NineRange::HalfOpen(2, 5);
            r.clip_max(3);
            assert_eq!(r, NineRange::HalfOpen(2, 3));
        }
        {
            let mut r = NineRange::HalfOpen(2, 5);
            r.clip_min(5);
            assert!(r.is_empty());
        }
        {
            let mut r = NineRange::HalfOpen(2, 5);
            r.clip_max(2);
            assert!(r.is_empty());
        }
    }
}