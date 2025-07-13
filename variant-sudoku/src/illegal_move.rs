use crate::core::{Attribution, ConstraintResult, Error, Index, Key, Value};

pub const ILLEGAL_ACTION: Error = Error::new_const("A violation already exists; can't apply further actions.");
pub const UNDO_MISMATCH: Error = Error::new_const("Undo value mismatch");

pub struct IllegalMove<V: Value>{
    action: Option<(Index, V, Key<Attribution>)>,
}

impl <V: Value> IllegalMove<V> {
    pub fn new() -> Self { Self { action: None } }

    pub fn set(&mut self, index: Index, value: V, attr: Key<Attribution>) {
        self.action = Some((index, value, attr));
    }

    pub fn check_unset(&self) -> Result<(), Error> {
        if self.action.is_some() {
            Err(ILLEGAL_ACTION)
        } else {
            Ok(())
        }
    }

    pub fn reset(&mut self) {
        self.action = None;
    }

    pub fn undo(&mut self, index: Index, value: V) -> Result<bool, Error> {
        if let Some((i, v, _)) = self.action {
            if i == index && v == value {
                self.action = None;
                Ok(true)
            } else {
                Err(UNDO_MISMATCH)
            }
        } else {
            Ok(false)
        }
    }

    pub fn to_contradiction(&self) -> Option<ConstraintResult<V>> {
        self.action.map(|(_, _, attr)| {
            ConstraintResult::Contradiction(attr)
        })
    }

    pub fn write_dbg(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some((i, v, a)) = &self.action {
            write!(f, "Illegal move: {:?}={:?} ({})\n", i, v, a.name())?;
        }
        Ok(())
    }

    pub fn debug_at(&self, index: Index) -> Option<String> {
        if let Some((i, v, a)) = &self.action {
            if *i == index {
                return Some(format!("Illegal move: {:?}={:?} ({})", i, v, a.name()));
            }
        }
        None
    }

    pub fn debug_highlight(&self, index: Index) -> Option<(u8, u8, u8)> {
        if let Some((i, _, _)) = &self.action {
            if *i == index {
                return Some((200, 0, 0));
            }
        }
        None

    }
}