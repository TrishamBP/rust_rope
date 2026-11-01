//! Position indexing.
//!
//! ## Implemented (Stage 3)
//!
//! - [`Position`]: one token position `m` (0, 1, 2, ...). A validated
//!   newtype around `usize`.
//! - [`PositionRange`]: a sequence of consecutive positions
//!   `start, start + 1, ..., start + length − 1`, with a configurable start
//!   and a configurable length. It produces the positions lazily (without a
//!   heap allocation).
//!
//! The conversion of positions into angles `m · θⱼ` is in
//! [`crate::rope::frequencies`], because it needs the frequencies.
//!
//! ## Token position versus dimension-pair index
//!
//! RoPE uses two different integer indices. Do not confuse them.
//!
//! | | Token position `m` | Dimension-pair index `j` |
//! |---|---|---|
//! | Counts | tokens in the sequence | pairs inside one vector |
//! | Range | `0, 1, 2, ...` (no fixed limit) | `0, 1, ..., d/2 − 1` |
//! | Changes the | **angle** (`m` multiplies `θⱼ`) | **frequency** `θⱼ` |
//! | Rust type | [`Position`] | `usize` |
//!
//! Example: in the sentence "the cat sat", the token "sat" has position
//! `m = 2`. Its query vector with `d = 8` has 4 pairs, `j = 0, 1, 2, 3`.
//! Each of the 4 pairs rotates by its own angle `2 · θⱼ`.
//!
//! Both indices are integers. If both were plain `usize` values, a call such
//! as `angle(j, m)` instead of `angle(m, j)` would compile and give a wrong
//! result. The newtype [`Position`] makes this mistake a compile error.

use crate::utils::validation::{MAX_POSITION, RopeError, validate_position};

/// One token position `m` in a sequence.
///
/// `Position` is a **newtype**: a struct with exactly one field. It has the
/// same size and layout as `usize` (8 bytes on a 64-bit computer). It has no
/// run-time cost. It gives two guarantees:
///
/// 1. The compiler does not accept a plain `usize` (for example a pair
///    index) where a `Position` is necessary.
/// 2. The value is at most [`MAX_POSITION`] (`2^53`). Thus the conversion to
///    `f64` in [`Position::as_f64`] is exact.
///
/// # Example
///
/// ```
/// use rust_rope::rope::position::Position;
///
/// let m = Position::new(3)?;
/// assert_eq!(m.value(), 3);
/// assert_eq!(m.as_f64(), 3.0);
/// assert_eq!(Position::ZERO.value(), 0);
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Position(usize);

impl Position {
    /// The first position in a sequence, `m = 0`.
    pub const ZERO: Position = Position(0);

    /// Creates a position `m`.
    ///
    /// # Errors
    ///
    /// [`RopeError::PositionTooLarge`] if `m > 2^53`.
    pub fn new(value: usize) -> Result<Self, RopeError> {
        validate_position(value)?;
        Ok(Position(value))
    }

    /// Returns the position `m` as an integer.
    pub const fn value(self) -> usize {
        self.0
    }

    /// Returns the position `m` as an `f64`, for the angle `m · θⱼ`.
    ///
    /// The conversion is exact, because `m ≤ 2^53`.
    pub const fn as_f64(self) -> f64 {
        self.0 as f64
    }
}

/// A sequence of consecutive token positions:
/// `start, start + 1, ..., start + length − 1`.
///
/// - `start` is configurable. A model that generates text one token at a
///   time continues at the position after the previous tokens, not at 0.
/// - `length` is configurable. It is the number of tokens `L`.
///
/// A `PositionRange` stores only `start` and `length` (16 bytes). It does
/// not store the individual positions. [`PositionRange::iter`] calculates
/// them one at a time. Thus a range of one million positions also uses only
/// 16 bytes.
///
/// # Example
///
/// ```
/// use rust_rope::rope::position::PositionRange;
///
/// let range = PositionRange::new(5, 3)?;
/// let values: Vec<usize> = range.iter().map(|m| m.value()).collect();
/// assert_eq!(values, vec![5, 6, 7]);
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionRange {
    start: usize,
    length: usize,
}

impl PositionRange {
    /// Creates the range `start, start + 1, ..., start + length − 1`.
    ///
    /// `length = 0` gives an empty range. This is valid: it is a sequence
    /// with no tokens.
    ///
    /// # Errors
    ///
    /// - [`RopeError::PositionTooLarge`] if `start > 2^53`.
    /// - [`RopeError::SequenceTooLong`] if the last position
    ///   `start + length − 1` is larger than `2^53`, or if the addition
    ///   overflows a `usize`.
    pub fn new(start: usize, length: usize) -> Result<Self, RopeError> {
        validate_position(start)?;
        if length > 0 {
            // `checked_add` returns `None` if the sum does not fit in a
            // `usize`. A plain `+` would panic in a debug build and wrap
            // around in a release build.
            let last = start
                .checked_add(length - 1)
                .ok_or(RopeError::SequenceTooLong { start, length })?;
            if last as u64 > MAX_POSITION {
                return Err(RopeError::SequenceTooLong { start, length });
            }
        }
        Ok(PositionRange { start, length })
    }

    /// Creates the range `0, 1, ..., length − 1`. This is the usual case: a
    /// sequence that starts at the first token.
    ///
    /// # Errors
    ///
    /// [`RopeError::SequenceTooLong`] if `length − 1 > 2^53`.
    pub fn from_zero(length: usize) -> Result<Self, RopeError> {
        PositionRange::new(0, length)
    }

    /// Returns the first position of the range.
    ///
    /// For an empty range, this is the position where the range would start.
    pub fn start(self) -> Position {
        // `new` validated `start`.
        Position(self.start)
    }

    /// Returns the number of positions in the range, `L`.
    pub fn len(self) -> usize {
        self.length
    }

    /// Returns `true` if the range has no positions.
    pub fn is_empty(self) -> bool {
        self.length == 0
    }

    /// Returns the row index of `position` inside the range:
    /// `position − start`. Returns `None` if the range does not contain
    /// `position`.
    ///
    /// Example: the range `5, 6, 7` contains position 6 at index 1.
    pub fn index_of(self, position: Position) -> Option<usize> {
        // `checked_sub` returns `None` if `position < start`.
        let index = position.0.checked_sub(self.start)?;
        if index < self.length {
            Some(index)
        } else {
            None
        }
    }

    /// Returns an iterator over the positions of the range, in increasing
    /// order.
    ///
    /// The return type `impl Iterator<Item = Position>` means: "a type that
    /// implements `Iterator` and produces `Position` values". The caller does
    /// not need to know the exact type. The compiler knows it and generates
    /// direct code for it, without dynamic dispatch.
    ///
    /// The iterator is **lazy**: it calculates each position only when the
    /// caller asks for the next value. It does not allocate memory.
    pub fn iter(self) -> impl Iterator<Item = Position> {
        // `start..end` is a `Range<usize>`. It is an iterator over integers.
        // `map(Position)` uses the tuple-struct constructor as a function:
        // `Position(5)` is the same as calling a function `fn(usize) -> Position`.
        // `new` validated that `start + length − 1 ≤ 2^53`, so `start + length`
        // cannot overflow and every value is a valid position.
        (self.start..self.start + self.length).map(Position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_zero_and_one() {
        assert_eq!(Position::ZERO, Position::new(0).unwrap());
        assert_eq!(Position::ZERO.as_f64(), 0.0);
        assert_eq!(Position::new(1).unwrap().as_f64(), 1.0);
    }

    #[test]
    fn position_conversion_is_exact_up_to_the_limit() {
        #[cfg(target_pointer_width = "64")]
        {
            let max = Position::new(MAX_POSITION as usize).unwrap();
            assert_eq!(max.as_f64(), 9_007_199_254_740_992.0);
            assert!(Position::new(MAX_POSITION as usize + 1).is_err());
        }
    }

    #[test]
    fn positions_are_ordered() {
        let a = Position::new(2).unwrap();
        let b = Position::new(10).unwrap();
        assert!(a < b);
    }

    #[test]
    fn range_from_zero() {
        let range = PositionRange::from_zero(5).unwrap();
        let values: Vec<usize> = range.iter().map(Position::value).collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4]);
        assert_eq!(range.len(), 5);
        assert_eq!(range.start(), Position::ZERO);
    }

    #[test]
    fn range_with_start_offset() {
        let range = PositionRange::new(100, 3).unwrap();
        let values: Vec<usize> = range.iter().map(Position::value).collect();
        assert_eq!(values, vec![100, 101, 102]);
    }

    #[test]
    fn empty_range() {
        let range = PositionRange::new(7, 0).unwrap();
        assert!(range.is_empty());
        assert_eq!(range.iter().count(), 0);
        assert_eq!(range.index_of(Position::new(7).unwrap()), None);
    }

    #[test]
    fn index_of_finds_rows() {
        let range = PositionRange::new(5, 3).unwrap();
        let p = |v| Position::new(v).unwrap();
        assert_eq!(range.index_of(p(4)), None);
        assert_eq!(range.index_of(p(5)), Some(0));
        assert_eq!(range.index_of(p(7)), Some(2));
        assert_eq!(range.index_of(p(8)), None);
    }

    #[test]
    fn range_limits() {
        // `start + length − 1` overflows a `usize`.
        assert_eq!(
            PositionRange::new(10, usize::MAX),
            Err(RopeError::SequenceTooLong {
                start: 10,
                length: usize::MAX
            })
        );
        #[cfg(target_pointer_width = "64")]
        {
            let max = MAX_POSITION as usize;
            // The last position is exactly 2^53: valid.
            assert!(PositionRange::new(max - 1, 2).is_ok());
            // The last position is 2^53 + 1: invalid.
            assert_eq!(
                PositionRange::new(max - 1, 3),
                Err(RopeError::SequenceTooLong {
                    start: max - 1,
                    length: 3
                })
            );
            // The start itself is too large.
            assert_eq!(
                PositionRange::new(max + 1, 0),
                Err(RopeError::PositionTooLarge { position: max + 1 })
            );
        }
    }
}
