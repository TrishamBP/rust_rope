//! Positional rotation frequencies and positional angles.
//!
//! ## Implemented (Stage 3)
//!
//! - [`inverse_frequency`]: one inverse frequency `θⱼ = b^(−2j/d)`.
//! - [`inverse_frequencies`]: all `d/2` inverse frequencies
//!   `[θ₀, θ₁, ..., θ_(d/2−1)]`.
//! - [`angle`]: one positional angle `φ(m, j) = m · θⱼ`.
//! - [`AngleTable`]: the angles for many positions and all pairs.
//!
//! ## Not implemented in this stage
//!
//! The sine, the cosine and the rotation of embedding vectors are planned
//! for Stage 4. This module only calculates the angles.
//!
//! **Status update (Stage 4):** the rotation is implemented in
//! [`crate::rope::rotations`] and [`crate::rope::embeddings`]. This module
//! still only calculates frequencies and angles.
//!
//! ## Notation
//!
//! | Symbol | Meaning | Rust name |
//! |---|---|---|
//! | `d` | embedding dimension (even) | `config.dimension()` |
//! | `b` | frequency base (`b > 1`) | `config.base()` |
//! | `j` | pair index, `0 ≤ j < d/2` | `pair_index` |
//! | `θⱼ` | inverse frequency of pair `j` (radians per position) | `inverse_frequency` |
//! | `m` | token position | `position` |
//! | `φ(m, j)` | angle of pair `j` at position `m` (radians) | return value of `angle` |
//!
//! Some texts write the pair index as `i`. We use `j`, because `i` is the
//! imaginary unit in `src/math/complex.rs`.
//!
//! ## The equations
//!
//! ```text
//! θⱼ      = 1 / b^(2j/d) = b^(−2j/d)
//! φ(m, j) = m · θⱼ
//! ```
//!
//! Example with `d = 8`, `b = 10000`: `θ = [1, 0.1, 0.01, 0.001]`.
//! At position `m = 3`: `φ = [3, 0.3, 0.03, 0.003]`.

use crate::rope::config::RopeConfig;
use crate::rope::position::{Position, PositionRange};
use crate::utils::validation::{
    RopeError, validate_inverse_frequencies, validate_inverse_frequency,
};

// ---------------------------------------------------------------------------
// Inverse frequencies
// ---------------------------------------------------------------------------

/// Calculates `θⱼ = b^(−2j/d)` without checks.
///
/// This private function is the only place in the library with the
/// frequency formula. The public functions check their input and then call
/// it.
///
/// - `2j / d` is the exponent. It is in the range `[0, 1)`.
/// - `powf(−x)` calculates `b^(−x)` with one rounding. The form
///   `1.0 / b.powf(x)` has two roundings (the power and the division).
#[inline]
pub(crate) fn inverse_frequency_unchecked(pair_index: usize, dimension: usize, base: f64) -> f64 {
    let exponent = (2 * pair_index) as f64 / dimension as f64;
    base.powf(-exponent)
}

/// Returns the inverse frequency `θⱼ = b^(−2j/d)` of the pair `j`.
///
/// `θⱼ` is the rotation speed of pair `j`, in radians per position.
///
/// - `j = 0` gives `θ₀ = b⁰ = 1` (one radian per position, the fastest pair).
/// - A larger `j` gives a smaller `θⱼ` (a slower pair), because `b > 1`.
///
/// # Errors
///
/// [`RopeError::PairIndexOutOfRange`] if `j ≥ d/2`.
///
/// # Example
///
/// ```
/// use rust_rope::rope::config::RopeConfig;
/// use rust_rope::rope::frequencies::inverse_frequency;
/// use rust_rope::math::{approx_eq, DEFAULT_TOLERANCE};
///
/// let config = RopeConfig::new(8, 10_000.0)?;
/// // θ₂ = 10000^(−4/8) = 10000^(−0.5) = 1/100
/// let theta_2 = inverse_frequency(&config, 2)?;
/// assert!(approx_eq(theta_2, 0.01, DEFAULT_TOLERANCE));
/// assert!(inverse_frequency(&config, 4).is_err()); // only j = 0..3 exist
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
pub fn inverse_frequency(config: &RopeConfig, pair_index: usize) -> Result<f64, RopeError> {
    let num_pairs = config.num_pairs();
    if pair_index >= num_pairs {
        return Err(RopeError::PairIndexOutOfRange {
            pair_index,
            num_pairs,
        });
    }
    Ok(inverse_frequency_unchecked(
        pair_index,
        config.dimension(),
        config.base(),
    ))
}

/// Returns all `d/2` inverse frequencies `[θ₀, θ₁, ..., θ_(d/2−1)]`.
///
/// The values form a **geometric sequence**: each value is the previous
/// value multiplied by the constant ratio `b^(−2/d)`.
///
/// This function cannot fail. A [`RopeConfig`] is always valid, and the
/// function creates only valid pair indices.
///
/// # Memory
///
/// The function allocates one `Vec<f64>` with exactly `d/2` elements on the
/// heap (`d/2 · 8` bytes) and moves it to the caller.
///
/// # Example
///
/// ```
/// use rust_rope::rope::config::RopeConfig;
/// use rust_rope::rope::frequencies::inverse_frequencies;
///
/// let theta = inverse_frequencies(&RopeConfig::new(8, 10_000.0)?);
/// assert_eq!(theta.len(), 4);
/// assert_eq!(theta[0], 1.0);
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
pub fn inverse_frequencies(config: &RopeConfig) -> Vec<f64> {
    let dimension = config.dimension();
    let base = config.base();
    // `0..n` is an iterator over the pair indices. `map` applies the formula
    // to each index. `collect` builds the `Vec`. The range knows its exact
    // length, so `collect` allocates the correct capacity one time.
    (0..config.num_pairs())
        .map(|j| inverse_frequency_unchecked(j, dimension, base))
        .collect()
}

// ---------------------------------------------------------------------------
// Positional angles
// ---------------------------------------------------------------------------

/// Calculates `φ = m · θ` without checks.
///
/// `pub(crate)`: visible in the whole `rust_rope` crate, but not to users of
/// the library. `rope::embeddings` uses it (Stage 4), because its
/// frequencies come from a validated `RopeConfig` and need no new check.
#[inline]
pub(crate) fn angle_unchecked(position: Position, inverse_frequency: f64) -> f64 {
    position.as_f64() * inverse_frequency
}

/// Returns the rotation angle `φ(m, j) = m · θⱼ`, in radians.
///
/// - `position` is the token position `m`.
/// - `inverse_frequency` is `θⱼ`, for example a value from
///   [`inverse_frequencies`].
///
/// Position 0 always gives the angle 0: the first token is not rotated.
///
/// # Errors
///
/// [`RopeError::InvalidInverseFrequency`] if `θⱼ` is not a finite number
/// greater than 0.
///
/// # Example
///
/// ```
/// use rust_rope::rope::frequencies::angle;
/// use rust_rope::rope::position::Position;
///
/// let m = Position::new(3)?;
/// assert_eq!(angle(m, 0.5)?, 1.5);
/// assert_eq!(angle(Position::ZERO, 0.5)?, 0.0);
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
pub fn angle(position: Position, inverse_frequency: f64) -> Result<f64, RopeError> {
    validate_inverse_frequency(inverse_frequency)?;
    Ok(angle_unchecked(position, inverse_frequency))
}

/// The positional angles `φ(m, j) = m · θⱼ` for a range of positions and all
/// pairs.
///
/// The table has one **row** for each position `m` and one **column** for
/// each pair `j`. With `d = 8`, `b = 10000` and positions `0..5`:
///
/// ```text
///          j = 0    j = 1    j = 2    j = 3
/// m = 0    0        0        0        0
/// m = 1    1        0.1      0.01     0.001
/// m = 2    2        0.2      0.02     0.002
/// m = 3    3        0.3      0.03     0.003
/// m = 4    4        0.4      0.04     0.004
/// ```
///
/// # Memory layout
///
/// The table stores all angles in **one** `Vec<f64>`, row after row
/// ("row-major" order). The angle of row `r` and pair `j` is at the index
/// `r · P + j`, where `P` is the number of pairs. One contiguous block of
/// memory is faster to read than a `Vec<Vec<f64>>`, which has one separate
/// heap allocation for each row.
///
/// # Example
///
/// ```
/// use rust_rope::rope::config::RopeConfig;
/// use rust_rope::rope::frequencies::{inverse_frequencies, AngleTable};
/// use rust_rope::rope::position::{Position, PositionRange};
///
/// let theta = inverse_frequencies(&RopeConfig::new(8, 10_000.0)?);
/// let table = AngleTable::new(PositionRange::from_zero(5)?, &theta)?;
///
/// assert_eq!(table.num_positions(), 5);
/// assert_eq!(table.num_pairs(), 4);
/// assert_eq!(table.get(Position::new(2)?, 0), Some(2.0));
/// assert_eq!(table.row(Position::ZERO), Some(&[0.0, 0.0, 0.0, 0.0][..]));
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AngleTable {
    positions: PositionRange,
    num_pairs: usize,
    /// Row-major: `angles[r * num_pairs + j] = (start + r) · θⱼ`.
    angles: Vec<f64>,
}

impl AngleTable {
    /// Calculates the angle `m · θⱼ` for each position `m` in `positions`
    /// and each inverse frequency `θⱼ` in `inverse_frequencies`.
    ///
    /// The function **borrows** `inverse_frequencies`: it reads the values,
    /// and the caller keeps the `Vec`. The function allocates one new
    /// `Vec<f64>` with exactly `L · P` elements (`L` positions, `P` pairs).
    ///
    /// # Errors
    ///
    /// - [`RopeError::EmptyFrequencies`] if `inverse_frequencies` is empty.
    /// - [`RopeError::InvalidInverseFrequency`] if a value is not a finite
    ///   number greater than 0.
    /// - [`RopeError::TableTooLarge`] if `L · P` does not fit in a `usize`.
    pub fn new(positions: PositionRange, inverse_frequencies: &[f64]) -> Result<Self, RopeError> {
        validate_inverse_frequencies(inverse_frequencies)?;
        let num_pairs = inverse_frequencies.len();
        let num_elements =
            positions
                .len()
                .checked_mul(num_pairs)
                .ok_or(RopeError::TableTooLarge {
                    rows: positions.len(),
                    columns: num_pairs,
                })?;

        // One allocation with the exact final size. `push` never needs to
        // grow the `Vec` (no reallocation, no copying).
        let mut angles = Vec::with_capacity(num_elements);
        for position in positions.iter() {
            // `for &theta in slice` iterates over references `&f64` and
            // copies each value out of the reference.
            for &theta in inverse_frequencies {
                angles.push(angle_unchecked(position, theta));
            }
        }
        // Internal consistency check (a bug if it fails, not a caller error).
        debug_assert_eq!(angles.len(), num_elements);

        Ok(AngleTable {
            positions,
            num_pairs,
            angles,
        })
    }

    /// Returns the range of positions (the rows of the table).
    pub fn positions(&self) -> PositionRange {
        self.positions
    }

    /// Returns the number of positions `L` (the number of rows).
    pub fn num_positions(&self) -> usize {
        self.positions.len()
    }

    /// Returns the number of pairs `P` (the number of columns).
    pub fn num_pairs(&self) -> usize {
        self.num_pairs
    }

    /// Returns the angles of one position: `[m · θ₀, m · θ₁, ...]`.
    ///
    /// Returns `None` if the table does not contain the position.
    ///
    /// The result is a **slice** `&[f64]`: a borrowed view into the table.
    /// It does not copy the data. The table cannot change or move while the
    /// slice exists; the borrow checker enforces this.
    pub fn row(&self, position: Position) -> Option<&[f64]> {
        let r = self.positions.index_of(position)?;
        let start = r * self.num_pairs;
        Some(&self.angles[start..start + self.num_pairs])
    }

    /// Returns the angle `φ(m, j) = m · θⱼ`.
    ///
    /// Returns `None` if the table does not contain the position, or if
    /// `pair_index ≥ P`.
    pub fn get(&self, position: Position, pair_index: usize) -> Option<f64> {
        // `slice.get(j)` returns `Option<&f64>`. `copied()` turns it into
        // `Option<f64>`.
        self.row(position)?.get(pair_index).copied()
    }

    /// Returns all angles as one row-major slice of length `L · P`.
    pub fn as_slice(&self) -> &[f64] {
        &self.angles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DEFAULT_TOLERANCE, approx_eq};

    fn close(a: f64, b: f64) -> bool {
        approx_eq(a, b, DEFAULT_TOLERANCE)
    }

    fn config(d: usize, b: f64) -> RopeConfig {
        RopeConfig::new(d, b).unwrap()
    }

    fn pos(m: usize) -> Position {
        Position::new(m).unwrap()
    }

    #[test]
    fn worked_example_dimension_8() {
        // θⱼ = 10000^(−2j/8) = 10000^(−j/4) = 10^(−j)
        let theta = inverse_frequencies(&config(8, 10_000.0));
        let expected = [1.0, 0.1, 0.01, 0.001];
        assert_eq!(theta.len(), expected.len());
        for (j, (&t, &e)) in theta.iter().zip(expected.iter()).enumerate() {
            assert!(close(t, e), "j = {j}: {t} vs {e}");
        }
    }

    #[test]
    fn first_frequency_is_exactly_one() {
        // b^(−0) = 1 exactly: IEEE 754 defines pow(x, ±0) = 1 for every x.
        for b in [2.0, 10_000.0, 500_000.0] {
            assert_eq!(inverse_frequency(&config(64, b), 0).unwrap(), 1.0);
        }
    }

    #[test]
    fn single_and_list_functions_agree() {
        let c = config(128, 10_000.0);
        let all = inverse_frequencies(&c);
        for (j, &t) in all.iter().enumerate() {
            assert_eq!(inverse_frequency(&c, j).unwrap(), t);
        }
    }

    #[test]
    fn pair_index_out_of_range() {
        let c = config(8, 10_000.0);
        assert!(inverse_frequency(&c, 3).is_ok());
        assert_eq!(
            inverse_frequency(&c, 4),
            Err(RopeError::PairIndexOutOfRange {
                pair_index: 4,
                num_pairs: 4
            })
        );
    }

    #[test]
    fn angle_is_position_times_frequency() {
        assert_eq!(angle(Position::ZERO, 0.37).unwrap(), 0.0);
        assert_eq!(angle(pos(1), 0.37).unwrap(), 0.37);
        assert!(close(angle(pos(4), 0.1).unwrap(), 0.4));
    }

    #[test]
    fn angle_rejects_invalid_frequency() {
        assert!(angle(pos(1), 0.0).is_err());
        assert!(angle(pos(1), -1.0).is_err());
        assert!(angle(pos(1), f64::NAN).is_err());
    }

    #[test]
    fn table_layout_is_row_major() {
        let theta = [1.0, 0.5];
        let table = AngleTable::new(PositionRange::new(2, 3).unwrap(), &theta).unwrap();
        // rows m = 2, 3, 4; columns θ = 1, 0.5
        assert_eq!(table.as_slice(), &[2.0, 1.0, 3.0, 1.5, 4.0, 2.0]);
        assert_eq!(table.row(pos(3)), Some(&[3.0, 1.5][..]));
        assert_eq!(table.get(pos(4), 1), Some(2.0));
    }

    #[test]
    fn table_lookup_outside_the_table() {
        let table = AngleTable::new(PositionRange::new(2, 3).unwrap(), &[1.0, 0.5]).unwrap();
        assert_eq!(table.row(pos(1)), None); // before the start
        assert_eq!(table.row(pos(5)), None); // after the end
        assert_eq!(table.get(pos(2), 2), None); // pair index too large
    }

    #[test]
    fn table_rejects_invalid_frequencies() {
        let range = PositionRange::from_zero(3).unwrap();
        assert_eq!(
            AngleTable::new(range, &[]),
            Err(RopeError::EmptyFrequencies)
        );
        assert_eq!(
            AngleTable::new(range, &[1.0, 0.0]),
            Err(RopeError::InvalidInverseFrequency {
                pair_index: Some(1),
                value: 0.0
            })
        );
    }

    #[test]
    fn empty_position_range_gives_an_empty_table() {
        let table = AngleTable::new(PositionRange::from_zero(0).unwrap(), &[1.0]).unwrap();
        assert_eq!(table.num_positions(), 0);
        assert!(table.as_slice().is_empty());
    }

    #[test]
    fn table_too_large_is_an_error() {
        // 2^53 positions × 2^11 pairs = 2^64 elements: one more than the
        // largest 64-bit `usize`. `checked_mul` detects this before any
        // allocation. The frequency slice itself is small (2048 · 8 B = 16 KiB).
        #[cfg(target_pointer_width = "64")]
        {
            let range = PositionRange::from_zero(1 << 53).unwrap();
            let theta = vec![1.0; 1 << 11];
            assert_eq!(
                AngleTable::new(range, &theta),
                Err(RopeError::TableTooLarge {
                    rows: 1 << 53,
                    columns: 1 << 11
                })
            );
        }
    }
}
