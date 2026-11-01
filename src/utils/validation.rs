//! Input and configuration validation.
//!
//! ## Implemented (Stage 3)
//!
//! - [`RopeError`]: the one error type of the RoPE part of the library.
//!   Each variant describes one kind of invalid input.
//! - [`validate_dimension`]: the embedding dimension `d` must be even and
//!   not zero, because RoPE rotates the dimensions in pairs.
//! - [`validate_base`]: the frequency base `b` must be a finite number
//!   greater than 1.
//! - [`validate_position`]: a token position `m` must be exactly
//!   representable as an `f64` (`m ≤ 2^53`).
//! - [`validate_inverse_frequency`] and [`validate_inverse_frequencies`]:
//!   an inverse frequency `θⱼ` must be a finite number greater than 0.
//!
//! ## Implemented (Stage 4)
//!
//! - [`validate_embedding_dimension`]: an embedding must have exactly `d`
//!   components.
//!
//! ## Why validation is a separate module
//!
//! The checks are small, but several modules use them. If each module had
//! its own copy, the copies could become different. One module gives one
//! rule and one error message for each kind of invalid input.
//!
//! Each function returns `Result<(), RopeError>`. `Ok(())` means "the input
//! is valid". The `()` type ("unit") is the empty value: there is no data
//! to return when the input is valid.

use std::fmt;

/// The largest token position that we accept: `2^53 = 9 007 199 254 740 992`.
///
/// An `f64` value has 53 significant bits. Thus every integer from 0 to
/// `2^53` has an exact `f64` representation. Above `2^53`, some integers
/// have no exact representation. Example: `2^53 + 1` converts to `2^53`.
/// Then two different positions would get the same rotation angle.
///
/// The type is `u64`, not `usize`. The value `2^53` does not fit in a 32-bit
/// `usize`. A `u64` constant compiles on all targets.
pub const MAX_POSITION: u64 = 1 << 53;

/// The errors that the RoPE functions return for invalid input.
///
/// A caller can cause each of these errors with bad input data. Thus the
/// functions return `Err(RopeError)`. They do not panic.
///
/// `RopeError` is an `enum`. An enum value is exactly one of its variants.
/// Some variants carry data (the invalid value), so that the error message
/// can show it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RopeError {
    /// The embedding dimension `d` is 0. RoPE needs at least one pair.
    ZeroDimension,
    /// The embedding dimension `d` is odd. RoPE rotates dimensions in
    /// pairs, so one dimension would have no partner.
    OddDimension {
        /// The invalid dimension.
        dimension: usize,
    },
    /// The frequency base `b` is not a finite number greater than 1.
    InvalidBase {
        /// The invalid base.
        base: f64,
    },
    /// A pair index `j` is not in the range `0 ≤ j < d/2`.
    PairIndexOutOfRange {
        /// The invalid pair index.
        pair_index: usize,
        /// The number of pairs, `d/2`. Valid indices are `0..num_pairs`.
        num_pairs: usize,
    },
    /// A token position is larger than [`MAX_POSITION`].
    PositionTooLarge {
        /// The invalid position.
        position: usize,
    },
    /// A sequence of positions `start, start + 1, ..., start + length − 1`
    /// has a last position that is larger than [`MAX_POSITION`], or that
    /// does not fit in a `usize`.
    SequenceTooLong {
        /// The first position of the sequence.
        start: usize,
        /// The number of positions in the sequence.
        length: usize,
    },
    /// A list of inverse frequencies is empty.
    EmptyFrequencies,
    /// An inverse frequency `θⱼ` is not a finite number greater than 0.
    InvalidInverseFrequency {
        /// The index `j` of the invalid value in a list. `None` if the value
        /// was not part of a list.
        pair_index: Option<usize>,
        /// The invalid value.
        value: f64,
    },
    /// The number of elements of a table (`rows × columns`) does not fit
    /// in a `usize`.
    TableTooLarge {
        /// The number of rows (positions).
        rows: usize,
        /// The number of columns (pairs).
        columns: usize,
    },
    /// An embedding vector does not have the dimension `d` of the RoPE
    /// configuration. (Added in Stage 4.)
    EmbeddingDimensionMismatch {
        /// The dimension `d` of the configuration.
        expected: usize,
        /// The number of components of the embedding.
        actual: usize,
    },
    /// A query tensor and a key tensor have different batch sizes or
    /// sequence lengths. (Added in Stage 5.) Each query token and each key
    /// token at the same index must have the same position, so `B` and `S`
    /// must be equal. The number of heads may differ.
    QueryKeyShapeMismatch {
        /// `(B, S)` of the query tensor.
        query: (usize, usize),
        /// `(B, S)` of the key tensor.
        key: (usize, usize),
    },
    /// The rotary dimension `d` is larger than the head dimension `D`.
    /// (Added in Stage 7.) RoPE can rotate only components that exist.
    RotaryDimensionTooLarge {
        /// The rotary dimension `d`.
        rotary_dimension: usize,
        /// The head dimension `D`.
        head_dimension: usize,
    },
    /// A scaling factor `s` is not finite or `s < 1`. (Added in Stage 8.)
    InvalidScalingFactor {
        /// The invalid factor.
        factor: f64,
    },
    /// The context lengths of a scaling method are invalid: the original
    /// length `L` is 0, or the target length `L'` is smaller than `L`.
    /// (Added in Stage 8.)
    InvalidContextLengths {
        /// `L`.
        original: usize,
        /// `L'`.
        target: usize,
    },
    /// The YaRN ramp boundaries are invalid: they must be finite with
    /// `0 ≤ α < β`. (Added in Stage 8.)
    InvalidYarnRange {
        /// `α`.
        alpha: f64,
        /// `β`.
        beta: f64,
    },
    /// NTK-aware scaling uses the exponent `d/(d − 2)`, which is undefined
    /// for `d = 2`. (Added in Stage 8.)
    NtkRequiresLargerDimension {
        /// The rotary dimension `d`.
        rotary_dimension: usize,
    },
}

impl fmt::Display for RopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `match` must handle every variant. If we add a variant later and
        // forget it here, the code does not compile.
        match *self {
            RopeError::ZeroDimension => {
                write!(
                    f,
                    "the embedding dimension is 0; it must be even and at least 2"
                )
            }
            RopeError::OddDimension { dimension } => write!(
                f,
                "the embedding dimension {dimension} is odd; RoPE needs an even dimension, \
                 because it rotates the dimensions in pairs"
            ),
            RopeError::InvalidBase { base } => write!(
                f,
                "the frequency base {base} is not valid; it must be a finite number greater than 1"
            ),
            RopeError::PairIndexOutOfRange {
                pair_index,
                num_pairs,
            } => write!(
                f,
                "the pair index {pair_index} is out of range; valid indices are 0 to {}",
                // A valid configuration always has `num_pairs ≥ 1`.
                // `saturating_sub` gives 0 instead of an overflow if it is 0.
                num_pairs.saturating_sub(1)
            ),
            RopeError::PositionTooLarge { position } => write!(
                f,
                "the position {position} is larger than the maximum position {MAX_POSITION} (2^53)"
            ),
            RopeError::SequenceTooLong { start, length } => write!(
                f,
                "the sequence with start {start} and length {length} has a last position \
                 larger than the maximum position {MAX_POSITION} (2^53)"
            ),
            RopeError::EmptyFrequencies => write!(f, "the list of inverse frequencies is empty"),
            RopeError::InvalidInverseFrequency {
                pair_index: Some(j),
                value,
            } => write!(
                f,
                "the inverse frequency of pair {j} is {value}; it must be a finite number greater than 0"
            ),
            RopeError::InvalidInverseFrequency {
                pair_index: None,
                value,
            } => write!(
                f,
                "the inverse frequency {value} is not valid; it must be a finite number greater than 0"
            ),
            RopeError::TableTooLarge { rows, columns } => write!(
                f,
                "a table with {rows} rows and {columns} columns has too many elements"
            ),
            RopeError::EmbeddingDimensionMismatch { expected, actual } => write!(
                f,
                "the embedding has {actual} components, but the RoPE configuration \
                 has the dimension {expected}"
            ),
            RopeError::QueryKeyShapeMismatch { query, key } => write!(
                f,
                "the query tensor has (batch, sequence) = {query:?},                  but the key tensor has {key:?}; they must be equal"
            ),
            RopeError::RotaryDimensionTooLarge {
                rotary_dimension,
                head_dimension,
            } => write!(
                f,
                "the rotary dimension {rotary_dimension} is larger than the head \
                 dimension {head_dimension}"
            ),
            RopeError::InvalidScalingFactor { factor } => write!(
                f,
                "the scaling factor {factor} is not valid; it must be a finite number of at least 1"
            ),
            RopeError::InvalidContextLengths { original, target } => write!(
                f,
                "the context lengths are not valid (original {original}, target {target}); \
                 the original must be at least 1 and the target at least the original"
            ),
            RopeError::InvalidYarnRange { alpha, beta } => write!(
                f,
                "the YaRN range alpha = {alpha}, beta = {beta} is not valid; \
                 it must be finite with 0 <= alpha < beta"
            ),
            RopeError::NtkRequiresLargerDimension { rotary_dimension } => write!(
                f,
                "NTK-aware scaling needs a rotary dimension of at least 4, got {rotary_dimension}"
            ),
        }
    }
}

// All methods of `std::error::Error` have default implementations. The empty
// `impl` makes `RopeError` usable with `?` in functions that return
// `Box<dyn std::error::Error>`.
impl std::error::Error for RopeError {}

/// Checks that the embedding dimension `d` is valid for RoPE.
///
/// RoPE divides the `d` dimensions into `d/2` pairs. Thus `d` must be even.
/// `d = 0` is even, but it gives no pairs, so it is also invalid.
///
/// # Errors
///
/// - [`RopeError::ZeroDimension`] if `dimension == 0`.
/// - [`RopeError::OddDimension`] if `dimension` is odd.
///
/// # Example
///
/// ```
/// use rust_rope::utils::validation::{validate_dimension, RopeError};
///
/// assert_eq!(validate_dimension(64), Ok(()));
/// assert_eq!(validate_dimension(7), Err(RopeError::OddDimension { dimension: 7 }));
/// ```
pub fn validate_dimension(dimension: usize) -> Result<(), RopeError> {
    if dimension == 0 {
        return Err(RopeError::ZeroDimension);
    }
    // `%` is the remainder operator. An even number has remainder 0.
    if dimension % 2 != 0 {
        return Err(RopeError::OddDimension { dimension });
    }
    Ok(())
}

/// Checks that the frequency base `b` is valid for RoPE.
///
/// The rule is: `b` is finite and `b > 1`.
///
/// - `b > 1` makes the inverse frequencies `θⱼ = b^(−2j/d)` **decrease** as
///   the pair index `j` increases. This gives each pair a different
///   rotation speed: from 1 radian per position down to about `1/b`.
/// - `b = 1` gives `θⱼ = 1` for all pairs. All pairs rotate at the same
///   speed, so the extra pairs carry no extra positional information.
/// - `0 < b < 1` gives frequencies greater than 1 that increase with `j`.
///   This is the opposite of the RoPE design.
/// - `b ≤ 0`: the power `b^x` is not a real number for most `x`.
/// - `NaN` and infinity are never meaningful.
///
/// # Errors
///
/// [`RopeError::InvalidBase`] if `b` is not finite or `b ≤ 1`.
///
/// # Example
///
/// ```
/// use rust_rope::utils::validation::validate_base;
///
/// assert!(validate_base(10_000.0).is_ok());
/// assert!(validate_base(1.0).is_err());
/// assert!(validate_base(f64::NAN).is_err());
/// ```
pub fn validate_base(base: f64) -> Result<(), RopeError> {
    // A comparison with NaN is always false. Thus `base > 1.0` is false for
    // NaN, and the check rejects NaN. `is_finite` rejects +∞.
    // (−∞ is already rejected by `base > 1.0`.)
    if base > 1.0 && base.is_finite() {
        Ok(())
    } else {
        Err(RopeError::InvalidBase { base })
    }
}

/// Checks that a token position `m` is at most [`MAX_POSITION`] (`2^53`).
///
/// A position is a `usize`, so it cannot be negative. The only problem is a
/// position that is too large to convert exactly to `f64`.
///
/// # Errors
///
/// [`RopeError::PositionTooLarge`] if `position > 2^53`.
pub fn validate_position(position: usize) -> Result<(), RopeError> {
    // `as u64` is a lossless conversion: a `usize` has 32 or 64 bits.
    if position as u64 > MAX_POSITION {
        Err(RopeError::PositionTooLarge { position })
    } else {
        Ok(())
    }
}

/// Checks that one inverse frequency `θ` is a finite number greater than 0.
///
/// `θ` is a rotation speed in radians per position. `θ = 0` means "no
/// rotation", so the pair would carry no positional information. A negative
/// `θ` would rotate in the other direction; the standard formula never
/// produces it. `NaN` and infinity would make every angle invalid.
///
/// # Errors
///
/// [`RopeError::InvalidInverseFrequency`] with `pair_index: None`.
pub fn validate_inverse_frequency(value: f64) -> Result<(), RopeError> {
    // The same NaN logic as in `validate_base`.
    if value > 0.0 && value.is_finite() {
        Ok(())
    } else {
        Err(RopeError::InvalidInverseFrequency {
            pair_index: None,
            value,
        })
    }
}

/// Checks that a list of inverse frequencies is not empty, and that each
/// value is a finite number greater than 0.
///
/// The function borrows the list (`&[f64]`). It only reads the values.
///
/// # Errors
///
/// - [`RopeError::EmptyFrequencies`] if the list is empty.
/// - [`RopeError::InvalidInverseFrequency`] with `pair_index: Some(j)` for
///   the **first** invalid value. `j` is its index in the list.
pub fn validate_inverse_frequencies(values: &[f64]) -> Result<(), RopeError> {
    if values.is_empty() {
        return Err(RopeError::EmptyFrequencies);
    }
    // `iter()` gives a reference `&f64` to each element. `enumerate()` adds
    // the index: it gives pairs `(0, &v0), (1, &v1), ...`. The pattern
    // `(j, &value)` takes the pair apart and copies the `f64` out of the
    // reference.
    for (j, &value) in values.iter().enumerate() {
        if validate_inverse_frequency(value).is_err() {
            return Err(RopeError::InvalidInverseFrequency {
                pair_index: Some(j),
                value,
            });
        }
    }
    Ok(())
}

/// Checks that an embedding has exactly `expected` components.
///
/// RoPE divides the embedding into `d/2` pairs and rotates pair `j` with
/// the frequency `θⱼ`. If the embedding has a different length, some pairs
/// would have no frequency, or some frequencies would have no pair. (Added
/// in Stage 4.)
///
/// An odd length is also caught by this check, because `expected` is the
/// validated, even dimension `d` of a `RopeConfig`.
///
/// # Errors
///
/// [`RopeError::EmbeddingDimensionMismatch`] if `actual != expected`.
///
/// # Example
///
/// ```
/// use rust_rope::utils::validation::{validate_embedding_dimension, RopeError};
///
/// assert_eq!(validate_embedding_dimension(8, 8), Ok(()));
/// assert_eq!(
///     validate_embedding_dimension(7, 8),
///     Err(RopeError::EmbeddingDimensionMismatch { expected: 8, actual: 7 })
/// );
/// ```
pub fn validate_embedding_dimension(actual: usize, expected: usize) -> Result<(), RopeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RopeError::EmbeddingDimensionMismatch { expected, actual })
    }
}

/// Checks that the head dimension `D` is at least the rotary dimension `d`.
/// (Added in Stage 7.)
///
/// `d` itself must already be valid (even, ≥ 2). `D` may be odd: only the
/// first `d` components form pairs.
///
/// # Errors
///
/// [`RopeError::RotaryDimensionTooLarge`] if `head_dimension < rotary_dimension`.
pub fn validate_head_dimension(
    head_dimension: usize,
    rotary_dimension: usize,
) -> Result<(), RopeError> {
    if head_dimension >= rotary_dimension {
        Ok(())
    } else {
        Err(RopeError::RotaryDimensionTooLarge {
            rotary_dimension,
            head_dimension,
        })
    }
}

/// Checks a scaling factor `s`: finite and `s ≥ 1`. (Added in Stage 8.)
///
/// `s = 1` means "no scaling". `s < 1` would shrink the context; no scaling
/// method in this project uses it.
///
/// # Errors
///
/// [`RopeError::InvalidScalingFactor`].
pub fn validate_scaling_factor(factor: f64) -> Result<(), RopeError> {
    // Written for the valid case, so that NaN is rejected (Entry 7, 4.6).
    if factor >= 1.0 && factor.is_finite() {
        Ok(())
    } else {
        Err(RopeError::InvalidScalingFactor { factor })
    }
}

/// Checks context lengths: `L ≥ 1` and `L' ≥ L`. (Added in Stage 8.)
///
/// # Errors
///
/// [`RopeError::InvalidContextLengths`].
pub fn validate_context_lengths(original: usize, target: usize) -> Result<(), RopeError> {
    if original >= 1 && target >= original {
        Ok(())
    } else {
        Err(RopeError::InvalidContextLengths { original, target })
    }
}

/// Checks the YaRN ramp boundaries: finite and `0 ≤ α < β`. (Added in
/// Stage 8.)
///
/// # Errors
///
/// [`RopeError::InvalidYarnRange`].
pub fn validate_yarn_range(alpha: f64, beta: f64) -> Result<(), RopeError> {
    if alpha >= 0.0 && beta > alpha && beta.is_finite() {
        Ok(())
    } else {
        Err(RopeError::InvalidYarnRange { alpha, beta })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_parameter_checks() {
        assert!(validate_scaling_factor(1.0).is_ok());
        assert!(validate_scaling_factor(32.0).is_ok());
        assert!(validate_scaling_factor(0.99).is_err());
        assert!(validate_scaling_factor(f64::NAN).is_err());
        assert!(validate_context_lengths(2048, 8192).is_ok());
        assert!(validate_context_lengths(2048, 2048).is_ok());
        assert!(validate_context_lengths(0, 8192).is_err());
        assert!(validate_context_lengths(4096, 2048).is_err());
        assert!(validate_yarn_range(1.0, 32.0).is_ok());
        assert!(validate_yarn_range(32.0, 32.0).is_err());
        assert!(validate_yarn_range(-1.0, 32.0).is_err());
        assert!(validate_yarn_range(1.0, f64::INFINITY).is_err());
        assert!(validate_yarn_range(f64::NAN, 32.0).is_err());
    }

    #[test]
    fn head_dimension_must_not_be_smaller_than_rotary_dimension() {
        assert_eq!(validate_head_dimension(128, 64), Ok(()));
        assert_eq!(validate_head_dimension(64, 64), Ok(()));
        assert!(validate_head_dimension(62, 64).is_err());
    }

    #[test]
    fn embedding_dimension_must_match_exactly() {
        assert_eq!(validate_embedding_dimension(64, 64), Ok(()));
        for actual in [0, 7, 62, 66, 128] {
            assert_eq!(
                validate_embedding_dimension(actual, 64),
                Err(RopeError::EmbeddingDimensionMismatch {
                    expected: 64,
                    actual
                })
            );
        }
        let message = RopeError::EmbeddingDimensionMismatch {
            expected: 8,
            actual: 6,
        }
        .to_string();
        assert!(message.contains('6') && message.contains('8'), "{message}");
    }

    #[test]
    fn even_nonzero_dimensions_are_valid() {
        for d in [2, 4, 8, 64, 128, 4096] {
            assert_eq!(validate_dimension(d), Ok(()), "d = {d}");
        }
    }

    #[test]
    fn zero_and_odd_dimensions_are_invalid() {
        assert_eq!(validate_dimension(0), Err(RopeError::ZeroDimension));
        for d in [1, 3, 7, 63, 127] {
            assert_eq!(
                validate_dimension(d),
                Err(RopeError::OddDimension { dimension: d })
            );
        }
    }

    #[test]
    fn bases_greater_than_one_are_valid() {
        for b in [1.000_001, 2.0, 500.0, 10_000.0, 500_000.0, 1e9] {
            assert_eq!(validate_base(b), Ok(()), "b = {b}");
        }
    }

    #[test]
    fn other_bases_are_invalid() {
        for b in [
            1.0,
            0.5,
            0.0,
            -0.0,
            -10_000.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(validate_base(b), Err(RopeError::InvalidBase { base: b }));
        }
        // NaN ≠ NaN, so `assert_eq!` cannot compare the error. Use `matches!`.
        assert!(matches!(
            validate_base(f64::NAN),
            Err(RopeError::InvalidBase { .. })
        ));
    }

    #[test]
    fn position_limit_is_two_to_the_53() {
        assert_eq!(validate_position(0), Ok(()));
        assert_eq!(validate_position(1_000_000), Ok(()));
        // This test needs a 64-bit `usize`, because 2^53 does not fit in 32 bits.
        #[cfg(target_pointer_width = "64")]
        {
            let max = MAX_POSITION as usize;
            assert_eq!(validate_position(max), Ok(()));
            assert_eq!(
                validate_position(max + 1),
                Err(RopeError::PositionTooLarge { position: max + 1 })
            );
            // The reason for the limit: above 2^53, f64 cannot separate
            // neighbouring integers.
            assert_eq!(max as f64, (max + 1) as f64);
            assert_ne!((max - 1) as f64, max as f64);
        }
    }

    #[test]
    fn inverse_frequency_rules() {
        assert_eq!(validate_inverse_frequency(1.0), Ok(()));
        assert_eq!(validate_inverse_frequency(1e-300), Ok(()));
        for v in [0.0, -1.0, f64::INFINITY] {
            assert_eq!(
                validate_inverse_frequency(v),
                Err(RopeError::InvalidInverseFrequency {
                    pair_index: None,
                    value: v
                })
            );
        }
        assert!(validate_inverse_frequency(f64::NAN).is_err());
    }

    #[test]
    fn inverse_frequency_list_reports_the_first_bad_index() {
        assert_eq!(validate_inverse_frequencies(&[1.0, 0.1, 0.01]), Ok(()));
        assert_eq!(
            validate_inverse_frequencies(&[]),
            Err(RopeError::EmptyFrequencies)
        );
        assert_eq!(
            validate_inverse_frequencies(&[1.0, 0.1, -0.5, 0.0]),
            Err(RopeError::InvalidInverseFrequency {
                pair_index: Some(2),
                value: -0.5
            })
        );
    }

    #[test]
    fn error_messages_contain_the_invalid_value() {
        let message = RopeError::OddDimension { dimension: 7 }.to_string();
        assert!(message.contains('7'), "{message}");
        let message = RopeError::PairIndexOutOfRange {
            pair_index: 4,
            num_pairs: 4,
        }
        .to_string();
        assert!(message.contains("0 to 3"), "{message}");
    }
}
