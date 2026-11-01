//! RoPE configuration.
//!
//! ## Implemented (Stage 3)
//!
//! - [`RopeConfig`]: the two parameters that define the RoPE frequencies:
//!   the embedding dimension `d` and the frequency base `b`.
//! - [`DEFAULT_BASE`]: `b = 10000`, the value of the RoFormer paper.
//!
//! ## Planned
//!
//! - Scaling parameters for long contexts (Stage 5).
//!
//! ## Design: validate one time, then trust the type
//!
//! [`RopeConfig::new`] checks `d` and `b`. If a check fails, it returns an
//! error and no `RopeConfig` exists. The fields are private, so no other code
//! can create a `RopeConfig` with invalid values. Thus every function that
//! receives a `&RopeConfig` knows that `d` is even and not zero, and that
//! `b > 1`. These functions do not need to check again.

use crate::utils::validation::{RopeError, validate_base, validate_dimension};

/// The default frequency base `b = 10000`.
///
/// The original Transformer (Vaswani et al., 2017) uses this value for its
/// sinusoidal position encoding. The RoFormer paper (Su et al., 2021) uses
/// the same value for RoPE. Some newer models use larger bases (for example
/// 500 000 in LLaMA 3) to support longer contexts.
pub const DEFAULT_BASE: f64 = 10_000.0;

/// The parameters that define the RoPE frequencies.
///
/// - `dimension` is `d`, the number of dimensions that RoPE rotates. In a
///   Transformer, this is usually the dimension of one attention head (for
///   example 64 or 128). It must be even.
/// - `base` is `b`, the frequency base. It must be finite and greater
///   than 1.
///
/// # Example
///
/// ```
/// use rust_rope::rope::config::RopeConfig;
///
/// let config = RopeConfig::new(8, 10_000.0)?;
/// assert_eq!(config.dimension(), 8);
/// assert_eq!(config.num_pairs(), 4);
///
/// // An odd dimension is an error, not a panic.
/// assert!(RopeConfig::new(7, 10_000.0).is_err());
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    dimension: usize,
    base: f64,
}

impl RopeConfig {
    /// Creates a configuration with the embedding dimension `d` and the
    /// frequency base `b`.
    ///
    /// # Errors
    ///
    /// - [`RopeError::ZeroDimension`] or [`RopeError::OddDimension`] if `d`
    ///   is not a positive even number.
    /// - [`RopeError::InvalidBase`] if `b` is not finite or `b ≤ 1`.
    pub fn new(dimension: usize, base: f64) -> Result<Self, RopeError> {
        // `?` returns the error to the caller immediately. If the check
        // returns `Ok(())`, execution continues on the next line.
        validate_dimension(dimension)?;
        validate_base(base)?;
        Ok(RopeConfig { dimension, base })
    }

    /// Creates a configuration with the dimension `d` and the default base
    /// [`DEFAULT_BASE`] (`b = 10000`).
    ///
    /// # Errors
    ///
    /// The same dimension errors as [`RopeConfig::new`].
    pub fn with_default_base(dimension: usize) -> Result<Self, RopeError> {
        RopeConfig::new(dimension, DEFAULT_BASE)
    }

    /// Returns the embedding dimension `d`.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the frequency base `b`.
    pub fn base(&self) -> f64 {
        self.base
    }

    /// Returns the number of dimension pairs, `d/2`.
    ///
    /// The valid pair indices are `j = 0, 1, ..., d/2 − 1`. The division is
    /// exact, because `d` is even.
    pub fn num_pairs(&self) -> usize {
        self.dimension / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_configuration_keeps_its_values() {
        let config = RopeConfig::new(128, 500_000.0).unwrap();
        assert_eq!(config.dimension(), 128);
        assert_eq!(config.base(), 500_000.0);
        assert_eq!(config.num_pairs(), 64);
    }

    #[test]
    fn default_base_is_ten_thousand() {
        let config = RopeConfig::with_default_base(64).unwrap();
        assert_eq!(config.base(), 10_000.0);
        assert_eq!(config.num_pairs(), 32);
    }

    #[test]
    fn invalid_values_are_rejected() {
        assert_eq!(RopeConfig::new(0, 10_000.0), Err(RopeError::ZeroDimension));
        assert_eq!(
            RopeConfig::new(9, 10_000.0),
            Err(RopeError::OddDimension { dimension: 9 })
        );
        assert_eq!(
            RopeConfig::new(8, 1.0),
            Err(RopeError::InvalidBase { base: 1.0 })
        );
    }

    #[test]
    fn dimension_is_checked_before_base() {
        // Both values are invalid. The dimension error comes first, because
        // `new` checks the dimension first and `?` stops at the first error.
        assert_eq!(
            RopeConfig::new(3, -1.0),
            Err(RopeError::OddDimension { dimension: 3 })
        );
    }

    #[test]
    fn smallest_configuration_has_one_pair() {
        assert_eq!(RopeConfig::new(2, 2.0).unwrap().num_pairs(), 1);
    }
}
