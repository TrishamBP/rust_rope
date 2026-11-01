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
//! ## Review for Stage 4 (the RoPE transformation)
//!
//! The transformation in [`crate::rope::embeddings`] needs exactly these two
//! parameters. We did not add more options.
//!
//! | Parameter | Symbol | Rule | Effect on the transformation |
//! |---|---|---|---|
//! | `dimension` | `d` | even, `≥ 2` | The input vector must have `d` components. RoPE forms `d/2` pairs. |
//! | `base` | `b` | finite, `> 1` | Sets the speeds `θⱼ = b^(−2j/d)`: from 1 rad/position down to about `1/b`. |
//!
//! Options that we deliberately did **not** add:
//!
//! - **Dimension layout.** The implementation uses only the interleaved
//!   (adjacent-pair) layout. One fixed layout needs no option.
//! - **Partial rotation** (rotate only the first `r < d` dimensions, as some
//!   models do). Not needed now. The caller can pass the first `r`
//!   components as a sub-slice with a configuration of dimension `r`.
//! - **Maximum sequence length.** The transformation works for every valid
//!   [`crate::rope::position::Position`]. A limit belongs to a model, not to
//!   the RoPE mathematics.
//!
//! ## Design: validate one time, then trust the type
//!
//! [`RopeConfig::new`] checks `d` and `b`. If a check fails, it returns an
//! error and no `RopeConfig` exists. The fields are private, so no other code
//! can create a `RopeConfig` with invalid values. Thus every function that
//! receives a `&RopeConfig` knows that `d` is even and not zero, and that
//! `b > 1`. These functions do not need to check again.

use crate::utils::validation::{
    RopeError, validate_base, validate_dimension, validate_head_dimension,
};

/// The default frequency base `b = 10000`.
///
/// The original Transformer (Vaswani et al., 2017) uses this value for its
/// sinusoidal position encoding. The RoFormer paper (Su et al., 2021) uses
/// the same value for RoPE. Some newer models use larger bases (for example
/// 500 000 in LLaMA 3) to support longer contexts.
pub const DEFAULT_BASE: f64 = 10_000.0;

/// Which embedding components form the rotation pairs (Stage 7).
///
/// `r` is the rotary dimension. Pair `j` (with `j = 0, ..., r/2 − 1`) is:
///
/// | Variant | Pair `j` | `r = 6` |
/// |---|---|---|
/// | `Interleaved` | `(x₂ⱼ, x₂ⱼ₊₁)` (adjacent) | `(x0,x1) (x2,x3) (x4,x5)` |
/// | `SplitHalf` | `(xⱼ, x_(j + r/2))` (one from each half) | `(x0,x3) (x1,x4) (x2,x5)` |
///
/// Both layouts rotate pair `j` by `m·θⱼ`. They are not interchangeable:
/// they pair different components. Use the layout of the model whose
/// weights you use.
///
/// An `enum` value is exactly one variant. A `match` on it must handle
/// every variant, so the compiler finds every place that must change if we
/// add a layout. A string such as `"split_half"` could contain a typing
/// mistake that only fails at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeLayout {
    /// Adjacent pairs `(x₂ⱼ, x₂ⱼ₊₁)`. The RoFormer paper and the original
    /// LLaMA (complex-number) code. The default.
    #[default]
    Interleaved,
    /// Pairs `(xⱼ, x_(j + r/2))`. The "rotate_half" form of many PyTorch
    /// implementations (for example Hugging Face Llama and GPT-NeoX).
    SplitHalf,
}

/// The parameters that define the RoPE frequencies.
///
/// - `dimension` is `d`, the number of dimensions that RoPE rotates. In a
///   Transformer, this is usually the dimension of one attention head (for
///   example 64 or 128). It must be even.
/// - `base` is `b`, the frequency base. It must be finite and greater
///   than 1.
///
/// **Stage 7 additions** (with defaults that keep the earlier behaviour):
///
/// - `layout` ([`RopeLayout`]): which components form the pairs. Default:
///   `Interleaved`.
/// - `head_dimension` `D`: the length of each input vector. Default: `d`
///   (full RoPE). With `D > d` (**partial RoPE**), only the first `d`
///   components rotate; the components `d..D` are copied unchanged. The
///   frequencies use the rotary dimension: `θⱼ = b^(−2j/d)`.
///
/// ```
/// use rust_rope::rope::config::{RopeConfig, RopeLayout};
///
/// let config = RopeConfig::new(64, 10_000.0)?         // rotary dimension d = 64
///     .with_layout(RopeLayout::SplitHalf)
///     .with_head_dimension(128)?;                      // head dimension D = 128
/// assert_eq!((config.rotary_dimension(), config.head_dimension()), (64, 128));
/// assert!(RopeConfig::new(64, 10_000.0)?.with_head_dimension(32).is_err());
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
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
    /// The rotary dimension `d` (even, ≥ 2).
    dimension: usize,
    base: f64,
    /// The head dimension `D ≥ d` (Stage 7).
    head_dimension: usize,
    /// The pair layout (Stage 7).
    layout: RopeLayout,
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
        Ok(RopeConfig {
            dimension,
            base,
            head_dimension: dimension,
            layout: RopeLayout::Interleaved,
        })
    }

    /// Returns a copy of the configuration with the pair layout `layout`
    /// (Stage 7).
    ///
    /// `self` is taken **by value**: `RopeConfig` is `Copy`, so the caller's
    /// value stays usable. The method cannot fail: every layout is valid
    /// for every even rotary dimension.
    pub fn with_layout(self, layout: RopeLayout) -> Self {
        RopeConfig { layout, ..self }
    }

    /// Returns a copy of the configuration with the head dimension
    /// `head_dimension` (Stage 7). `head_dimension > d` selects partial
    /// RoPE.
    ///
    /// # Errors
    ///
    /// [`RopeError::RotaryDimensionTooLarge`] if `head_dimension < d`.
    pub fn with_head_dimension(self, head_dimension: usize) -> Result<Self, RopeError> {
        validate_head_dimension(head_dimension, self.dimension)?;
        Ok(RopeConfig {
            head_dimension,
            ..self
        })
    }

    /// Returns the rotary dimension `d` (the same value as
    /// [`RopeConfig::dimension`]).
    pub fn rotary_dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the head dimension `D`: the required length of each input
    /// vector. `D = d` unless the configuration uses partial RoPE.
    pub fn head_dimension(&self) -> usize {
        self.head_dimension
    }

    /// Returns the pair layout.
    pub fn layout(&self) -> RopeLayout {
        self.layout
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

    /// Returns the rotary dimension `d`: the number of rotated components,
    /// and the `d` of the frequency formula `θⱼ = b^(−2j/d)`.
    ///
    /// (Before Stage 7, `d` was also the input length. With partial RoPE,
    /// the input length is [`RopeConfig::head_dimension`].)
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
    fn defaults_are_full_interleaved_rope() {
        let c = RopeConfig::new(16, 10_000.0).unwrap();
        assert_eq!(c.layout(), RopeLayout::Interleaved);
        assert_eq!(c.head_dimension(), 16);
        assert_eq!(c.rotary_dimension(), 16);
        assert_eq!(RopeLayout::default(), RopeLayout::Interleaved);
    }

    #[test]
    fn builder_methods_change_only_their_field() {
        let c = RopeConfig::new(64, 500_000.0).unwrap();
        let p = c
            .with_layout(RopeLayout::SplitHalf)
            .with_head_dimension(128)
            .unwrap();
        assert_eq!(p.layout(), RopeLayout::SplitHalf);
        assert_eq!((p.rotary_dimension(), p.head_dimension()), (64, 128));
        assert_eq!(p.base(), 500_000.0);
        assert_eq!(p.num_pairs(), 32);
        // `c` is `Copy`: still usable and unchanged.
        assert_eq!(c.layout(), RopeLayout::Interleaved);
    }

    #[test]
    fn head_dimension_smaller_than_rotary_is_rejected() {
        let c = RopeConfig::new(64, 10_000.0).unwrap();
        assert_eq!(
            c.with_head_dimension(63),
            Err(RopeError::RotaryDimensionTooLarge {
                rotary_dimension: 64,
                head_dimension: 63
            })
        );
        assert!(c.with_head_dimension(64).is_ok());
        // An odd head dimension is allowed: only the rotary part needs pairs.
        assert!(c.with_head_dimension(65).is_ok());
    }

    #[test]
    fn smallest_configuration_has_one_pair() {
        assert_eq!(RopeConfig::new(2, 2.0).unwrap().num_pairs(), 1);
    }
}
