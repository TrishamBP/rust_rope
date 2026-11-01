//! General mathematical primitives.
//!
//! This module contains mathematics that is not specific to RoPE.
//! It must not depend on [`crate::rope`]. This rule keeps the primitives
//! reusable and easy to test in isolation.
//!
//! ## Status (Stage 2)
//!
//! - [`matrix`]: vectors (`Vector`), 2 × 2 matrices (`Matrix2`) and
//!   two-dimensional rotations. Implemented.
//! - [`complex`]: complex numbers (`Complex`), polar form and rotation by
//!   multiplication (Euler's formula). Implemented.
//! - [`trigonometry`]: angle conversion, sine, cosine and unit-circle
//!   points. Implemented.
//! - [`approx_eq`]: tolerance-based comparison of floating-point numbers.
//!   Implemented.
//!
//! ## Status (Stage 5)
//!
//! - [`tensor`]: `Tensor4`, a four-dimensional `[B, S, H, D]` array in one
//!   flat row-major `Vec<f64>`. Implemented.
//!
//! ## Dependency order inside `math`
//!
//! `trigonometry` depends on nothing. `matrix` and `complex` use
//! `trigonometry` to calculate sine and cosine. `tensor` depends on nothing.

pub mod complex;
pub mod matrix;
pub mod tensor;
pub mod trigonometry;

/// The default tolerance for [`approx_eq`].
///
/// An `f64` value has a relative precision of about `2.2e-16`
/// ([`f64::EPSILON`]). A short calculation (a few additions,
/// multiplications, or one call to `sin` or `cos`) gathers an error of a
/// few multiples of this value. `1e-12` is about 4500 times `f64::EPSILON`.
/// It is large enough to accept these rounding errors. It is small enough
/// to detect a real mistake, for example a wrong sign or a wrong formula.
pub const DEFAULT_TOLERANCE: f64 = 1e-12;

/// Returns `true` if `a` and `b` are equal within `tolerance`.
///
/// The test is:
///
/// ```text
/// |a − b| ≤ tolerance × max(1, |a|, |b|)
/// ```
///
/// - Near zero (`|a|` and `|b|` below 1), this is an **absolute** test:
///   `|a − b| ≤ tolerance`.
/// - For large values, this is a **relative** test: the permitted error
///   grows with the size of the values. Floating-point errors also grow with
///   the size of the values, so a fixed absolute tolerance would be too
///   strict for large numbers.
///
/// Special values:
///
/// - Two equal infinities compare as equal.
/// - `NaN` ("not a number") is never equal to any value, also not to `NaN`.
///
/// # Example
///
/// ```
/// use rust_rope::math::{approx_eq, DEFAULT_TOLERANCE};
///
/// // 0.1 + 0.2 is not exactly 0.3 in binary floating point.
/// assert_ne!(0.1 + 0.2, 0.3);
/// assert!(approx_eq(0.1 + 0.2, 0.3, DEFAULT_TOLERANCE));
/// ```
pub fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
    // Exact equality handles equal infinities. For equal infinities,
    // `a - b` is NaN, and the tolerance test below would fail.
    if a == b {
        return true;
    }
    // Infinities that are not equal (for example +∞ and −∞, or +∞ and
    // 1e308) must compare as different. Without this check, the relative
    // test fails: `difference` is ∞ and `scale` is ∞, and `∞ <= ∞` is true.
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let difference = (a - b).abs();
    let scale = 1.0_f64.max(a.abs()).max(b.abs());
    // If `a` or `b` is NaN, `difference` is NaN, and `NaN <= x` is false.
    difference <= tolerance * scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_values_are_approximately_equal() {
        assert!(approx_eq(1.5, 1.5, DEFAULT_TOLERANCE));
        assert!(approx_eq(0.0, -0.0, DEFAULT_TOLERANCE));
    }

    #[test]
    fn rounding_error_is_accepted() {
        assert!(approx_eq(0.1 + 0.2, 0.3, DEFAULT_TOLERANCE));
    }

    #[test]
    fn real_difference_is_rejected() {
        assert!(!approx_eq(1.0, 1.001, DEFAULT_TOLERANCE));
        assert!(!approx_eq(1.0, -1.0, DEFAULT_TOLERANCE));
    }

    #[test]
    fn tolerance_is_absolute_near_zero() {
        // 6.1e-17 is the value of cos(π/2) in f64. Mathematically it is 0.
        assert!(approx_eq(6.123_233_995_736_766e-17, 0.0, DEFAULT_TOLERANCE));
        assert!(!approx_eq(1e-9, 0.0, DEFAULT_TOLERANCE));
    }

    #[test]
    fn tolerance_is_relative_for_large_values() {
        // The difference is 1e-4, but relative to 1e10 it is only 1e-14.
        assert!(approx_eq(1e10, 1e10 + 1e-4, DEFAULT_TOLERANCE));
        // A relative difference of 1e-6 is rejected.
        assert!(!approx_eq(1e10, 1e10 + 1e4, DEFAULT_TOLERANCE));
    }

    #[test]
    fn special_values() {
        assert!(approx_eq(f64::INFINITY, f64::INFINITY, DEFAULT_TOLERANCE));
        assert!(!approx_eq(
            f64::INFINITY,
            f64::NEG_INFINITY,
            DEFAULT_TOLERANCE
        ));
        assert!(!approx_eq(f64::NAN, f64::NAN, DEFAULT_TOLERANCE));
        assert!(!approx_eq(f64::INFINITY, f64::MAX, DEFAULT_TOLERANCE));
        assert!(!approx_eq(f64::NAN, 0.0, DEFAULT_TOLERANCE));
    }
}
