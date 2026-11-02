//! Vector rotation operations.
//!
//! ## Implemented (Stage 4)
//!
//! - [`rotate_pair`]: rotates one pair of embedding dimensions
//!   `(x₂ⱼ, x₂ⱼ₊₁)` by the angle `φ`.
//!
//! ## Implemented (Stage 9)
//!
//! - [`rotate_pair_sin_cos`]: the same rotation with precomputed `sin φ` and
//!   `cos φ` (from a [`crate::rope::frequencies::SinCosTable`]). It skips
//!   the `sin_cos` call, which is the most expensive step of the rotation.
//!
//! ## The equation
//!
//! ```text
//! x'₂ⱼ   = x₂ⱼ · cos φ − x₂ⱼ₊₁ · sin φ
//! x'₂ⱼ₊₁ = x₂ⱼ · sin φ + x₂ⱼ₊₁ · cos φ
//! ```
//!
//! This is the rotation matrix `R(φ)` (Entry 5) applied to the pair:
//!
//! ```text
//! [ x'₂ⱼ   ]   [ cos φ  −sin φ ] [ x₂ⱼ   ]
//! [ x'₂ⱼ₊₁ ] = [ sin φ   cos φ ] [ x₂ⱼ₊₁ ]
//! ```
//!
//! In RoPE, `φ = φ(m, j) = m · θⱼ` (see [`crate::rope::frequencies`]).
//!
//! ## Reuse of `math`
//!
//! The general rotation `R(θ)·v` already exists:
//! [`crate::math::matrix::rotate_vector`]. It calculates `sin` and `cos`
//! with one `sin_cos` call, builds `R(θ)` and multiplies. [`rotate_pair`]
//! calls it. Thus the rotation formula exists in exactly one place in the
//! library. This module gives the operation its RoPE name and documents the
//! RoPE meaning of the inputs.

use crate::math::matrix::{Vector2, rotate_vector, rotate_vector_sin_cos};

/// Rotates one pair of embedding values `[x₂ⱼ, x₂ⱼ₊₁]` by the angle `φ`
/// (in radians) and returns the rotated pair `[x'₂ⱼ, x'₂ⱼ₊₁]`.
///
/// ```text
/// x'₂ⱼ   = x₂ⱼ · cos φ − x₂ⱼ₊₁ · sin φ
/// x'₂ⱼ₊₁ = x₂ⱼ · sin φ + x₂ⱼ₊₁ · cos φ
/// ```
///
/// - The rotation keeps the length: `x'₂ⱼ² + x'₂ⱼ₊₁² = x₂ⱼ² + x₂ⱼ₊₁²`.
/// - `φ = 0` returns the pair unchanged.
/// - Two rotations add: `rotate_pair(rotate_pair(p, a), b) ≈ rotate_pair(p, a + b)`.
///
/// # Ownership
///
/// `Vector2` is `[f64; 2]`, a `Copy` type of 16 bytes. The function receives
/// a **copy** of the pair and returns a **new** pair by value. Nothing is
/// borrowed, nothing is allocated on the heap. The caller's pair does not
/// change.
///
/// # Example
///
/// ```
/// use rust_rope::rope::rotations::rotate_pair;
/// use rust_rope::math::{approx_eq, DEFAULT_TOLERANCE};
/// use std::f64::consts::FRAC_PI_2;
///
/// // A quarter turn: (3, 4) → (−4, 3).
/// let [a, b] = rotate_pair([3.0, 4.0], FRAC_PI_2);
/// assert!(approx_eq(a, -4.0, DEFAULT_TOLERANCE));
/// assert!(approx_eq(b, 3.0, DEFAULT_TOLERANCE));
///
/// // Angle 0: no change (exactly).
/// assert_eq!(rotate_pair([3.0, 4.0], 0.0), [3.0, 4.0]);
/// ```
#[inline]
pub fn rotate_pair(pair: Vector2, phi: f64) -> Vector2 {
    // `rotate_vector` = `Matrix2::rotation(φ).mul_vector(pair)`:
    //   one `sin_cos(φ)` call, then 4 multiplications and 2 additions.
    rotate_vector(pair, phi)
}

/// Rotates one pair with precomputed `sin φ` and `cos φ` (Stage 9).
///
/// ```text
/// x'₂ⱼ   = x₂ⱼ · cos φ − x₂ⱼ₊₁ · sin φ
/// x'₂ⱼ₊₁ = x₂ⱼ · sin φ + x₂ⱼ₊₁ · cos φ
/// ```
///
/// The cost is 4 multiplications and 2 additions: no trigonometry. If
/// `(sin, cos) == φ.sin_cos()`, the result is **bit-identical** to
/// [`rotate_pair`]`(pair, φ)`. Both call the same matrix code.
///
/// # Example
///
/// ```
/// use rust_rope::rope::rotations::{rotate_pair, rotate_pair_sin_cos};
///
/// let phi = 1.25_f64;
/// let (sin, cos) = phi.sin_cos();
/// assert_eq!(rotate_pair_sin_cos([3.0, 4.0], sin, cos), rotate_pair([3.0, 4.0], phi));
/// ```
#[inline]
pub fn rotate_pair_sin_cos(pair: Vector2, sin: f64, cos: f64) -> Vector2 {
    rotate_vector_sin_cos(pair, sin, cos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::complex::Complex;
    use crate::math::{DEFAULT_TOLERANCE, approx_eq};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn close(a: Vector2, b: Vector2) -> bool {
        approx_eq(a[0], b[0], DEFAULT_TOLERANCE) && approx_eq(a[1], b[1], DEFAULT_TOLERANCE)
    }

    #[test]
    fn matches_the_rope_formula() {
        let [x0, x1] = [1.5, -2.0];
        for phi in [-3.0_f64, -0.5, 0.0, 0.001, 0.1, 1.0, 2.5, 100.0] {
            let (s, c) = phi.sin_cos();
            let expected = [x0 * c - x1 * s, x0 * s + x1 * c];
            assert!(close(rotate_pair([x0, x1], phi), expected), "φ = {phi}");
        }
    }

    #[test]
    fn precomputed_sin_cos_is_bit_identical() {
        for phi in [-3.0_f64, 0.0, 1e-9, 0.5, 2.0, 1e4, 1e9] {
            let (s, c) = phi.sin_cos();
            for pair in [[1.0, 0.0], [1.5, -2.0], [-1e10, 3e-8]] {
                assert_eq!(rotate_pair_sin_cos(pair, s, c), rotate_pair(pair, phi));
            }
        }
    }

    #[test]
    fn angle_zero_is_the_identity() {
        // cos 0 = 1 and sin 0 = 0 exactly, so the result is exact.
        for pair in [[0.0, 0.0], [1.0, 2.0], [-3.5, 1e-9], [1e10, -1e10]] {
            assert_eq!(rotate_pair(pair, 0.0), pair);
        }
    }

    #[test]
    fn quarter_and_half_turns() {
        assert!(close(rotate_pair([1.0, 0.0], FRAC_PI_2), [0.0, 1.0]));
        assert!(close(rotate_pair([1.0, 2.0], PI), [-1.0, -2.0]));
    }

    #[test]
    fn keeps_the_length() {
        for phi in [0.3, 1.0, 4.0, 1e4] {
            let [a, b] = rotate_pair([3.0, -4.0], phi);
            assert!(
                approx_eq(a * a + b * b, 25.0, DEFAULT_TOLERANCE),
                "φ = {phi}"
            );
        }
    }

    #[test]
    fn two_rotations_add_their_angles() {
        let p = [0.7, -1.3];
        for (a, b) in [(0.1, 0.2), (1.0, -2.5), (3.0, 3.0)] {
            assert!(close(
                rotate_pair(rotate_pair(p, a), b),
                rotate_pair(p, a + b)
            ));
        }
    }

    #[test]
    fn equals_complex_multiplication_by_euler_factor() {
        // (x + iy)·e^(iφ): the same rotation in complex form (Entry 6).
        for phi in [0.0, 0.1, 1.0, -2.0] {
            let z = Complex::new(2.0, -0.5).rotate(phi);
            assert_eq!(rotate_pair([2.0, -0.5], phi), [z.re(), z.im()]);
        }
    }

    #[test]
    fn input_pair_is_not_changed() {
        let pair = [1.0, 2.0];
        let _rotated = rotate_pair(pair, 1.0);
        // `pair` was copied into the function, so it is still usable.
        assert_eq!(pair, [1.0, 2.0]);
    }
}
