//! Sine and cosine operations.
//!
//! ## Implemented
//!
//! - [`degrees_to_radians`] and [`radians_to_degrees`]: angle conversion.
//! - [`sine`], [`cosine`] and [`sin_cos`]: the trigonometric functions.
//! - [`unit_circle_point`]: the point `(cos(θ), sin(θ))` on the unit circle.
//!
//! ## Notation
//!
//! - `θ` (the Greek letter "theta") is an angle.
//! - `π` (the Greek letter "pi") is the ratio of the circumference of a
//!   circle to its diameter: `π ≈ 3.14159`. In Rust: [`std::f64::consts::PI`].
//! - All functions in this module that take an angle expect **radians**,
//!   except [`degrees_to_radians`].

use std::f64::consts::PI;

/// Converts an angle from degrees to radians:
/// `θ_radians = θ_degrees × π / 180`.
///
/// The code calculates `θ_degrees × (π / 180)`. The compiler calculates the
/// constant `π / 180` one time, at compile time. Thus, each call needs only
/// one multiplication. This is the same formula as the standard library
/// method [`f64::to_radians`].
///
/// # Example
///
/// ```
/// use rust_rope::math::trigonometry::degrees_to_radians;
/// use std::f64::consts::PI;
///
/// assert_eq!(degrees_to_radians(180.0), PI);
/// ```
pub fn degrees_to_radians(degrees: f64) -> f64 {
    degrees * (PI / 180.0)
}

/// Converts an angle from radians to degrees:
/// `θ_degrees = θ_radians × 180 / π`.
///
/// This is the same formula as the standard library method
/// [`f64::to_degrees`].
pub fn radians_to_degrees(radians: f64) -> f64 {
    radians * (180.0 / PI)
}

/// Returns `sin(θ)` for an angle `θ` in radians.
///
/// On the unit circle, `sin(θ)` is the `y` coordinate of the point at the
/// angle `θ`. This function calls [`f64::sin`].
#[inline]
pub fn sine(theta: f64) -> f64 {
    theta.sin()
}

/// Returns `cos(θ)` for an angle `θ` in radians.
///
/// On the unit circle, `cos(θ)` is the `x` coordinate of the point at the
/// angle `θ`. This function calls [`f64::cos`].
#[inline]
pub fn cosine(theta: f64) -> f64 {
    theta.cos()
}

/// Returns `(sin(θ), cos(θ))` for an angle `θ` in radians.
///
/// Rotations always need both values. [`f64::sin_cos`] can calculate both
/// together. On some platforms this is faster than two separate calls,
/// because both functions share the first step (the reduction of `θ` into a
/// small range).
///
/// The order is `(sin, cos)`, the same order as [`f64::sin_cos`].
#[inline]
pub fn sin_cos(theta: f64) -> (f64, f64) {
    theta.sin_cos()
}

/// Returns the point `[x, y] = [cos(θ), sin(θ)]` on the unit circle.
///
/// The unit circle is the circle with radius 1 around the origin `(0, 0)`.
/// The point is at the angle `θ` (in radians), measured counterclockwise
/// from the positive `x` axis.
///
/// The return type is `[f64; 2]`: a fixed-size array on the stack.
///
/// # Example
///
/// ```
/// use rust_rope::math::trigonometry::unit_circle_point;
///
/// let [x, y] = unit_circle_point(0.0);
/// assert_eq!((x, y), (1.0, 0.0));
/// ```
pub fn unit_circle_point(theta: f64) -> [f64; 2] {
    let (sin, cos) = sin_cos(theta);
    [cos, sin]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DEFAULT_TOLERANCE, approx_eq};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_6, TAU};

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            approx_eq(actual, expected, DEFAULT_TOLERANCE),
            "expected {expected:e}, got {actual:e}"
        );
    }

    // --- Angle conversion ---------------------------------------------------

    #[test]
    fn degrees_to_radians_at_the_four_quarter_turns() {
        assert_eq!(degrees_to_radians(0.0), 0.0);
        assert_close(degrees_to_radians(90.0), FRAC_PI_2);
        assert_close(degrees_to_radians(180.0), PI);
        assert_close(degrees_to_radians(270.0), 3.0 * FRAC_PI_2);
        assert_close(degrees_to_radians(360.0), TAU);
    }

    #[test]
    fn negative_angles_convert_with_the_sign() {
        assert_close(degrees_to_radians(-90.0), -FRAC_PI_2);
        assert_close(radians_to_degrees(-PI), -180.0);
    }

    #[test]
    fn radians_to_degrees_at_the_four_quarter_turns() {
        assert_eq!(radians_to_degrees(0.0), 0.0);
        assert_close(radians_to_degrees(FRAC_PI_2), 90.0);
        assert_close(radians_to_degrees(PI), 180.0);
        assert_close(radians_to_degrees(3.0 * FRAC_PI_2), 270.0);
    }

    #[test]
    fn conversion_matches_the_standard_library_exactly() {
        // Our formula is the same as the standard library formula, so the
        // results are bit-for-bit equal. This is an exact test on purpose.
        for degrees in [-720.0, -90.0, 0.0, 1.0, 30.0, 45.0, 90.0, 123.456, 1e6] {
            assert_eq!(degrees_to_radians(degrees), f64::to_radians(degrees));
        }
        for radians in [-10.0, -1.0, 0.0, 0.5, 1.0, PI, 100.0] {
            assert_eq!(radians_to_degrees(radians), f64::to_degrees(radians));
        }
    }

    #[test]
    fn round_trip_conversion() {
        for degrees in [-270.0, -45.0, 0.0, 17.5, 90.0, 359.0, 1e4] {
            assert_close(radians_to_degrees(degrees_to_radians(degrees)), degrees);
        }
    }

    // --- Sine and cosine at the four quarter turns -------------------------

    #[test]
    fn sine_and_cosine_at_0_degrees() {
        assert_eq!(sine(0.0), 0.0);
        assert_eq!(cosine(0.0), 1.0);
    }

    #[test]
    fn sine_and_cosine_at_90_degrees() {
        assert_close(sine(FRAC_PI_2), 1.0);
        // cos(π/2) is mathematically 0, but in f64 it is about 6.1e-17.
        assert_ne!(cosine(FRAC_PI_2), 0.0);
        assert_close(cosine(FRAC_PI_2), 0.0);
    }

    #[test]
    fn sine_and_cosine_at_180_degrees() {
        // sin(π) is mathematically 0, but in f64 it is about 1.2e-16.
        assert_ne!(sine(PI), 0.0);
        assert_close(sine(PI), 0.0);
        assert_close(cosine(PI), -1.0);
    }

    #[test]
    fn sine_and_cosine_at_270_degrees() {
        let theta = 3.0 * FRAC_PI_2;
        assert_close(sine(theta), -1.0);
        assert_close(cosine(theta), 0.0);
    }

    #[test]
    fn known_values_at_30_and_45_degrees() {
        assert_close(sine(FRAC_PI_6), 0.5);
        assert_close(cosine(FRAC_PI_4), std::f64::consts::FRAC_1_SQRT_2);
    }

    // --- Properties ---------------------------------------------------------

    #[test]
    fn pythagorean_identity() {
        // sin²(θ) + cos²(θ) = 1 for all θ.
        for theta in [-7.0, -1.0, 0.0, 0.3, 1.0, 2.5, PI, 10.0, 1000.0] {
            let (s, c) = sin_cos(theta);
            assert_close(s * s + c * c, 1.0);
        }
    }

    #[test]
    fn sine_is_odd_and_cosine_is_even() {
        // sin(−θ) = −sin(θ) and cos(−θ) = cos(θ).
        for theta in [0.1, 1.0, 2.0, 3.0] {
            assert_close(sine(-theta), -sine(theta));
            assert_close(cosine(-theta), cosine(theta));
        }
    }

    #[test]
    fn periodicity_with_period_two_pi() {
        // sin(θ + 2πk) = sin(θ) and cos(θ + 2πk) = cos(θ) for integers k.
        for theta in [-1.0, 0.0, 0.5, 2.0] {
            for k in [-3.0, -1.0, 1.0, 2.0, 5.0] {
                assert_close(sine(theta + k * TAU), sine(theta));
                assert_close(cosine(theta + k * TAU), cosine(theta));
            }
        }
    }

    #[test]
    fn precision_loss_for_large_angles() {
        // Mathematically, sin(2π·10⁶) = 0. But the f64 value of 2π·10⁶ is
        // not exact. Its error is about 4.5e-10 rad, and sin returns this
        // error. The absolute error grows with the size of the angle.
        let large = 1e6 * TAU;
        let error = sine(large).abs();
        assert!(error > 1e-12, "error was {error:e}");
        assert!(error < 1e-8, "error was {error:e}");
    }

    #[test]
    fn sin_cos_equals_separate_calls() {
        for theta in [-2.0, 0.0, 0.7, PI, 50.0] {
            let (s, c) = sin_cos(theta);
            assert_close(s, sine(theta));
            assert_close(c, cosine(theta));
        }
    }

    // --- Unit circle --------------------------------------------------------

    #[test]
    fn unit_circle_points_at_the_four_quarter_turns() {
        let expected = [
            (0.0, [1.0, 0.0]),
            (FRAC_PI_2, [0.0, 1.0]),
            (PI, [-1.0, 0.0]),
            (3.0 * FRAC_PI_2, [0.0, -1.0]),
        ];
        for (theta, [ex, ey]) in expected {
            let [x, y] = unit_circle_point(theta);
            assert_close(x, ex);
            assert_close(y, ey);
        }
    }

    #[test]
    fn unit_circle_points_have_length_one() {
        for theta in [-3.0, -0.5, 0.0, 1.0, 2.0, 4.0, 100.0] {
            let [x, y] = unit_circle_point(theta);
            assert_close(x.hypot(y), 1.0);
        }
    }

    #[test]
    fn unit_circle_point_at_45_degrees() {
        let [x, y] = unit_circle_point(FRAC_PI_4);
        assert_close(x, std::f64::consts::FRAC_1_SQRT_2);
        assert_close(y, std::f64::consts::FRAC_1_SQRT_2);
    }
}
