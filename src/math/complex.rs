//! Complex number operations.
//!
//! ## Implemented
//!
//! - [`Complex`]: a complex number `z = a + b·i`, stored as two `f64`
//!   values.
//! - Construction, component access, addition, subtraction,
//!   multiplication, conjugate, magnitude and phase.
//! - The polar form [`Complex::from_polar`] (Euler's formula
//!   `e^(iφ) = cos(φ) + i·sin(φ)`) and rotation by multiplication
//!   [`Complex::rotate`].
//!
//! ## Notation
//!
//! - `i` is the imaginary unit. By definition, `i² = −1`.
//! - `a` is the real component. `b` is the imaginary component.
//! - `z̄` (z with a bar) is the conjugate of `z`.
//! - `|z|` is the magnitude of `z`. `arg(z)` is the phase of `z`.

use crate::math::trigonometry::sin_cos;
use std::ops::{Add, Mul, Sub};

/// A complex number `z = a + b·i`.
///
/// `a` is the real component (`re`). `b` is the imaginary component (`im`).
/// The value is a pair of two `f64` numbers. It is 16 bytes in size. It has
/// no heap data, so it is `Copy`.
///
/// # Example
///
/// ```
/// use rust_rope::math::complex::Complex;
///
/// let z = Complex::new(1.0, 2.0);
/// let w = Complex::new(3.0, 4.0);
///
/// // (1 + 2i)(3 + 4i) = (3 − 8) + (4 + 6)i = −5 + 10i
/// assert_eq!(z * w, Complex::new(-5.0, 10.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    /// The number `0 + 0i`.
    pub const ZERO: Complex = Complex::new(0.0, 0.0);
    /// The number `1 + 0i`.
    pub const ONE: Complex = Complex::new(1.0, 0.0);
    /// The imaginary unit `i = 0 + 1i`.
    pub const I: Complex = Complex::new(0.0, 1.0);

    /// Creates the complex number `re + im·i`.
    ///
    /// This is a `const fn`: the compiler can call it at compile time. The
    /// constants [`Complex::ZERO`], [`Complex::ONE`] and [`Complex::I`] use
    /// this.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Returns the real component `a` of `z = a + b·i`.
    pub const fn re(self) -> f64 {
        self.re
    }

    /// Returns the imaginary component `b` of `z = a + b·i`.
    pub const fn im(self) -> f64 {
        self.im
    }

    /// Returns the conjugate `z̄ = a − b·i`.
    ///
    /// Geometrically, the conjugate is the reflection of `z` across the real
    /// axis.
    pub const fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Returns the magnitude `|z| = √(a² + b²)`.
    ///
    /// The magnitude is the distance from the origin to the point `(a, b)`.
    /// This function uses [`f64::hypot`]. `hypot` calculates `√(a² + b²)`
    /// without overflow for very large `a` or `b`.
    pub fn magnitude(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Returns the phase (argument) `arg(z)` in radians, in the range
    /// `[−π, π]`.
    ///
    /// The phase is the angle between the positive real axis and the arrow
    /// from the origin to `(a, b)`. This function uses [`f64::atan2`],
    /// called as `b.atan2(a)`. `atan2` uses the signs of both `a` and `b`,
    /// so it finds the correct quadrant.
    ///
    /// The phase of `0 + 0i` is mathematically undefined. `atan2` returns
    /// `0.0` for it.
    pub fn phase(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// Creates a complex number from its polar form: magnitude `r` and
    /// phase `φ` (in radians).
    ///
    /// ```text
    /// z = r·e^(iφ) = r·(cos(φ) + i·sin(φ)) = r·cos(φ) + i·r·sin(φ)
    /// ```
    ///
    /// The middle step is **Euler's formula**: `e^(iφ) = cos(φ) + i·sin(φ)`.
    /// With `r = 1`, the result is the point at the angle `φ` on the unit
    /// circle. This function is the inverse of [`Complex::magnitude`] and
    /// [`Complex::phase`].
    ///
    /// # Example
    ///
    /// ```
    /// use rust_rope::math::complex::Complex;
    /// use rust_rope::math::{approx_eq, DEFAULT_TOLERANCE};
    /// use std::f64::consts::PI;
    ///
    /// // Euler's identity: e^(iπ) = −1.
    /// let z = Complex::from_polar(1.0, PI);
    /// assert!(approx_eq(z.re(), -1.0, DEFAULT_TOLERANCE));
    /// assert!(approx_eq(z.im(), 0.0, DEFAULT_TOLERANCE));
    /// ```
    pub fn from_polar(r: f64, phi: f64) -> Self {
        let (sin, cos) = sin_cos(phi);
        Self::new(r * cos, r * sin)
    }

    /// Rotates `z` around the origin by the angle `θ` (in radians).
    ///
    /// The rotation is a multiplication by the unit complex number
    /// `e^(iθ) = cos(θ) + i·sin(θ)`:
    ///
    /// ```text
    /// z·e^(iθ) = (a + bi)(cos θ + i·sin θ)
    ///          = (a·cos θ − b·sin θ) + (a·sin θ + b·cos θ)i
    /// ```
    ///
    /// The magnitude of `e^(iθ)` is 1, so the magnitude of `z` does not
    /// change. The phase of `e^(iθ)` is `θ`, so `θ` is added to the phase
    /// of `z`. The result is the same as the rotation matrix `R(θ)` applied
    /// to the vector `[a, b]`.
    pub fn rotate(self, theta: f64) -> Self {
        self * Self::from_polar(1.0, theta)
    }
}

/// `(a + bi) + (c + di) = (a + c) + (b + d)i`
impl Add for Complex {
    type Output = Complex;

    fn add(self, other: Complex) -> Complex {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

/// `(a + bi) − (c + di) = (a − c) + (b − d)i`
impl Sub for Complex {
    type Output = Complex;

    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.re - other.re, self.im - other.im)
    }
}

/// `(a + bi)(c + di) = (ac − bd) + (ad + bc)i`
impl Mul for Complex {
    type Output = Complex;

    fn mul(self, other: Complex) -> Complex {
        let (a, b) = (self.re, self.im);
        let (c, d) = (other.re, other.im);
        Complex::new(a * c - b * d, a * d + b * c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DEFAULT_TOLERANCE, approx_eq};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    fn assert_complex_eq(actual: Complex, expected: Complex) {
        assert!(
            approx_eq(actual.re(), expected.re(), DEFAULT_TOLERANCE)
                && approx_eq(actual.im(), expected.im(), DEFAULT_TOLERANCE),
            "expected {expected:?}, got {actual:?}"
        );
    }

    // --- Construction and access -------------------------------------------

    #[test]
    fn new_and_accessors() {
        let z = Complex::new(3.0, -4.0);
        assert_eq!(z.re(), 3.0);
        assert_eq!(z.im(), -4.0);
    }

    #[test]
    fn constants_and_default() {
        assert_eq!(Complex::ZERO, Complex::new(0.0, 0.0));
        assert_eq!(Complex::ONE, Complex::new(1.0, 0.0));
        assert_eq!(Complex::I, Complex::new(0.0, 1.0));
        assert_eq!(Complex::default(), Complex::ZERO);
    }

    #[test]
    fn copy_keeps_the_original_usable() {
        let z = Complex::new(1.0, 2.0);
        let w = z; // a bitwise copy, not a move
        assert_eq!(z, w); // `z` is still usable
    }

    // --- Addition and subtraction ------------------------------------------

    #[test]
    fn addition() {
        let sum = Complex::new(1.0, 2.0) + Complex::new(3.0, -5.0);
        assert_eq!(sum, Complex::new(4.0, -3.0));
    }

    #[test]
    fn subtraction() {
        let difference = Complex::new(1.0, 2.0) - Complex::new(3.0, -5.0);
        assert_eq!(difference, Complex::new(-2.0, 7.0));
    }

    #[test]
    fn zero_is_the_additive_identity() {
        let z = Complex::new(-2.5, 7.0);
        assert_eq!(z + Complex::ZERO, z);
        assert_eq!(z - z, Complex::ZERO);
    }

    // --- Multiplication -----------------------------------------------------

    #[test]
    fn i_squared_is_minus_one() {
        assert_eq!(Complex::I * Complex::I, Complex::new(-1.0, 0.0));
    }

    #[test]
    fn multiplication_of_normal_values() {
        // (1 + 2i)(3 + 4i) = (3 − 8) + (4 + 6)i
        assert_eq!(
            Complex::new(1.0, 2.0) * Complex::new(3.0, 4.0),
            Complex::new(-5.0, 10.0)
        );
    }

    #[test]
    fn multiplication_with_negative_values() {
        // (−1 + 3i)(2 − i) = (−2 + 3) + (1 + 6)i
        assert_eq!(
            Complex::new(-1.0, 3.0) * Complex::new(2.0, -1.0),
            Complex::new(1.0, 7.0)
        );
    }

    #[test]
    fn one_is_the_multiplicative_identity_and_zero_absorbs() {
        let z = Complex::new(-2.5, 7.0);
        assert_eq!(z * Complex::ONE, z);
        assert_eq!(z * Complex::ZERO, Complex::ZERO);
    }

    #[test]
    fn multiplication_is_commutative() {
        let z = Complex::new(0.3, -1.7);
        let w = Complex::new(2.2, 0.9);
        assert_complex_eq(z * w, w * z);
    }

    #[test]
    fn multiplication_by_i_rotates_by_a_quarter_turn() {
        // i · (1 + 0i) = 0 + 1i : the point (1, 0) moves to (0, 1).
        assert_eq!(Complex::I * Complex::ONE, Complex::I);
        // i · (0 + 1i) = −1 + 0i : the point (0, 1) moves to (−1, 0).
        assert_eq!(Complex::I * Complex::I, Complex::new(-1.0, 0.0));
    }

    #[test]
    fn magnitudes_multiply() {
        // |z·w| = |z|·|w|
        let z = Complex::new(3.0, 4.0); // |z| = 5
        let w = Complex::new(5.0, 12.0); // |w| = 13
        assert!(approx_eq((z * w).magnitude(), 65.0, DEFAULT_TOLERANCE));
    }

    // --- Conjugate ----------------------------------------------------------

    #[test]
    fn conjugate_negates_the_imaginary_component() {
        assert_eq!(Complex::new(3.0, 4.0).conjugate(), Complex::new(3.0, -4.0));
        assert_eq!(Complex::new(3.0, -4.0).conjugate(), Complex::new(3.0, 4.0));
        assert_eq!(Complex::new(2.0, 0.0).conjugate(), Complex::new(2.0, 0.0));
    }

    #[test]
    fn z_times_conjugate_is_magnitude_squared() {
        // (a + bi)(a − bi) = a² + b²
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z * z.conjugate(), Complex::new(25.0, 0.0));
    }

    // --- Magnitude ----------------------------------------------------------

    #[test]
    fn magnitude_of_normal_negative_and_zero_values() {
        assert_eq!(Complex::new(3.0, 4.0).magnitude(), 5.0);
        assert_eq!(Complex::new(-3.0, -4.0).magnitude(), 5.0);
        assert_eq!(Complex::ZERO.magnitude(), 0.0);
        assert_eq!(Complex::I.magnitude(), 1.0);
    }

    #[test]
    fn magnitude_does_not_overflow_for_large_components() {
        // A naive √(a² + b²) gives infinity here. `hypot` does not.
        let z = Complex::new(1e200, 1e200);
        let expected = 1e200 * std::f64::consts::SQRT_2;
        assert!(approx_eq(z.magnitude(), expected, DEFAULT_TOLERANCE));
    }

    // --- Phase --------------------------------------------------------------

    #[test]
    fn phase_on_the_axes() {
        assert!(approx_eq(
            Complex::new(1.0, 0.0).phase(),
            0.0,
            DEFAULT_TOLERANCE
        ));
        assert!(approx_eq(
            Complex::new(0.0, 1.0).phase(),
            FRAC_PI_2,
            DEFAULT_TOLERANCE
        ));
        assert!(approx_eq(
            Complex::new(-1.0, 0.0).phase(),
            PI,
            DEFAULT_TOLERANCE
        ));
        assert!(approx_eq(
            Complex::new(0.0, -1.0).phase(),
            -FRAC_PI_2,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn phase_finds_the_correct_quadrant() {
        // A plain atan(b / a) gives π/4 for both of these numbers.
        // atan2 uses the signs and separates them.
        assert!(approx_eq(
            Complex::new(1.0, 1.0).phase(),
            FRAC_PI_4,
            DEFAULT_TOLERANCE
        ));
        assert!(approx_eq(
            Complex::new(-1.0, -1.0).phase(),
            -3.0 * FRAC_PI_4,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn phase_of_zero_is_zero_by_convention() {
        assert_eq!(Complex::ZERO.phase(), 0.0);
    }

    #[test]
    fn phases_add_under_multiplication() {
        // arg(z·w) = arg(z) + arg(w), if the sum stays inside [−π, π].
        let z = Complex::new(1.0, 1.0); // π/4
        let w = Complex::new(0.0, 2.0); // π/2
        assert!(approx_eq(
            (z * w).phase(),
            3.0 * FRAC_PI_4,
            DEFAULT_TOLERANCE
        ));
    }

    // =======================================================================
    // Euler's formula, polar form and rotation
    // =======================================================================

    use std::f64::consts::TAU;

    #[test]
    fn euler_formula_at_the_four_quarter_turns() {
        // e^(i·0) = 1, e^(iπ/2) = i, e^(iπ) = −1, e^(i3π/2) = −i
        assert_complex_eq(Complex::from_polar(1.0, 0.0), Complex::ONE);
        assert_complex_eq(Complex::from_polar(1.0, FRAC_PI_2), Complex::I);
        assert_complex_eq(Complex::from_polar(1.0, PI), Complex::new(-1.0, 0.0));
        assert_complex_eq(
            Complex::from_polar(1.0, 3.0 * FRAC_PI_2),
            Complex::new(0.0, -1.0),
        );
    }

    #[test]
    fn eulers_identity() {
        // e^(iπ) + 1 = 0
        assert_complex_eq(Complex::from_polar(1.0, PI) + Complex::ONE, Complex::ZERO);
    }

    #[test]
    fn from_polar_with_magnitude_zero_and_negative() {
        assert_complex_eq(Complex::from_polar(0.0, 1.234), Complex::ZERO);
        // A negative magnitude points in the opposite direction.
        assert_complex_eq(Complex::from_polar(-2.0, 0.0), Complex::new(-2.0, 0.0));
    }

    #[test]
    fn polar_round_trip() {
        // from_polar(|z|, arg(z)) = z
        for z in [
            Complex::new(3.0, 4.0),
            Complex::new(-1.0, 2.0),
            Complex::new(-0.5, -0.5),
            Complex::new(2.0, -7.0),
        ] {
            assert_complex_eq(Complex::from_polar(z.magnitude(), z.phase()), z);
        }
    }

    #[test]
    fn unit_complex_numbers_have_magnitude_one() {
        for phi in [-3.0, -1.0, 0.0, 0.5, 2.0, 100.0] {
            let w = Complex::from_polar(1.0, phi);
            assert!(approx_eq(w.magnitude(), 1.0, DEFAULT_TOLERANCE));
        }
    }

    #[test]
    fn rotation_at_the_four_quarter_turns() {
        let z = Complex::new(2.0, 3.0);
        assert_complex_eq(z.rotate(0.0), Complex::new(2.0, 3.0));
        assert_complex_eq(z.rotate(FRAC_PI_2), Complex::new(-3.0, 2.0));
        assert_complex_eq(z.rotate(PI), Complex::new(-2.0, -3.0));
        assert_complex_eq(z.rotate(3.0 * FRAC_PI_2), Complex::new(3.0, -2.0));
    }

    #[test]
    fn rotation_by_a_quarter_turn_equals_multiplication_by_i() {
        let z = Complex::new(2.0, 3.0);
        assert_complex_eq(z.rotate(FRAC_PI_2), z * Complex::I);
    }

    #[test]
    fn rotation_keeps_the_magnitude_and_adds_the_phase() {
        let z = Complex::new(1.0, 1.0); // |z| = √2, arg(z) = π/4
        let rotated = z.rotate(FRAC_PI_4);
        assert!(approx_eq(
            rotated.magnitude(),
            z.magnitude(),
            DEFAULT_TOLERANCE
        ));
        assert!(approx_eq(rotated.phase(), FRAC_PI_2, DEFAULT_TOLERANCE));
        assert_complex_eq(rotated, Complex::new(0.0, std::f64::consts::SQRT_2));
    }

    #[test]
    fn rotating_zero_gives_zero() {
        assert_complex_eq(Complex::ZERO.rotate(2.0), Complex::ZERO);
    }

    #[test]
    fn rotations_compose_by_adding_angles() {
        // e^(iα)·e^(iβ) = e^(i(α + β))
        let z = Complex::new(-1.5, 0.25);
        for (alpha, beta) in [(0.3, 0.4), (1.0, -2.5), (PI, FRAC_PI_2)] {
            assert_complex_eq(z.rotate(alpha).rotate(beta), z.rotate(alpha + beta));
        }
    }

    #[test]
    fn de_moivre_formula() {
        // (e^(iθ))ⁿ = e^(inθ): n rotations by θ are one rotation by n·θ.
        let theta = 0.3;
        let w = Complex::from_polar(1.0, theta);
        let mut power = Complex::ONE;
        for _ in 0..5 {
            power = power * w;
        }
        assert_complex_eq(power, Complex::from_polar(1.0, 5.0 * theta));
    }

    #[test]
    fn conjugate_reverses_a_rotation() {
        // The conjugate of e^(iθ) is e^(−iθ).
        let theta = 0.8;
        let w = Complex::from_polar(1.0, theta);
        assert_complex_eq(w.conjugate(), Complex::from_polar(1.0, -theta));
        let z = Complex::new(2.0, -1.0);
        assert_complex_eq(z * w * w.conjugate(), z);
    }

    #[test]
    fn full_turn_returns_to_the_start() {
        let z = Complex::new(2.0, -1.0);
        assert_complex_eq(z.rotate(TAU), z);
    }

    #[test]
    fn relative_rotation_property() {
        // Re[(q·e^(iα))·conj(k·e^(iβ))] = Re[q·conj(k)·e^(i(α − β))].
        // The result depends only on α − β. RoPE uses this (Stage 3).
        let q = Complex::new(0.7, -1.2);
        let k = Complex::new(-0.4, 2.1);
        let score = |alpha: f64, beta: f64| (q.rotate(alpha) * k.rotate(beta).conjugate()).re();
        // Three pairs of angles with the same difference α − β = 0.5:
        let reference = score(0.5, 0.0);
        for (alpha, beta) in [(1.5, 1.0), (10.5, 10.0), (-2.0, -2.5)] {
            assert!(approx_eq(score(alpha, beta), reference, DEFAULT_TOLERANCE));
        }
    }
}
