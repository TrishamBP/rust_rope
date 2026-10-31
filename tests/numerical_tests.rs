//! Numerical and mathematical-invariant tests.
//!
//! These tests use the library from the outside, through the public API.
//! They check properties that connect several modules. Each module has its
//! own unit tests for the individual operations.
//!
//! ## Implemented (Stage 2)
//!
//! - The complex rotation `z·e^(iθ)` equals the matrix rotation `R(θ)·v`.
//! - Rotations keep lengths and dot products.
//! - The relative-rotation property, in matrix form and in complex form.
//!
//! ## Planned
//!
//! - RoPE invariants (Stages 3–4) and comparison against a reference
//!   implementation (Stage 7).

use rust_rope::math::complex::Complex;
use rust_rope::math::matrix::{Matrix2, Vector, Vector2, rotate_vector};
use rust_rope::math::trigonometry::unit_circle_point;
use rust_rope::math::{DEFAULT_TOLERANCE, approx_eq};

/// A set of angles that covers all four quadrants, negative angles, and a
/// large angle.
const ANGLES: [f64; 9] = [-5.0, -1.0, -0.1, 0.0, 0.4, 1.5, 3.0, 4.5, 1000.0];

/// A set of test vectors with zero, positive, negative and mixed components.
const VECTORS: [Vector2; 6] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [3.0, 4.0],
    [-2.5, 0.75],
    [-1e3, -1e-3],
];

fn close(a: f64, b: f64) -> bool {
    approx_eq(a, b, DEFAULT_TOLERANCE)
}

fn dot(a: Vector2, b: Vector2) -> f64 {
    Vector::from_slice(&a)
        .dot(&Vector::from_slice(&b))
        .expect("2-D vectors always have the same dimension")
}

#[test]
fn complex_rotation_equals_matrix_rotation() {
    // Euler: (x + iy)·e^(iθ) must equal R(θ)·[x, y] for every θ and vector.
    for theta in ANGLES {
        for [x, y] in VECTORS {
            let by_complex = Complex::new(x, y).rotate(theta);
            let [mx, my] = rotate_vector([x, y], theta);
            assert!(close(by_complex.re(), mx), "θ = {theta}, v = [{x}, {y}]");
            assert!(close(by_complex.im(), my), "θ = {theta}, v = [{x}, {y}]");
        }
    }
}

#[test]
fn unit_circle_point_equals_euler_formula() {
    // e^(iθ) = cos θ + i·sin θ is the point (cos θ, sin θ).
    for theta in ANGLES {
        let [x, y] = unit_circle_point(theta);
        let w = Complex::from_polar(1.0, theta);
        assert_eq!((w.re(), w.im()), (x, y));
    }
}

#[test]
fn rotation_matrix_columns_are_the_rotated_axis_vectors() {
    for theta in ANGLES {
        let r = Matrix2::rotation(theta);
        let [c, s] = unit_circle_point(theta);
        // Column 1 = image of [1, 0]; column 2 = image of [0, 1].
        assert_eq!(r.mul_vector([1.0, 0.0]), [c, s]);
        assert_eq!(r.mul_vector([0.0, 1.0]), [-s, c]);
    }
}

#[test]
fn rotation_keeps_lengths_in_both_representations() {
    for theta in ANGLES {
        for v in VECTORS {
            let length = Vector::from_slice(&v).magnitude();
            let by_matrix = Vector::from_slice(&rotate_vector(v, theta)).magnitude();
            let by_complex = Complex::new(v[0], v[1]).rotate(theta).magnitude();
            assert!(close(by_matrix, length));
            assert!(close(by_complex, length));
        }
    }
}

#[test]
fn relative_position_property_matrix_form() {
    // ⟨R(mθ)·q, R(nθ)·k⟩ depends only on the difference n − m.
    // This is the core property of RoPE, with one pair of dimensions.
    let theta = 0.37;
    let q = [0.8, -1.1];
    let k = [-0.3, 1.9];
    let score = |m: f64, n: f64| dot(rotate_vector(q, m * theta), rotate_vector(k, n * theta));

    let reference = score(0.0, 3.0); // n − m = 3
    for (m, n) in [(1.0, 4.0), (10.0, 13.0), (100.0, 103.0), (-5.0, -2.0)] {
        assert!(
            close(score(m, n), reference),
            "m = {m}, n = {n}: {} vs {reference}",
            score(m, n)
        );
    }
    // A different distance gives a different score.
    assert!(!close(score(0.0, 4.0), reference));
}

#[test]
fn relative_position_property_matrix_and_complex_forms_agree() {
    // The dot product ⟨u, w⟩ of two 2-D vectors equals Re[u·conj(w)] of the
    // matching complex numbers. Thus both RoPE formulations give the same
    // score.
    let theta = 0.37;
    let q = [0.8, -1.1];
    let k = [-0.3, 1.9];
    for (m, n) in [(0.0, 0.0), (2.0, 7.0), (50.0, 3.0)] {
        let by_matrix = dot(rotate_vector(q, m * theta), rotate_vector(k, n * theta));
        let qc = Complex::new(q[0], q[1]).rotate(m * theta);
        let kc = Complex::new(k[0], k[1]).rotate(n * theta);
        let by_complex = (qc * kc.conjugate()).re();
        assert!(close(by_matrix, by_complex));
    }
}

#[test]
fn many_small_rotations_drift_more_than_one_large_rotation() {
    // Precision: 10 000 rotations by θ accumulate rounding errors. One
    // rotation by 10 000·θ has only the error of one calculation.
    let theta = 0.001;
    let steps = 10_000;
    let start = [1.0, 0.0];

    let mut repeated = start;
    for _ in 0..steps {
        repeated = rotate_vector(repeated, theta);
    }
    let direct = rotate_vector(start, steps as f64 * theta);

    let error_repeated = Vector::from_slice(&repeated).magnitude() - 1.0;
    let error_direct = Vector::from_slice(&direct).magnitude() - 1.0;

    // Measured on x86-64: the length error of the repeated rotation is about
    // 7.8e-14 (about 350 × f64::EPSILON). The direct rotation has an error
    // of at most a few f64::EPSILON (measured: exactly 0).
    assert!(
        error_repeated.abs() > 10.0 * f64::EPSILON,
        "the drift is visible: {error_repeated:e}"
    );
    assert!(
        error_repeated.abs() < 1e-11,
        "but still small: {error_repeated:e}"
    );
    assert!(
        error_direct.abs() <= 2.0 * f64::EPSILON,
        "direct rotation: {error_direct:e}"
    );
    // The two end points differ only by the accumulated drift.
    assert!((direct[0] - repeated[0]).abs() < 1e-11);
    assert!((direct[1] - repeated[1]).abs() < 1e-11);
}
