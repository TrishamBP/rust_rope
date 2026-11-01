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
//! ## Implemented (Stage 3)
//!
//! - The inverse frequencies of `d = 8, b = 10000` are `[1, 0.1, 0.01, 0.001]`.
//! - `θ₀ = 1` for every valid base.
//! - The frequencies decrease strictly, as a geometric sequence with the
//!   ratio `b^(−2/d)`, and stay in the range `(1/b, 1]`.
//! - Position 0 gives zero angles. Angles increase linearly with the
//!   position. Different pairs give different angles at nonzero positions.
//! - The angle table defines rotation matrices with the RoPE properties.
//!
//! ## Planned
//!
//! - RoPE invariants of the full vector rotation (Stage 4) and comparison
//!   against a reference implementation (Stage 7).

use rust_rope::math::complex::Complex;
use rust_rope::math::matrix::{Matrix2, Vector, Vector2, rotate_vector};
use rust_rope::math::trigonometry::unit_circle_point;
use rust_rope::math::{DEFAULT_TOLERANCE, approx_eq};
use rust_rope::rope::config::RopeConfig;
use rust_rope::rope::frequencies::{AngleTable, angle, inverse_frequencies, inverse_frequency};
use rust_rope::rope::position::{Position, PositionRange};

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

// ===========================================================================
// Stage 3: inverse frequencies and positional angles
// ===========================================================================

/// The dimensions and bases of the Stage 3 tests.
const DIMENSIONS: [usize; 3] = [8, 64, 128];
const BASES: [f64; 4] = [2.0, 10_000.0, 500_000.0, 1_000_000.0];

fn config(d: usize, b: f64) -> RopeConfig {
    RopeConfig::new(d, b).expect("the test uses only valid configurations")
}

fn pos(m: usize) -> Position {
    Position::new(m).expect("the test uses only small positions")
}

#[test]
fn worked_example_d8_b10000() {
    // θⱼ = 10000^(−2j/8) = (10^4)^(−j/4) = 10^(−j)
    let theta = inverse_frequencies(&config(8, 10_000.0));
    let expected = [1.0, 0.1, 0.01, 0.001];
    assert_eq!(theta.len(), 4);
    for (j, (&t, &e)) in theta.iter().zip(&expected).enumerate() {
        assert!(close(t, e), "j = {j}: θ = {t:e}, expected {e:e}");
    }
}

#[test]
fn reference_values_d64_and_d128() {
    // θ₁ = 10000^(−2/d) and θ_last = 10000^(−(d−2)/d), from an independent
    // calculation (θ = 10^(−4x) with x = 2j/d):
    //   d = 64:  10^(−1/8)  = 0.749894209...,  10^(−31/8) = 1.33352143e-4
    //   d = 128: 10^(−1/16) = 0.865964323...,  10^(−63/16) = 1.15478198e-4
    // Tolerance: 1e-9 relative, because the reference values have 9 digits.
    let cases = [
        (64, 0.749_894_209, 1.333_521_43e-4),
        (128, 0.865_964_323, 1.154_781_98e-4),
    ];
    for (d, theta_1, theta_last) in cases {
        let theta = inverse_frequencies(&config(d, 10_000.0));
        assert!(
            (theta[1] - theta_1).abs() <= 1e-9 * theta_1,
            "d = {d}: {:e}",
            theta[1]
        );
        let last = theta[d / 2 - 1];
        assert!(
            (last - theta_last).abs() <= 1e-8 * theta_last,
            "d = {d}: {last:e}"
        );
    }
}

#[test]
fn first_inverse_frequency_is_one_for_every_base() {
    for d in DIMENSIONS {
        for b in BASES {
            let theta_0 = inverse_frequency(&config(d, b), 0).unwrap();
            assert!(close(theta_0, 1.0), "d = {d}, b = {b}: {theta_0}");
        }
    }
}

#[test]
fn inverse_frequencies_decrease_strictly() {
    for d in DIMENSIONS {
        for b in BASES {
            let theta = inverse_frequencies(&config(d, b));
            // `windows(2)` gives every pair of neighbours: [θ₀, θ₁], [θ₁, θ₂], ...
            for (j, w) in theta.windows(2).enumerate() {
                assert!(w[1] < w[0], "d = {d}, b = {b}, j = {j}: {w:?}");
            }
        }
    }
}

#[test]
fn inverse_frequencies_form_a_geometric_sequence() {
    // θ_(j+1) / θⱼ = b^(−2(j+1)/d) / b^(−2j/d) = b^(−2/d) for every j.
    // Tolerance: each θ has a relative error of a few ε, so the ratio has
    // a relative error of about 4ε ≈ 1e-15. 1e-12 is a safe margin.
    for d in DIMENSIONS {
        for b in BASES {
            let theta = inverse_frequencies(&config(d, b));
            let ratio = b.powf(-2.0 / d as f64);
            for w in theta.windows(2) {
                assert!(close(w[1] / w[0], ratio), "d = {d}, b = {b}");
            }
        }
    }
}

#[test]
fn inverse_frequencies_stay_between_one_over_base_and_one() {
    // 0 ≤ 2j/d ≤ (d − 2)/d < 1, thus 1/b < θⱼ ≤ 1.
    for d in DIMENSIONS {
        for b in BASES {
            for t in inverse_frequencies(&config(d, b)) {
                assert!(t <= 1.0 && t > 1.0 / b, "d = {d}, b = {b}: {t:e}");
            }
        }
    }
}

#[test]
fn larger_base_gives_slower_pairs() {
    // For a fixed j > 0, a larger base gives a smaller θⱼ.
    let slow = inverse_frequencies(&config(64, 500_000.0));
    let fast = inverse_frequencies(&config(64, 10_000.0));
    assert_eq!(slow[0], fast[0]); // both are exactly 1
    for j in 1..32 {
        assert!(slow[j] < fast[j], "j = {j}");
    }
}

#[test]
fn position_zero_gives_zero_angles() {
    for d in DIMENSIONS {
        let theta = inverse_frequencies(&config(d, 10_000.0));
        let table = AngleTable::new(PositionRange::from_zero(3).unwrap(), &theta).unwrap();
        // 0 · θ is exactly 0 for every finite θ, so exact equality is correct.
        let row = table.row(Position::ZERO).unwrap();
        assert!(row.iter().all(|&phi| phi == 0.0), "d = {d}: {row:?}");
        for &t in &theta {
            assert_eq!(angle(Position::ZERO, t).unwrap(), 0.0);
        }
    }
}

#[test]
fn position_one_gives_the_inverse_frequencies() {
    // 1 · θ = θ exactly.
    let theta = inverse_frequencies(&config(64, 10_000.0));
    let table = AngleTable::new(PositionRange::from_zero(2).unwrap(), &theta).unwrap();
    assert_eq!(table.row(pos(1)).unwrap(), theta.as_slice());
}

#[test]
fn angles_increase_linearly_with_position() {
    // φ(m, j) = m·θⱼ. Thus φ(m + 1, j) − φ(m, j) = θⱼ, and φ(m, j) = m·φ(1, j).
    // Tolerance: φ(m, j) has an absolute error of about m·θⱼ·ε ≤ 64·1.1e-16.
    for d in DIMENSIONS {
        let theta = inverse_frequencies(&config(d, 10_000.0));
        let table = AngleTable::new(PositionRange::from_zero(65).unwrap(), &theta).unwrap();
        for (j, &t) in theta.iter().enumerate() {
            for m in 0..64 {
                let now = table.get(pos(m), j).unwrap();
                let next = table.get(pos(m + 1), j).unwrap();
                assert!(close(next - now, t), "d = {d}, j = {j}, m = {m}");
                assert!(close(now, m as f64 * t), "d = {d}, j = {j}, m = {m}");
            }
        }
    }
}

#[test]
fn different_pairs_give_different_angles_at_nonzero_positions() {
    // θⱼ are all different, so m·θⱼ are all different for m ≠ 0.
    for d in DIMENSIONS {
        let theta = inverse_frequencies(&config(d, 10_000.0));
        let table = AngleTable::new(PositionRange::from_zero(8).unwrap(), &theta).unwrap();
        for m in 1..8 {
            let row = table.row(pos(m)).unwrap();
            for w in row.windows(2) {
                // Strictly decreasing, thus pairwise different.
                assert!(w[1] < w[0], "d = {d}, m = {m}: {w:?}");
                assert!(!close(w[0], w[1]), "d = {d}, m = {m}: {w:?}");
            }
        }
    }
}

#[test]
fn start_offset_table_equals_the_matching_rows_of_a_table_from_zero() {
    let theta = inverse_frequencies(&config(8, 10_000.0));
    let full = AngleTable::new(PositionRange::from_zero(20).unwrap(), &theta).unwrap();
    let part = AngleTable::new(PositionRange::new(12, 8).unwrap(), &theta).unwrap();
    for m in part.positions().iter() {
        // The same arithmetic (m as f64 · θ), so the results are bit-for-bit equal.
        assert_eq!(part.row(m), full.row(m), "m = {}", m.value());
    }
}

#[test]
fn angle_table_defines_rope_rotation_matrices() {
    // R(m, j) = R(m·θⱼ). The table gives the angle; Stage 2 gives R.
    // Check: R(0, j) = I, R(m, j) = R(1, j)^m, and R(m, j)ᵀ·R(n, j) = R(n − m, j).
    let theta = inverse_frequencies(&config(8, 10_000.0));
    let table = AngleTable::new(PositionRange::from_zero(8).unwrap(), &theta).unwrap();
    let r = |m: usize, j: usize| Matrix2::rotation(table.get(pos(m), j).unwrap());
    let matrices_close = |a: Matrix2, b: Matrix2| {
        let (a, b) = (a.rows(), b.rows());
        (0..2).all(|row| (0..2).all(|col| close(a[row][col], b[row][col])))
    };
    for j in 0..4 {
        assert!(matrices_close(r(0, j), Matrix2::IDENTITY), "j = {j}");
        let mut power = Matrix2::IDENTITY;
        for m in 0..8 {
            assert!(matrices_close(r(m, j), power), "j = {j}, m = {m}");
            power = power.mul_matrix(r(1, j));
        }
        for (m, n) in [(1, 4), (2, 5), (3, 6), (4, 7)] {
            let relative = r(m, j).transpose().mul_matrix(r(n, j));
            assert!(
                matrices_close(relative, r(n - m, j)),
                "j = {j}, m = {m}, n = {n}"
            );
        }
    }
}
