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
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::frequencies::{AngleTable, angle, inverse_frequencies, inverse_frequency};
use rust_rope::rope::position::{Position, PositionRange};
use rust_rope::rope::rotations::rotate_pair;

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

// ===========================================================================
// Stage 4: the RoPE transformation of an embedding vector
// ===========================================================================

fn rotary(d: usize, b: f64) -> RotaryEmbedding {
    RotaryEmbedding::new(config(d, b))
}

/// A deterministic test vector with positive, negative and mixed values.
fn test_vector(d: usize, seed: f64) -> Vec<f64> {
    (0..d)
        .map(|k| 3.0 * (1.3 * k as f64 + seed).sin() - 0.5)
        .collect()
}

fn squared_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum()
}

fn dot_slices(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Invariant A: RoPE(x, 0) = x, exactly.
#[test]
fn invariant_a_position_zero_is_the_identity() {
    for d in [2, 4, 8, 64, 128] {
        for b in BASES {
            let x = test_vector(d, 0.5);
            // cos 0 = 1 and sin 0 = 0 exactly, so x·1 − y·0 = x exactly.
            assert_eq!(rotary(d, b).rotate(&x, Position::ZERO).unwrap(), x);
        }
    }
}

/// Invariant B: each pair keeps its length; thus the whole vector keeps
/// its length.
#[test]
fn invariant_b_pair_magnitudes_are_preserved() {
    for d in [2, 8, 128] {
        let r = rotary(d, 10_000.0);
        let x = test_vector(d, 1.0);
        for m in [1, 2, 17, 1000, 1_000_000] {
            let y = r.rotate(&x, pos(m)).unwrap();
            for (j, (p, q)) in x
                .as_chunks::<2>()
                .0
                .iter()
                .zip(y.as_chunks::<2>().0)
                .enumerate()
            {
                let before = p[0] * p[0] + p[1] * p[1];
                let after = q[0] * q[0] + q[1] * q[1];
                assert!(close(before, after), "d = {d}, m = {m}, j = {j}");
            }
            assert!(
                close(squared_norm(&x), squared_norm(&y)),
                "d = {d}, m = {m}"
            );
        }
    }
}

/// Invariant C (pair form): R(b)·(R(a)·x) = R(a + b)·x.
#[test]
fn invariant_c_pair_rotations_compose() {
    let p = [1.25, -0.75];
    for (a, b) in [
        (0.0, 0.0),
        (0.3, 0.4),
        (1.0, -1.0),
        (2.5, 3.5),
        (-7.0, 0.01),
    ] {
        let twice = rotate_pair(rotate_pair(p, a), b);
        let once = rotate_pair(p, a + b);
        assert!(
            close(twice[0], once[0]) && close(twice[1], once[1]),
            "a = {a}, b = {b}"
        );
    }
}

/// Invariant C (vector form): RoPE(RoPE(x, m), n) = RoPE(x, m + n), because
/// every angle is linear in the position: m·θⱼ + n·θⱼ = (m + n)·θⱼ.
#[test]
fn invariant_c_embedding_rotations_compose() {
    let r = rotary(8, 10_000.0);
    let x = test_vector(8, 2.0);
    for (m, n) in [(0, 5), (1, 1), (3, 4), (10, 90), (250, 750)] {
        let twice = r.rotate(&r.rotate(&x, pos(m)).unwrap(), pos(n)).unwrap();
        let once = r.rotate(&x, pos(m + n)).unwrap();
        for k in 0..8 {
            assert!(close(twice[k], once[k]), "m = {m}, n = {n}, k = {k}");
        }
    }
}

/// Invariant D: pair j receives exactly the angle m·θⱼ.
///
/// The basis vector e₂ⱼ (1 at index 2j, 0 elsewhere) rotates to
/// (cos(m·θⱼ), sin(m·θⱼ)) in pair j. Its phase (atan2) is the angle
/// (reduced to (−π, π]). All other pairs stay exactly zero.
#[test]
fn invariant_d_each_pair_gets_its_own_angle() {
    use std::f64::consts::{PI, TAU};
    let d = 8;
    let r = rotary(d, 10_000.0);
    for j in 0..d / 2 {
        let mut e = vec![0.0; d];
        e[2 * j] = 1.0;
        for m in [1, 2, 3, 7] {
            let y = r.rotate(&e, pos(m)).unwrap();
            let expected = m as f64 * r.inverse_frequencies()[j];
            // Reduce the expected angle to (−π, π], the range of `phase`.
            let reduced = expected - TAU * ((expected + PI) / TAU).floor();
            let measured = Complex::new(y[2 * j], y[2 * j + 1]).phase();
            assert!(
                close(measured, reduced),
                "j = {j}, m = {m}: {measured} vs {reduced}"
            );
            for (k, &v) in y.iter().enumerate() {
                if k / 2 != j {
                    assert_eq!(v, 0.0, "j = {j}, m = {m}, other component {k}");
                }
            }
        }
    }
}

/// The complex interpretation: pair j as z = x₂ⱼ + i·x₂ⱼ₊₁, multiplied by
/// e^(i·m·θⱼ). Same operations in the same order, so bit-for-bit equal.
#[test]
fn rope_equals_complex_multiplication_per_pair() {
    let r = rotary(16, 10_000.0);
    let x = test_vector(16, 3.0);
    for m in [0, 1, 5, 123] {
        let y = r.rotate(&x, pos(m)).unwrap();
        for (j, &theta) in r.inverse_frequencies().iter().enumerate() {
            let z = Complex::new(x[2 * j], x[2 * j + 1]).rotate(m as f64 * theta);
            assert_eq!(
                [y[2 * j], y[2 * j + 1]],
                [z.re(), z.im()],
                "m = {m}, j = {j}"
            );
        }
    }
}

/// RoPE is a linear map: RoPE(a·u + c·v) = a·RoPE(u) + c·RoPE(v).
#[test]
fn rope_is_linear() {
    let r = rotary(8, 10_000.0);
    let (u, v) = (test_vector(8, 0.1), test_vector(8, 4.2));
    let (a, c) = (2.5, -0.75);
    let mix: Vec<f64> = u.iter().zip(&v).map(|(p, q)| a * p + c * q).collect();
    for m in [1, 9, 400] {
        let left = r.rotate(&mix, pos(m)).unwrap();
        let (ru, rv) = (r.rotate(&u, pos(m)).unwrap(), r.rotate(&v, pos(m)).unwrap());
        for k in 0..8 {
            assert!(close(left[k], a * ru[k] + c * rv[k]), "m = {m}, k = {k}");
        }
    }
}

/// The relative-position property for the full vector: the dot product of
/// RoPE(u, m) and RoPE(v, n) depends only on n − m. (Each pair contributes
/// uᵀR((n − m)θⱼ)v; the pairs add up.)
#[test]
fn full_vector_dot_product_depends_only_on_the_distance() {
    let r = rotary(16, 10_000.0);
    let (u, v) = (test_vector(16, 0.7), test_vector(16, -1.9));
    let score = |m: usize, n: usize| {
        dot_slices(
            &r.rotate(&u, pos(m)).unwrap(),
            &r.rotate(&v, pos(n)).unwrap(),
        )
    };
    let reference = score(0, 6);
    for (m, n) in [(1, 7), (10, 16), (100, 106), (5000, 5006)] {
        assert!(close(score(m, n), reference), "m = {m}, n = {n}");
    }
    assert!(
        !close(score(0, 7), reference),
        "another distance gives another score"
    );
}

/// Invariant E: large positions and different bases.
///
/// - The length stays correct for every position: a rotation by any angle
///   keeps the length, and `sin² + cos² = 1` up to a few ε.
/// - The composition RoPE(RoPE(x, m), n) = RoPE(x, m + n) gets less exact as
///   m + n grows. The angle m·θ has an absolute rounding error of about
///   (m + n)·ε. The bound below is 4·ε·(m + n + 10)·max|x|.
#[test]
fn invariant_e_large_positions_and_bases() {
    let x = test_vector(8, 0.3);
    let max_abs = x.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for b in [10_000.0, 500_000.0, 1_000_000.0] {
        let r = rotary(8, b);
        for m in [1_000, 100_000, 1_000_000, 10_000_000] {
            let y = r.rotate(&x, pos(m)).unwrap();
            assert!(y.iter().all(|v| v.is_finite()));
            assert!(
                close(squared_norm(&x), squared_norm(&y)),
                "b = {b}, m = {m}"
            );

            let n = m / 3;
            let twice = r.rotate(&y, pos(n)).unwrap();
            let once = r.rotate(&x, pos(m + n)).unwrap();
            let bound = 4.0 * f64::EPSILON * (m + n + 10) as f64 * max_abs;
            for k in 0..8 {
                let error = (twice[k] - once[k]).abs();
                assert!(
                    error <= bound,
                    "b = {b}, m = {m}, k = {k}: {error:e} > {bound:e}"
                );
            }
        }
    }
}

/// The transpose undoes the rotation: rotating pair j back by −m·θⱼ gives x.
#[test]
fn rotating_back_by_the_negative_angle_restores_the_input() {
    let r = rotary(8, 10_000.0);
    let x = test_vector(8, 5.0);
    let m = pos(37);
    let y = r.rotate(&x, m).unwrap();
    for (j, &theta) in r.inverse_frequencies().iter().enumerate() {
        let back = rotate_pair([y[2 * j], y[2 * j + 1]], -(m.as_f64() * theta));
        assert!(
            close(back[0], x[2 * j]) && close(back[1], x[2 * j + 1]),
            "j = {j}"
        );
    }
}

// ===========================================================================
// Stage 5: RoPE on query and key tensors [B, S, H, D]
// ===========================================================================

use rust_rope::math::tensor::{Tensor4, TensorShape};

fn qk_tensor(shape: TensorShape, seed: f64) -> Tensor4 {
    Tensor4::from_fn(shape, |b, s, h, d| {
        2.0 * ((b * 7 + s * 5 + h * 3 + d) as f64 * 0.61 + seed).cos()
    })
    .unwrap()
}

/// Test A on tensors: start = 0 and S = 1 means every token is at position
/// 0, so RoPE is the identity.
#[test]
fn tensor_test_a_position_zero_is_the_identity() {
    let r = rotary(8, 10_000.0);
    let shape = TensorShape::new(2, 1, 3, 8);
    let (q, k) = (qk_tensor(shape, 0.0), qk_tensor(shape, 1.0));
    let (q2, k2) = r.rotate_query_key(&q, &k, Position::ZERO).unwrap();
    assert_eq!(q2, q);
    assert_eq!(k2, k);
}

/// Test B on tensors: every pair of every head vector keeps its length.
#[test]
fn tensor_test_b_pair_magnitudes_are_preserved() {
    let r = rotary(8, 10_000.0);
    let shape = TensorShape::new(2, 5, 3, 8);
    let q = qk_tensor(shape, 0.3);
    let mut q2 = q.clone();
    r.rotate_tensor_in_place(&mut q2, pos(1000)).unwrap();
    for (before, after) in q
        .as_slice()
        .as_chunks::<2>()
        .0
        .iter()
        .zip(q2.as_slice().as_chunks::<2>().0)
    {
        let (n0, n1) = (
            before[0] * before[0] + before[1] * before[1],
            after[0] * after[0] + after[1] * after[1],
        );
        assert!(close(n0, n1));
    }
}

/// Test C: (R(m)q)ᵀ(R(n)k) = qᵀ R(n − m) k for every batch element and
/// head, with query token s_q and key token s_k at positions m and n.
///
/// The right side uses `rotate(k, n − m)`: R(n − m)k is exactly RoPE at the
/// position n − m (for n ≥ m). For n < m, use the equal form
/// (R(m − n)q)ᵀ k.
#[test]
fn tensor_test_c_relative_position_property() {
    let r = rotary(8, 10_000.0);
    let shape = TensorShape::new(2, 6, 2, 8);
    let (q, k) = (qk_tensor(shape, 0.4), qk_tensor(shape, -1.1));
    let start = 20;
    let (q2, k2) = r.rotate_query_key(&q, &k, pos(start)).unwrap();
    for b in 0..2 {
        for h in 0..2 {
            for sq in 0..6 {
                for sk in 0..6 {
                    let (m, n) = (start + sq, start + sk);
                    let left =
                        dot_slices(q2.vector(b, sq, h).unwrap(), k2.vector(b, sk, h).unwrap());
                    let (qv, kv) = (q.vector(b, sq, h).unwrap(), k.vector(b, sk, h).unwrap());
                    let right = if n >= m {
                        dot_slices(qv, &r.rotate(kv, pos(n - m)).unwrap())
                    } else {
                        dot_slices(&r.rotate(qv, pos(m - n)).unwrap(), kv)
                    };
                    assert!(
                        close(left, right),
                        "b={b} h={h} m={m} n={n}: {left} vs {right}"
                    );
                }
            }
        }
    }
}

/// The score depends only on n − m: shifting the start position does not
/// change any query-key dot product.
#[test]
fn tensor_scores_do_not_depend_on_the_start_position() {
    let r = rotary(16, 10_000.0);
    let shape = TensorShape::new(1, 4, 2, 16);
    let (q, k) = (qk_tensor(shape, 0.0), qk_tensor(shape, 2.0));
    let scores = |start: usize| {
        let (q2, k2) = r.rotate_query_key(&q, &k, pos(start)).unwrap();
        let mut out = Vec::new();
        for h in 0..2 {
            for sq in 0..4 {
                for sk in 0..4 {
                    out.push(dot_slices(
                        q2.vector(0, sq, h).unwrap(),
                        k2.vector(0, sk, h).unwrap(),
                    ));
                }
            }
        }
        out
    };
    let base = scores(0);
    for start in [1, 100, 10_000] {
        for (a, b) in base.iter().zip(scores(start)) {
            assert!(close(*a, b), "start = {start}");
        }
    }
}

/// The 2-D numerical demonstration of explain.md (Entry 12): q = [1, 2],
/// k = [3, −1], θ = 0.5, m = 2, n = 5.
#[test]
fn two_dimensional_relative_position_demonstration() {
    let theta = 0.5;
    let (q, k) = ([1.0, 2.0], [3.0, -1.0]);
    let (m, n) = (2.0, 5.0);
    let q_rot = rotate_pair(q, m * theta);
    let k_rot = rotate_pair(k, n * theta);
    let rotated_score = q_rot[0] * k_rot[0] + q_rot[1] * k_rot[1];
    let k_rel = rotate_pair(k, (n - m) * theta);
    let relative_score = q[0] * k_rel[0] + q[1] * k_rel[1];
    assert!(close(rotated_score, relative_score));
    // Hand calculation (explain.md): both ≈ 7.0532021 (7.053202107896084 in f64).
    assert!(
        (rotated_score - 7.053_202_107_896_084).abs() < 1e-12,
        "{rotated_score}"
    );
}

// ===========================================================================
// Stage 7: layouts and partial RoPE
// ===========================================================================

use rust_rope::rope::config::RopeLayout;

fn rotary_layout(rotary: usize, head: usize, layout: RopeLayout) -> RotaryEmbedding {
    RotaryEmbedding::new(
        config(rotary, 10_000.0)
            .with_layout(layout)
            .with_head_dimension(head)
            .unwrap(),
    )
}

/// The invariants of Stage 4 hold for both layouts and partial RoPE:
/// position 0 is the identity, lengths are kept, rotations compose, and the
/// dot product depends only on the distance.
#[test]
fn stage_4_invariants_hold_for_every_layout_and_partial_rope() {
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        for (rotary, head) in [(8, 8), (8, 16), (64, 128), (2, 5)] {
            let r = rotary_layout(rotary, head, layout);
            let (u, v) = (test_vector(head, 0.2), test_vector(head, 1.7));
            assert_eq!(r.rotate(&u, Position::ZERO).unwrap(), u);
            let y = r.rotate(&u, pos(500)).unwrap();
            assert!(
                close(squared_norm(&u), squared_norm(&y)),
                "{layout:?} {rotary}/{head}"
            );
            let twice = r.rotate(&r.rotate(&u, pos(30)).unwrap(), pos(12)).unwrap();
            let once = r.rotate(&u, pos(42)).unwrap();
            assert!(twice.iter().zip(&once).all(|(a, b)| close(*a, *b)));
            let score = |m, n| {
                dot_slices(
                    &r.rotate(&u, pos(m)).unwrap(),
                    &r.rotate(&v, pos(n)).unwrap(),
                )
            };
            assert!(
                close(score(3, 10), score(103, 110)),
                "{layout:?} {rotary}/{head}"
            );
        }
    }
}

/// Each split-half pair j receives exactly the angle m·θⱼ.
#[test]
fn split_half_pairs_get_the_correct_angles() {
    use std::f64::consts::{PI, TAU};
    let d = 8;
    let r = rotary_layout(d, d, RopeLayout::SplitHalf);
    for j in 0..d / 2 {
        let mut e = vec![0.0; d];
        e[j] = 1.0;
        for m in [1, 2, 3] {
            let y = r.rotate(&e, pos(m)).unwrap();
            let expected = m as f64 * r.inverse_frequencies()[j];
            let reduced = expected - TAU * ((expected + PI) / TAU).floor();
            let measured = Complex::new(y[j], y[j + d / 2]).phase();
            assert!(close(measured, reduced), "j = {j}, m = {m}");
        }
    }
}

/// Both layouts use the same frequency convention θⱼ = b^(−2j/d), with d the
/// rotary dimension (not the head dimension).
#[test]
fn both_layouts_use_the_same_frequencies() {
    let a = rotary_layout(64, 128, RopeLayout::Interleaved);
    let b = rotary_layout(64, 128, RopeLayout::SplitHalf);
    assert_eq!(a.inverse_frequencies(), b.inverse_frequencies());
    assert_eq!(
        a.inverse_frequencies(),
        inverse_frequencies(&config(64, 10_000.0))
    );
}

// ===========================================================================
// Stage 8: scaling methods
// ===========================================================================

use rust_rope::rope::scaling::{RopeScaling, scaled_inverse_frequencies};

/// Position interpolation equals "standard RoPE at the position m·L/L'"
/// whenever m·L/L' is an integer.
#[test]
fn position_interpolation_equals_standard_rope_at_the_scaled_position() {
    let config = config(64, 10_000.0);
    let pi = RotaryEmbedding::with_scaling(
        config,
        RopeScaling::PositionInterpolation {
            original_context: 2048,
            target_context: 8192,
        },
    )
    .unwrap();
    let standard = RotaryEmbedding::new(config);
    let x = test_vector(64, 0.4);
    for m in [0, 4, 400, 4096, 8188] {
        let a = pi.rotate(&x, pos(m)).unwrap();
        let b = standard.rotate(&x, pos(m / 4)).unwrap();
        assert!(a.iter().zip(&b).all(|(p, q)| close(*p, *q)), "m = {m}");
    }
}

/// Interpolation keeps every angle inside the trained range: the largest
/// angle of the slowest pair at m = L' − 1 is below the angle at m = L.
#[test]
fn interpolated_angles_stay_in_the_trained_range() {
    let c = config(8, 10_000.0);
    let theta = inverse_frequencies(&c);
    let scaled = scaled_inverse_frequencies(
        &c,
        &RopeScaling::PositionInterpolation {
            original_context: 2048,
            target_context: 8192,
        },
    )
    .unwrap();
    for j in 0..4 {
        assert!(8191.0 * scaled[j] < 2048.0 * theta[j], "j = {j}");
    }
}

/// The relative-position property survives every scaling method: each
/// method only changes θⱼ (and multiplies by a constant for YaRN).
#[test]
fn relative_position_property_holds_with_scaling() {
    let c = config(16, 10_000.0);
    let (u, v) = (test_vector(16, 1.0), test_vector(16, -2.0));
    for s in [
        RopeScaling::Linear { factor: 4.0 },
        RopeScaling::NtkAware { factor: 8.0 },
        RopeScaling::yarn(16.0, 4096),
    ] {
        let r = RotaryEmbedding::with_scaling(c, s).unwrap();
        let score = |m, n| {
            dot_slices(
                &r.rotate(&u, pos(m)).unwrap(),
                &r.rotate(&v, pos(n)).unwrap(),
            )
        };
        assert!(close(score(10, 30), score(9000, 9020)), "{s:?}");
    }
}

/// NTK-aware scaling for several rotary dimensions: the fastest pair stays,
/// the slowest pair is divided by s, and the base grows as b·s^(d/(d−2)).
#[test]
fn ntk_aware_for_different_rotary_dimensions() {
    for d in [4, 8, 32, 64, 128] {
        let c = config(d, 10_000.0);
        let theta = inverse_frequencies(&c);
        for s in [2.0, 4.0, 16.0] {
            let ntk = scaled_inverse_frequencies(&c, &RopeScaling::NtkAware { factor: s }).unwrap();
            assert_eq!(ntk[0], 1.0);
            let last = d / 2 - 1;
            assert!(
                (ntk[last] / (theta[last] / s) - 1.0).abs() < 1e-13,
                "d = {d}, s = {s}"
            );
            // The geometric structure stays: a constant ratio b'^(−2/d).
            let ratio = (10_000.0 * s.powf(d as f64 / (d as f64 - 2.0))).powf(-2.0 / d as f64);
            for w in ntk.windows(2) {
                assert!(close(w[1] / w[0], ratio), "d = {d}, s = {s}");
            }
        }
    }
}

/// Numerical stability: large factors and large positions give finite
/// outputs; lengths are kept (multiplied by the YaRN factor squared).
#[test]
fn scaling_is_numerically_stable_at_large_positions() {
    let c = config(128, 500_000.0);
    let x = test_vector(128, 0.6);
    for s in [
        RopeScaling::Linear { factor: 64.0 },
        RopeScaling::NtkAware { factor: 64.0 },
        RopeScaling::yarn(64.0, 8192),
    ] {
        let r = RotaryEmbedding::with_scaling(c, s).unwrap();
        let f2 = r.attention_factor() * r.attention_factor();
        for m in [100_000, 524_287, 10_000_000] {
            let y = r.rotate(&x, pos(m)).unwrap();
            assert!(y.iter().all(|v| v.is_finite()), "{s:?} m = {m}");
            assert!(
                close(squared_norm(&y), f2 * squared_norm(&x)),
                "{s:?} m = {m}"
            );
        }
    }
}

// ===========================================================================
// Stage 9: the sin/cos table and the parallel rotation are exact
// ===========================================================================
//
// The Stage 9 functions are optimizations. An optimization must not change
// the result. The tolerance in these tests is therefore **zero**: we compare
// with `assert_eq!` (bit-identical values). This is possible because the
// table stores the result of the same `sin_cos(m·θⱼ)` call, and the rotation
// does the same multiplications and additions in the same order. Rust does
// not reorder floating-point operations and does not fuse `a·b + c` into
// one instruction unless the code asks for it.

use rust_rope::rope::frequencies::SinCosTable;
use std::num::NonZeroUsize;

/// Every combination of layout, full/partial rotation and scaling method.
fn stage9_embeddings() -> Vec<RotaryEmbedding> {
    let mut all = Vec::new();
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        for rotary in [32, 16] {
            let c = RopeConfig::new(rotary, 10_000.0)
                .unwrap()
                .with_layout(layout)
                .with_head_dimension(32)
                .unwrap();
            for s in [
                RopeScaling::None,
                RopeScaling::Linear { factor: 4.0 },
                RopeScaling::NtkAware { factor: 8.0 },
                RopeScaling::yarn(16.0, 512),
            ] {
                all.push(RotaryEmbedding::with_scaling(c, s).unwrap());
            }
        }
    }
    all
}

/// The table rotation is bit-identical to the Stage 5 rotation, for every
/// configuration and for start positions inside a larger table.
#[test]
fn table_rotation_is_bit_identical_to_direct_rotation() {
    let shape = TensorShape::new(2, 9, 3, 32);
    let input = Tensor4::from_fn(shape, |b, s, h, d| {
        ((b * 1000 + s * 100 + h * 10 + d) as f64 * 0.37).sin() * 3.0
    })
    .unwrap();
    for rope in stage9_embeddings() {
        // One table for the positions 0..4096 serves every start below.
        let table = rope
            .sin_cos_table(PositionRange::from_zero(4096).unwrap())
            .unwrap();
        for start in [0, 1, 500, 4096 - 9] {
            let mut direct = input.clone();
            rope.rotate_tensor_in_place(&mut direct, pos(start))
                .unwrap();
            let mut cached = input.clone();
            rope.rotate_tensor_with_table(&mut cached, &table, pos(start))
                .unwrap();
            assert_eq!(cached, direct, "{:?} start = {start}", rope.config());
        }
    }
}

/// The parallel rotation is bit-identical to the single-threaded table
/// rotation, for thread counts that divide the tokens evenly and unevenly.
#[test]
fn parallel_rotation_is_bit_identical() {
    // B·S = 3·7 = 21 tokens: 2, 4, 5 and 8 threads give unequal blocks,
    // and some blocks start in the middle of a batch element.
    let shape = TensorShape::new(3, 7, 2, 32);
    let input = Tensor4::from_fn(shape, |b, s, h, d| {
        (b as f64 - s as f64) * 0.1 + (h * d) as f64
    })
    .unwrap();
    for rope in stage9_embeddings() {
        let table = rope
            .sin_cos_table(PositionRange::new(40, 20).unwrap())
            .unwrap();
        let mut expected = input.clone();
        rope.rotate_tensor_with_table(&mut expected, &table, pos(50))
            .unwrap();
        for n in [1, 2, 3, 4, 5, 7, 8, 21, 64] {
            let mut t = input.clone();
            let threads = NonZeroUsize::new(n).unwrap();
            rope.rotate_tensor_parallel(&mut t, &table, pos(50), threads)
                .unwrap();
            assert_eq!(t, expected, "{:?} threads = {n}", rope.config());
        }
    }
}

/// Each table entry satisfies sin² + cos² = 1 (within rounding), also at
/// large positions. Tolerance: `DEFAULT_TOLERANCE` (1e-12). Each of `sin`
/// and `cos` has an error of about 1 ulp (1.1e-16), so the sum of squares
/// has an error of a few times 1e-16.
#[test]
fn sin_cos_table_entries_lie_on_the_unit_circle() {
    let theta = inverse_frequencies(&config(64, 10_000.0));
    let table = SinCosTable::new(PositionRange::new(1_000_000, 64).unwrap(), &theta).unwrap();
    for m in table.positions().iter() {
        let (sin, cos) = (table.sin_row(m).unwrap(), table.cos_row(m).unwrap());
        for (j, (&s, &c)) in sin.iter().zip(cos).enumerate() {
            assert!(close(s * s + c * c, 1.0), "m = {}, j = {j}", m.value());
        }
    }
}

/// The relative-position property also holds for the table rotation:
/// the score of a query at m and a key at n depends only on m − n.
#[test]
fn table_rotation_keeps_the_relative_position_property() {
    let rope = RotaryEmbedding::new(config(16, 10_000.0));
    let table = rope
        .sin_cos_table(PositionRange::from_zero(2048).unwrap())
        .unwrap();
    let q0 = test_vector(16, 0.3);
    let k0 = test_vector(16, 1.7);
    let rotated_at = |v: &[f64], m: usize| {
        let mut t = Tensor4::new(TensorShape::new(1, 1, 1, 16), v.to_vec()).unwrap();
        rope.rotate_tensor_with_table(&mut t, &table, pos(m))
            .unwrap();
        t.into_vec()
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    // Distance 5 at three different absolute positions.
    let reference = dot(&rotated_at(&q0, 12), &rotated_at(&k0, 7));
    for (m, n) in [(105, 100), (1005, 1000), (2000, 1995)] {
        let score = dot(&rotated_at(&q0, m), &rotated_at(&k0, n));
        // Tolerance 1e-9: the angles m·θ are large (up to 2000 rad), and the
        // rounding error of φ grows with m (about m · 1.1e-16 rad).
        assert!(approx_eq(score, reference, 1e-9), "m = {m}, n = {n}");
    }
}
