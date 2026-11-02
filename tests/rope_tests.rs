//! Integration tests for the RoPE API.
//!
//! Integration tests use the library from the outside, as a user does.
//! Cargo compiles each file in `tests/` as a separate crate.
//!
//! ## Implemented (Stage 3)
//!
//! - `RopeConfig`: valid and invalid dimensions and bases.
//! - `inverse_frequency` / `inverse_frequencies`: lengths for `d = 8, 64, 128`.
//! - `Position` / `PositionRange`: position 0, position 1, consecutive
//!   positions, start offsets, limits.
//! - `angle` / `AngleTable`: shape, lookup, errors.
//!
//! ## Implemented (Stage 4)
//!
//! - `rotate_pair`: known values.
//! - `RotaryEmbedding`: `d = 2, 4, 8`, positions 0 and 1, several positions,
//!   several bases, invalid dimensions, zero vectors, negative values,
//!   `rotate` versus `rotate_in_place`, the worked example of Entry 10.
//!
//! The numerical properties (values, monotonicity, linearity, invariants)
//! are in `tests/numerical_tests.rs`.

use rust_rope::math::{DEFAULT_TOLERANCE, approx_eq};
use rust_rope::rope::config::{DEFAULT_BASE, RopeConfig};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::frequencies::{AngleTable, angle, inverse_frequencies, inverse_frequency};
use rust_rope::rope::position::{Position, PositionRange};
use rust_rope::rope::rotations::rotate_pair;
use rust_rope::utils::validation::RopeError;

/// Confirms that the library crate links into an external crate, and that
/// the public module paths exist.
#[test]
fn library_modules_are_public() {
    // `use ... as _` imports a path without a name. It only checks that the
    // path exists. It does not execute code.
    #[allow(unused_imports)]
    use rust_rope::{math as _, rope as _, utils as _};
}

// --- Configuration ---------------------------------------------------------

#[test]
fn configuration_for_common_dimensions() {
    for (d, pairs) in [(8, 4), (64, 32), (128, 64)] {
        let config = RopeConfig::with_default_base(d).unwrap();
        assert_eq!(config.dimension(), d);
        assert_eq!(config.num_pairs(), pairs);
        assert_eq!(config.base(), DEFAULT_BASE);
    }
}

#[test]
fn invalid_dimensions_are_errors() {
    assert_eq!(
        RopeConfig::with_default_base(0),
        Err(RopeError::ZeroDimension)
    );
    for d in [1, 7, 63, 129] {
        assert_eq!(
            RopeConfig::with_default_base(d),
            Err(RopeError::OddDimension { dimension: d })
        );
    }
}

#[test]
fn invalid_bases_are_errors() {
    for b in [1.0, 0.5, 0.0, -10_000.0, f64::INFINITY] {
        assert_eq!(
            RopeConfig::new(8, b),
            Err(RopeError::InvalidBase { base: b })
        );
    }
    assert!(matches!(
        RopeConfig::new(8, f64::NAN),
        Err(RopeError::InvalidBase { .. })
    ));
}

#[test]
fn errors_have_readable_messages() {
    let error = RopeConfig::new(7, 10_000.0).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("7") && message.contains("odd"),
        "{message}"
    );

    // `RopeError` implements `std::error::Error`, so `?` converts it into a
    // `Box<dyn Error>`.
    fn build() -> Result<RopeConfig, Box<dyn std::error::Error>> {
        Ok(RopeConfig::new(8, -1.0)?)
    }
    assert!(build().is_err());
}

// --- Inverse frequencies ---------------------------------------------------

#[test]
fn frequency_list_has_one_value_per_pair() {
    for d in [8, 64, 128] {
        let config = RopeConfig::with_default_base(d).unwrap();
        assert_eq!(inverse_frequencies(&config).len(), d / 2);
    }
}

#[test]
fn single_frequency_rejects_out_of_range_pair_index() {
    let config = RopeConfig::with_default_base(64).unwrap();
    assert!(inverse_frequency(&config, 31).is_ok());
    assert_eq!(
        inverse_frequency(&config, 32),
        Err(RopeError::PairIndexOutOfRange {
            pair_index: 32,
            num_pairs: 32
        })
    );
}

// --- Positions -------------------------------------------------------------

#[test]
fn position_zero_and_one() {
    assert_eq!(Position::ZERO.value(), 0);
    assert_eq!(Position::new(0), Ok(Position::ZERO));
    assert_eq!(Position::new(1).unwrap().value(), 1);
    assert_eq!(Position::new(1).unwrap().as_f64(), 1.0);
}

#[test]
fn consecutive_positions_from_zero_and_from_an_offset() {
    let from_zero: Vec<usize> = PositionRange::from_zero(4)
        .unwrap()
        .iter()
        .map(Position::value)
        .collect();
    assert_eq!(from_zero, [0, 1, 2, 3]);

    let from_offset: Vec<usize> = PositionRange::new(2048, 4)
        .unwrap()
        .iter()
        .map(Position::value)
        .collect();
    assert_eq!(from_offset, [2048, 2049, 2050, 2051]);
}

#[test]
fn range_reports_start_and_length() {
    let range = PositionRange::new(10, 5).unwrap();
    assert_eq!(range.start().value(), 10);
    assert_eq!(range.len(), 5);
    assert!(!range.is_empty());
    // The range is `Copy`: `iter()` does not consume the original value.
    assert_eq!(range.iter().count(), 5);
    assert_eq!(range.iter().count(), 5);
}

#[test]
fn too_long_sequence_is_an_error() {
    assert!(matches!(
        PositionRange::new(usize::MAX - 1, 10),
        Err(RopeError::PositionTooLarge { .. }) | Err(RopeError::SequenceTooLong { .. })
    ));
}

// --- Angles ----------------------------------------------------------------

#[test]
fn single_angle_validates_its_frequency() {
    let m = Position::new(5).unwrap();
    assert_eq!(angle(m, 0.25), Ok(1.25));
    assert_eq!(
        angle(m, -0.25),
        Err(RopeError::InvalidInverseFrequency {
            pair_index: None,
            value: -0.25
        })
    );
}

#[test]
fn angle_table_shape_for_common_dimensions() {
    for d in [8, 64, 128] {
        let theta = inverse_frequencies(&RopeConfig::with_default_base(d).unwrap());
        let table = AngleTable::new(PositionRange::from_zero(16).unwrap(), &theta).unwrap();
        assert_eq!(table.num_positions(), 16);
        assert_eq!(table.num_pairs(), d / 2);
        assert_eq!(table.as_slice().len(), 16 * d / 2);
        for m in table.positions().iter() {
            assert_eq!(table.row(m).map(<[f64]>::len), Some(d / 2));
        }
    }
}

#[test]
fn angle_table_lookup_with_a_start_offset() {
    let theta = [1.0, 0.5];
    let table = AngleTable::new(PositionRange::new(100, 2).unwrap(), &theta).unwrap();
    let p = |m| Position::new(m).unwrap();
    assert_eq!(table.row(p(100)), Some(&[100.0, 50.0][..]));
    assert_eq!(table.row(p(101)), Some(&[101.0, 50.5][..]));
    assert_eq!(table.row(p(0)), None);
    assert_eq!(table.row(p(102)), None);
}

#[test]
fn angle_table_rejects_invalid_frequency_lists() {
    let range = PositionRange::from_zero(2).unwrap();
    assert_eq!(
        AngleTable::new(range, &[]),
        Err(RopeError::EmptyFrequencies)
    );
    assert!(matches!(
        AngleTable::new(range, &[1.0, f64::NAN]),
        Err(RopeError::InvalidInverseFrequency {
            pair_index: Some(1),
            ..
        })
    ));
}

// --- Stage 4: the RoPE transformation --------------------------------------

fn rope(d: usize, b: f64) -> RotaryEmbedding {
    RotaryEmbedding::new(RopeConfig::new(d, b).expect("valid test configuration"))
}

fn pos(m: usize) -> Position {
    Position::new(m).expect("small test position")
}

fn all_close(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(&x, &y)| approx_eq(x, y, DEFAULT_TOLERANCE))
}

#[test]
fn rotate_pair_known_values() {
    // φ = 1: cos 1 = 0.5403023058681398, sin 1 = 0.8414709848078965.
    let [a, b] = rotate_pair([1.0, 2.0], 1.0);
    assert!(approx_eq(
        a,
        0.540_302_305_868_139_8 - 2.0 * 0.841_470_984_807_896_5,
        1e-15
    ));
    assert!(approx_eq(
        b,
        0.841_470_984_807_896_5 + 2.0 * 0.540_302_305_868_139_8,
        1e-15
    ));
}

#[test]
fn two_dimensional_embedding() {
    // d = 2: one pair, θ₀ = 1. RoPE([1, 0], m) = [cos m, sin m].
    let r = rope(2, DEFAULT_BASE);
    assert_eq!(r.inverse_frequencies(), &[1.0]);
    for m in 0..6 {
        let y = r.rotate(&[1.0, 0.0], pos(m)).unwrap();
        let (s, c) = (m as f64).sin_cos();
        assert!(all_close(&y, &[c, s]), "m = {m}: {y:?}");
    }
}

#[test]
fn four_dimensional_embedding() {
    // d = 4, b = 10000: θ = [1, 0.01].
    let r = rope(4, 10_000.0);
    let y = r.rotate(&[1.0, 0.0, 0.0, 1.0], pos(1)).unwrap();
    // pair 0: [1, 0] by 1 rad → [cos 1, sin 1]
    // pair 1: [0, 1] by 0.01 rad → [−sin 0.01, cos 0.01]
    let expected = [1f64.cos(), 1f64.sin(), -(0.01f64.sin()), 0.01f64.cos()];
    assert!(all_close(&y, &expected), "{y:?}");
}

#[test]
fn eight_dimensional_worked_example() {
    // Entry 10, section 3.9: x = [1, ..., 8], m = 1, b = 10000.
    let r = rope(8, 10_000.0);
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = r.rotate(&x, pos(1)).unwrap();
    // Reference values with 9 decimals from `cargo run --example basic_rope`.
    // The hand calculation in explain.md (Entry 10) agrees to 7 decimals.
    let expected = [
        -1.142_639_664,
        1.922_075_597,
        2.585_678_829,
        4.279_516_911,
        4.939_751_002,
        6.049_699_169,
        6.991_996_501,
        8.006_995_999,
    ];
    for (k, (&a, &e)) in y.iter().zip(&expected).enumerate() {
        // Tolerance 1e-9: the reference values have 9 decimals.
        assert!((a - e).abs() < 1e-9, "component {k}: {a} vs {e}");
    }
}

#[test]
fn position_zero_returns_an_exact_copy() {
    let x = [0.3, -1.7, 2.0, 0.0, -5.5, 1e-8, 1e8, -2.25];
    for b in [2.0, 10_000.0, 500_000.0] {
        assert_eq!(rope(8, b).rotate(&x, Position::ZERO).unwrap(), x);
    }
}

#[test]
fn position_one_rotates_pair_j_by_theta_j() {
    let r = rope(8, 10_000.0);
    let x = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let y = r.rotate(&x, pos(1)).unwrap();
    // `as_chunks::<2>()` gives the output pairs; `enumerate` gives j.
    for (j, pair) in y.as_chunks::<2>().0.iter().enumerate() {
        let theta = r.inverse_frequencies()[j];
        assert!(all_close(pair, &[theta.cos(), theta.sin()]), "pair {j}");
    }
}

#[test]
fn multiple_token_positions() {
    // The same vector at the positions 0..10: each result equals the
    // pair-by-pair rotation by m·θⱼ.
    let r = rope(8, 10_000.0);
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for m in PositionRange::from_zero(10).unwrap().iter() {
        let y = r.rotate(&x, m).unwrap();
        for (j, &theta) in r.inverse_frequencies().iter().enumerate() {
            let expected = rotate_pair([x[2 * j], x[2 * j + 1]], m.as_f64() * theta);
            assert_eq!(
                &y[2 * j..2 * j + 2],
                &expected,
                "m = {}, j = {j}",
                m.value()
            );
        }
    }
}

#[test]
fn different_bases() {
    // Pair 0 has θ₀ = 1 for every base: the same result. The other pairs
    // rotate more slowly with a larger base.
    let x = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let small = rope(8, 100.0).rotate(&x, pos(5)).unwrap();
    let large = rope(8, 1_000_000.0).rotate(&x, pos(5)).unwrap();
    assert_eq!(&small[0..2], &large[0..2]);
    for j in 1..4 {
        // The angle of [cos φ, sin φ] is φ; a smaller angle has a larger cos.
        assert!(large[2 * j] > small[2 * j], "pair {j}");
    }
}

#[test]
fn invalid_dimensions() {
    // Invalid configurations cannot be created.
    assert_eq!(
        RopeConfig::new(0, DEFAULT_BASE),
        Err(RopeError::ZeroDimension)
    );
    assert_eq!(
        RopeConfig::new(5, DEFAULT_BASE),
        Err(RopeError::OddDimension { dimension: 5 })
    );
    // An embedding with the wrong length is rejected.
    let r = rope(8, DEFAULT_BASE);
    for len in [0, 1, 2, 7, 9, 16] {
        let x = vec![1.0; len];
        assert_eq!(
            r.rotate(&x, pos(1)),
            Err(RopeError::EmbeddingDimensionMismatch {
                expected: 8,
                actual: len
            })
        );
    }
}

#[test]
fn zero_vector_stays_zero() {
    let r = rope(8, DEFAULT_BASE);
    for m in [0, 1, 7, 100_000] {
        let y = r.rotate(&[0.0; 8], pos(m)).unwrap();
        assert!(y.iter().all(|&v| v == 0.0), "m = {m}: {y:?}");
    }
}

#[test]
fn negative_values() {
    // RoPE is linear: RoPE(−x) = −RoPE(x). Exact, because negation is exact.
    let r = rope(8, DEFAULT_BASE);
    let x = [1.0, -2.0, 3.0, -4.0, -5.0, 6.0, -7.0, 8.0];
    let neg: Vec<f64> = x.iter().map(|v| -v).collect();
    let y = r.rotate(&x, pos(3)).unwrap();
    let y_neg = r.rotate(&neg, pos(3)).unwrap();
    for (a, b) in y.iter().zip(&y_neg) {
        assert_eq!(*a, -*b);
    }
}

#[test]
fn rotate_and_rotate_in_place_agree() {
    let r = rope(64, DEFAULT_BASE);
    let x: Vec<f64> = (0..64).map(|k| (k as f64 * 0.37).sin()).collect();
    let new_vector = r.rotate(&x, pos(42)).unwrap();
    let mut in_place = x.clone();
    r.rotate_in_place(&mut in_place, pos(42)).unwrap();
    assert_eq!(new_vector, in_place);
    assert_ne!(new_vector, x, "the input of `rotate` is unchanged");
}

#[test]
fn rotate_in_place_works_on_a_sub_slice() {
    // A `&mut [f64]` can be a part of a larger buffer, for example one
    // row of a matrix of embeddings. Only that part changes.
    let r = rope(4, DEFAULT_BASE);
    let mut buffer = [9.0, 1.0, 0.0, 1.0, 0.0, 9.0];
    r.rotate_in_place(&mut buffer[1..5], pos(1)).unwrap();
    assert_eq!(buffer[0], 9.0);
    assert_eq!(buffer[5], 9.0);
    assert!(approx_eq(buffer[1], 1f64.cos(), DEFAULT_TOLERANCE));
}

// --- Stage 5: query and key tensors [B, S, H, D] ---------------------------

use rust_rope::math::tensor::{Tensor4, TensorError, TensorShape};

/// Deterministic tensor values: each (b, s, h, d) gets a different number.
fn tensor(shape: TensorShape, seed: f64) -> Tensor4 {
    Tensor4::from_fn(shape, |b, s, h, d| {
        ((b * 1000 + s * 100 + h * 10 + d) as f64 * 0.37 + seed).sin()
    })
    .unwrap()
}

/// Checks every head vector of `out` against a single-vector RoPE of the
/// matching input vector at position `start + s`.
fn check_against_single_vector(r: &RotaryEmbedding, input: &Tensor4, out: &Tensor4, start: usize) {
    let shape = input.shape();
    for b in 0..shape.batch {
        for s in 0..shape.sequence {
            for h in 0..shape.heads {
                let expected = r
                    .rotate(input.vector(b, s, h).unwrap(), pos(start + s))
                    .unwrap();
                assert_eq!(
                    out.vector(b, s, h).unwrap(),
                    expected.as_slice(),
                    "b={b} s={s} h={h}"
                );
            }
        }
    }
}

#[test]
fn tensor_shapes_batch_sequence_heads_and_head_dimensions() {
    for d in [2, 4, 8] {
        let r = rope(d, DEFAULT_BASE);
        for (batch, seq, heads) in [(1, 1, 1), (1, 4, 2), (2, 4, 2), (2, 1, 3), (3, 5, 4)] {
            let shape = TensorShape::new(batch, seq, heads, d);
            let (q, k) = (tensor(shape, 0.1), tensor(shape, 2.0));
            let (q2, k2) = r.rotate_query_key(&q, &k, Position::ZERO).unwrap();
            assert_eq!(q2.shape(), shape);
            check_against_single_vector(&r, &q, &q2, 0);
            check_against_single_vector(&r, &k, &k2, 0);
        }
    }
}

#[test]
fn the_stage_5_example_shape() {
    // B = 2, S = 4, H = 2, D = 4: hidden dimension H·D = 8.
    let r = rope(4, DEFAULT_BASE);
    let shape = TensorShape::new(2, 4, 2, 4);
    let (q, k) = (tensor(shape, 0.0), tensor(shape, 1.0));
    assert_eq!(q.as_slice().len(), 2 * 4 * 2 * 4);
    let (q2, k2) = r.rotate_query_key(&q, &k, Position::ZERO).unwrap();
    check_against_single_vector(&r, &q, &q2, 0);
    check_against_single_vector(&r, &k, &k2, 0);
}

#[test]
fn different_starting_positions() {
    let r = rope(8, DEFAULT_BASE);
    let shape = TensorShape::new(2, 3, 2, 8);
    let (q, k) = (tensor(shape, 0.5), tensor(shape, 1.5));
    for start in [0, 1, 7, 1000, 123_456] {
        let (q2, k2) = r.rotate_query_key(&q, &k, pos(start)).unwrap();
        check_against_single_vector(&r, &q, &q2, start);
        check_against_single_vector(&r, &k, &k2, start);
    }
}

#[test]
fn in_place_and_copying_versions_agree() {
    let r = rope(8, DEFAULT_BASE);
    let shape = TensorShape::new(2, 4, 3, 8);
    let (q, k) = (tensor(shape, 0.2), tensor(shape, 0.9));
    let (q2, k2) = r.rotate_query_key(&q, &k, pos(5)).unwrap();
    let (mut q3, mut k3) = (q.clone(), k.clone());
    r.rotate_query_key_in_place(&mut q3, &mut k3, pos(5))
        .unwrap();
    // Compare references: a tuple of values would move `q3` and `k3`.
    assert_eq!((&q2, &k2), (&q3, &k3));
    let mut single = q.clone();
    r.rotate_tensor_in_place(&mut single, pos(5)).unwrap();
    assert_eq!(single, q3);
}

#[test]
fn query_and_key_may_have_different_head_counts() {
    // Grouped-query attention uses fewer key heads than query heads.
    let r = rope(4, DEFAULT_BASE);
    let q = tensor(TensorShape::new(1, 3, 4, 4), 0.0);
    let k = tensor(TensorShape::new(1, 3, 2, 4), 1.0);
    let (q2, k2) = r.rotate_query_key(&q, &k, pos(2)).unwrap();
    check_against_single_vector(&r, &q, &q2, 2);
    check_against_single_vector(&r, &k, &k2, 2);
}

#[test]
fn invalid_tensor_shapes() {
    // The tensor type itself rejects inconsistent data and zero extents.
    assert_eq!(
        Tensor4::new(TensorShape::new(2, 4, 2, 4), vec![0.0; 63]),
        Err(TensorError::LengthMismatch {
            expected: 64,
            actual: 63
        })
    );
    assert!(matches!(
        Tensor4::zeros(TensorShape::new(0, 4, 2, 4)),
        Err(TensorError::ZeroExtent { axis: "batch" })
    ));

    let r = rope(4, DEFAULT_BASE);
    let q = tensor(TensorShape::new(2, 4, 2, 4), 0.0);
    // Different batch size.
    let k = tensor(TensorShape::new(1, 4, 2, 4), 0.0);
    assert_eq!(
        r.rotate_query_key(&q, &k, Position::ZERO),
        Err(RopeError::QueryKeyShapeMismatch {
            query: (2, 4),
            key: (1, 4)
        })
    );
    // Different sequence length.
    let k = tensor(TensorShape::new(2, 3, 2, 4), 0.0);
    assert!(matches!(
        r.rotate_query_key(&q, &k, Position::ZERO),
        Err(RopeError::QueryKeyShapeMismatch { .. })
    ));
}

#[test]
fn invalid_head_dimensions() {
    let r = rope(8, DEFAULT_BASE);
    let mut t = tensor(TensorShape::new(1, 2, 2, 6), 0.0);
    let before = t.clone();
    assert_eq!(
        r.rotate_tensor_in_place(&mut t, Position::ZERO),
        Err(RopeError::EmbeddingDimensionMismatch {
            expected: 8,
            actual: 6
        })
    );
    assert_eq!(t, before, "an error leaves the tensor unchanged");
    // Query valid, key invalid: neither changes.
    let mut q = tensor(TensorShape::new(1, 2, 2, 8), 0.0);
    let mut k = tensor(TensorShape::new(1, 2, 2, 4), 0.0);
    let (q0, k0) = (q.clone(), k.clone());
    assert!(r.rotate_query_key_in_place(&mut q, &mut k, pos(3)).is_err());
    assert_eq!((q, k), (q0, k0));
}

#[test]
fn value_tensor_is_preserved() {
    // The RoPE API has no value parameter. A value tensor that goes through
    // the same "attention preparation" step stays bit-for-bit unchanged.
    let r = rope(4, DEFAULT_BASE);
    let shape = TensorShape::new(2, 4, 2, 4);
    let (q, k, v) = (tensor(shape, 0.0), tensor(shape, 1.0), tensor(shape, 2.0));
    let v_before = v.clone();
    let (q2, k2) = r.rotate_query_key(&q, &k, pos(1)).unwrap();
    assert_ne!(q2, q);
    assert_ne!(k2, k);
    assert_eq!(v, v_before);
}

#[test]
fn batch_isolation() {
    // Change batch element 1 of the input. Batch element 0 of the output
    // must not change.
    let r = rope(8, DEFAULT_BASE);
    let shape = TensorShape::new(2, 3, 2, 8);
    let mut t1 = tensor(shape, 0.0);
    let mut t2 = t1.clone();
    for s in 0..3 {
        for h in 0..2 {
            t2.vector_mut(1, s, h).unwrap().fill(99.0);
        }
    }
    r.rotate_tensor_in_place(&mut t1, pos(4)).unwrap();
    r.rotate_tensor_in_place(&mut t2, pos(4)).unwrap();
    for s in 0..3 {
        for h in 0..2 {
            assert_eq!(t1.vector(0, s, h), t2.vector(0, s, h));
            assert_ne!(t1.vector(1, s, h), t2.vector(1, s, h));
        }
    }
}

#[test]
fn head_isolation() {
    // Change head 0 of the input. Head 1 of the output must not change.
    let r = rope(4, DEFAULT_BASE);
    let shape = TensorShape::new(2, 3, 2, 4);
    let mut t1 = tensor(shape, 0.0);
    let mut t2 = t1.clone();
    for b in 0..2 {
        for s in 0..3 {
            t2.vector_mut(b, s, 0).unwrap().fill(-5.0);
        }
    }
    r.rotate_tensor_in_place(&mut t1, pos(0)).unwrap();
    r.rotate_tensor_in_place(&mut t2, pos(0)).unwrap();
    for b in 0..2 {
        for s in 0..3 {
            assert_eq!(t1.vector(b, s, 1), t2.vector(b, s, 1));
        }
    }
}

#[test]
fn sequence_order_gives_increasing_positions() {
    // The same vector at every sequence index: output s must equal the
    // single-vector RoPE at position start + s, and outputs must differ.
    let r = rope(2, DEFAULT_BASE);
    let mut t = Tensor4::from_fn(
        TensorShape::new(1, 4, 1, 2),
        |_, _, _, d| {
            if d == 0 { 1.0 } else { 0.0 }
        },
    )
    .unwrap();
    r.rotate_tensor_in_place(&mut t, pos(10)).unwrap();
    for s in 0..4 {
        let m = (10 + s) as f64; // θ₀ = 1, so the angle is m
        let v = t.vector(0, s, 0).unwrap();
        assert!(approx_eq(v[0], m.cos(), DEFAULT_TOLERANCE));
        assert!(approx_eq(v[1], m.sin(), DEFAULT_TOLERANCE));
    }
}

// --- Stage 7: layouts and partial RoPE --------------------------------------

use rust_rope::rope::config::RopeLayout;

fn rope_with(rotary: usize, head: usize, layout: RopeLayout) -> RotaryEmbedding {
    RotaryEmbedding::new(
        RopeConfig::new(rotary, DEFAULT_BASE)
            .unwrap()
            .with_layout(layout)
            .with_head_dimension(head)
            .unwrap(),
    )
}

#[test]
fn interleaved_layout_pairs_adjacent_components() {
    // Basis vector e_k: only one component is 1. With the interleaved layout,
    // e_0 rotates into components 0 and 1 only.
    let r = rope_with(6, 6, RopeLayout::Interleaved);
    for j in 0..3 {
        let mut e = [0.0; 6];
        e[2 * j] = 1.0;
        let y = r.rotate(&e, pos(1)).unwrap();
        let theta = r.inverse_frequencies()[j];
        for (k, &v) in y.iter().enumerate() {
            let expected = if k == 2 * j {
                theta.cos()
            } else if k == 2 * j + 1 {
                theta.sin()
            } else {
                0.0
            };
            assert!(
                approx_eq(v, expected, DEFAULT_TOLERANCE),
                "j={j} k={k}: {v}"
            );
        }
    }
}

#[test]
fn split_half_layout_pairs_component_j_with_j_plus_half() {
    // x = [x0, x1, x2 | x3, x4, x5]: pairs (x0,x3), (x1,x4), (x2,x5).
    let r = rope_with(6, 6, RopeLayout::SplitHalf);
    for j in 0..3 {
        let mut e = [0.0; 6];
        e[j] = 1.0;
        let y = r.rotate(&e, pos(1)).unwrap();
        let theta = r.inverse_frequencies()[j];
        for (k, &v) in y.iter().enumerate() {
            let expected = if k == j {
                theta.cos()
            } else if k == j + 3 {
                theta.sin()
            } else {
                0.0
            };
            assert!(
                approx_eq(v, expected, DEFAULT_TOLERANCE),
                "j={j} k={k}: {v}"
            );
        }
    }
}

#[test]
fn split_half_numerical_example() {
    // explain.md, Entry 14: x = [1, 2, 3, 4, 5, 6], d = 6, m = 1, b = 10000.
    // θ = [1, 10000^(−1/3), 10000^(−2/3)] = [1, 0.0464159, 0.0021544].
    let r = rope_with(6, 6, RopeLayout::SplitHalf);
    let y = r.rotate(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], pos(1)).unwrap();
    let theta = r.inverse_frequencies();
    for j in 0..3 {
        let (a, b) = ((j + 1) as f64, (j + 4) as f64); // (x_j, x_(j+3))
        let (s, c) = theta[j].sin_cos();
        assert!(approx_eq(y[j], a * c - b * s, DEFAULT_TOLERANCE));
        assert!(approx_eq(y[j + 3], a * s + b * c, DEFAULT_TOLERANCE));
    }
    // Reference values with 9 decimals, calculated independently with Python `math`.
    let expected = [
        -2.825_581_633,
        1.765_849_835,
        2.987_066_439,
        3.002_680_208,
        5.087_413_327,
        6.006_449_374,
    ];
    for (k, (&a, &e)) in y.iter().zip(&expected).enumerate() {
        assert!((a - e).abs() < 1e-9, "component {k}: {a} vs {e}");
    }
}

#[test]
fn both_layouts_keep_each_pair_length() {
    let x = [0.5, -1.5, 2.0, 0.25, -3.0, 1.0, 0.75, -0.5];
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        let r = rope_with(8, 8, layout);
        let y = r.rotate(&x, pos(321)).unwrap();
        let pairs: Vec<(usize, usize)> = match layout {
            RopeLayout::Interleaved => (0..4).map(|j| (2 * j, 2 * j + 1)).collect(),
            RopeLayout::SplitHalf => (0..4).map(|j| (j, j + 4)).collect(),
        };
        for (a, b) in pairs {
            let before = x[a] * x[a] + x[b] * x[b];
            let after = y[a] * y[a] + y[b] * y[b];
            assert!(
                approx_eq(before, after, DEFAULT_TOLERANCE),
                "{layout:?} ({a},{b})"
            );
        }
    }
}

#[test]
fn layouts_are_equivalent_after_a_permutation() {
    // Split-half pair j = (x_j, x_(j+d/2)). Move x_j to slot 2j and x_(j+d/2)
    // to slot 2j+1; then the interleaved layout sees the same pairs.
    let d = 8;
    let perm: Vec<usize> = (0..d / 2).flat_map(|j| [j, j + d / 2]).collect(); // [0,4,1,5,2,6,3,7]
    let x = [0.3, -1.2, 2.2, 0.9, -0.4, 1.7, -2.5, 0.6];
    let permuted: Vec<f64> = perm.iter().map(|&k| x[k]).collect();
    let split = rope_with(d, d, RopeLayout::SplitHalf)
        .rotate(&x, pos(9))
        .unwrap();
    let inter = rope_with(d, d, RopeLayout::Interleaved)
        .rotate(&permuted, pos(9))
        .unwrap();
    for (slot, &k) in perm.iter().enumerate() {
        assert_eq!(split[k], inter[slot], "slot {slot}");
    }
    // Without the permutation, the outputs differ.
    let wrong = rope_with(d, d, RopeLayout::Interleaved)
        .rotate(&x, pos(9))
        .unwrap();
    assert_ne!(split, wrong);
}

#[test]
fn partial_rope_rotates_only_the_rotary_part() {
    let x: Vec<f64> = (0..12).map(|k| k as f64 - 5.5).collect();
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        let partial = rope_with(4, 12, layout);
        let full4 = rope_with(4, 4, layout);
        let y = partial.rotate(&x, pos(3)).unwrap();
        // The first 4 values equal RoPE of x[..4] with d = 4.
        assert_eq!(&y[..4], full4.rotate(&x[..4], pos(3)).unwrap().as_slice());
        // The remaining 8 values are bit-for-bit unchanged.
        assert_eq!(&y[4..], &x[4..], "{layout:?}");
    }
}

#[test]
fn partial_rope_128_with_64() {
    let x: Vec<f64> = (0..128).map(|k| (k as f64 * 0.17).sin()).collect();
    let r = rope_with(64, 128, RopeLayout::SplitHalf);
    assert_eq!(r.inverse_frequencies().len(), 32);
    assert_eq!(r.dimension(), 128);
    assert_eq!(r.rotary_dimension(), 64);
    let y = r.rotate(&x, pos(77)).unwrap();
    assert_eq!(&y[64..], &x[64..]);
    assert_ne!(&y[..64], &x[..64]);
    // The frequencies use the rotary dimension: θ₁ = 10000^(−2/64).
    assert!(approx_eq(
        r.inverse_frequencies()[1],
        10_000f64.powf(-2.0 / 64.0),
        1e-15
    ));
}

#[test]
fn rotary_dimension_equal_to_head_dimension_is_full_rope() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let explicit = rope_with(8, 8, RopeLayout::Interleaved)
        .rotate(&x, pos(1))
        .unwrap();
    let default = rope(8, DEFAULT_BASE).rotate(&x, pos(1)).unwrap();
    assert_eq!(explicit, default);
}

#[test]
fn invalid_rotary_dimensions() {
    // Zero or odd rotary dimension.
    assert_eq!(
        RopeConfig::new(0, DEFAULT_BASE),
        Err(RopeError::ZeroDimension)
    );
    assert_eq!(
        RopeConfig::new(63, DEFAULT_BASE),
        Err(RopeError::OddDimension { dimension: 63 })
    );
    // Rotary dimension larger than the head dimension.
    assert_eq!(
        RopeConfig::new(64, DEFAULT_BASE)
            .unwrap()
            .with_head_dimension(32),
        Err(RopeError::RotaryDimensionTooLarge {
            rotary_dimension: 64,
            head_dimension: 32
        })
    );
    // With partial RoPE, inputs must have the head dimension, not the rotary one.
    let r = rope_with(4, 8, RopeLayout::Interleaved);
    assert_eq!(
        r.rotate(&[1.0; 4], pos(1)),
        Err(RopeError::EmbeddingDimensionMismatch {
            expected: 8,
            actual: 4
        })
    );
}

#[test]
fn tensors_with_layouts_and_partial_rope() {
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        for (rotary, head) in [(4, 4), (4, 8), (8, 12), (2, 6)] {
            let r = rope_with(rotary, head, layout);
            let shape = TensorShape::new(2, 5, 3, head);
            let (q, k) = (tensor(shape, 0.4), tensor(shape, -0.8));
            let (q2, k2) = r.rotate_query_key(&q, &k, pos(11)).unwrap();
            check_against_single_vector(&r, &q, &q2, 11);
            check_against_single_vector(&r, &k, &k2, 11);
            for b in 0..2 {
                for s in 0..5 {
                    for h in 0..3 {
                        let (vin, vout) = (q.vector(b, s, h).unwrap(), q2.vector(b, s, h).unwrap());
                        assert_eq!(
                            &vout[rotary..],
                            &vin[rotary..],
                            "{layout:?} r={rotary} D={head}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn tensor_head_dimension_must_match_the_head_dimension_of_the_config() {
    let r = rope_with(4, 8, RopeLayout::SplitHalf);
    let mut t = tensor(TensorShape::new(1, 2, 2, 4), 0.0);
    assert_eq!(
        r.rotate_tensor_in_place(&mut t, pos(0)),
        Err(RopeError::EmbeddingDimensionMismatch {
            expected: 8,
            actual: 4
        })
    );
}

// --- Stage 8: scaling methods -----------------------------------------------

use rust_rope::rope::scaling::{RopeScaling, scaled_inverse_frequencies};

fn all_methods() -> [RopeScaling; 4] {
    [
        RopeScaling::PositionInterpolation {
            original_context: 2048,
            target_context: 8192,
        },
        RopeScaling::Linear { factor: 4.0 },
        RopeScaling::NtkAware { factor: 4.0 },
        RopeScaling::yarn(4.0, 2048),
    ]
}

#[test]
fn scaling_none_gives_bit_identical_standard_rope() {
    let config = RopeConfig::new(64, DEFAULT_BASE).unwrap();
    let standard = RotaryEmbedding::new(config);
    let none = RotaryEmbedding::with_scaling(config, RopeScaling::None).unwrap();
    assert_eq!(standard, none);
    let x: Vec<f64> = (0..64).map(|k| (k as f64 * 0.3).cos()).collect();
    assert_eq!(
        standard.rotate(&x, pos(777)).unwrap(),
        none.rotate(&x, pos(777)).unwrap()
    );
}

#[test]
fn scaling_factor_one_gives_standard_rope() {
    let config = RopeConfig::new(32, DEFAULT_BASE).unwrap();
    let standard = RotaryEmbedding::new(config);
    let x: Vec<f64> = (0..32).map(|k| (k as f64 * 0.7).sin()).collect();
    let expected = standard.rotate(&x, pos(5000)).unwrap();
    for s in [
        RopeScaling::Linear { factor: 1.0 },
        RopeScaling::NtkAware { factor: 1.0 },
        RopeScaling::yarn(1.0, 2048),
        RopeScaling::PositionInterpolation {
            original_context: 2048,
            target_context: 2048,
        },
    ] {
        let y = RotaryEmbedding::with_scaling(config, s)
            .unwrap()
            .rotate(&x, pos(5000))
            .unwrap();
        assert!(all_close(&y, &expected), "{s:?}");
    }
}

#[test]
fn position_interpolation_produces_the_expected_angles() {
    // L = 2048, L' = 8192, m = 4096: m' = 4096 · 2048/8192 = 1024.
    let config = RopeConfig::new(8, DEFAULT_BASE).unwrap();
    let pi = RotaryEmbedding::with_scaling(
        config,
        RopeScaling::PositionInterpolation {
            original_context: 2048,
            target_context: 8192,
        },
    )
    .unwrap();
    assert_eq!(pi.inverse_frequencies(), &[0.25, 0.025, 0.0025, 0.00025]);
    // Basis vector in pair j: its phase is the angle m·θ'ⱼ = 1024·θⱼ.
    for j in 0..4 {
        let mut e = [0.0; 8];
        e[2 * j] = 1.0;
        let y = pi.rotate(&e, pos(4096)).unwrap();
        let angle = 4096.0 * pi.inverse_frequencies()[j];
        let expected = 1024.0 * [1.0, 0.1, 0.01, 0.001][j];
        assert!(approx_eq(angle, expected, 1e-12));
        assert!(approx_eq(y[2 * j], angle.cos(), DEFAULT_TOLERANCE));
        assert!(approx_eq(y[2 * j + 1], angle.sin(), DEFAULT_TOLERANCE));
    }
}

#[test]
fn longer_targets_give_slower_frequencies() {
    let config = RopeConfig::new(64, DEFAULT_BASE).unwrap();
    let freq = |s: RopeScaling| scaled_inverse_frequencies(&config, &s).unwrap();
    for j in 1..32 {
        let mut previous = f64::INFINITY;
        for target in [2048, 4096, 8192, 32_768] {
            let t = freq(RopeScaling::PositionInterpolation {
                original_context: 2048,
                target_context: target,
            })[j];
            assert!(t < previous, "j = {j}, target = {target}");
            previous = t;
        }
        let mut previous = f64::INFINITY;
        for factor in [1.0, 2.0, 4.0, 16.0] {
            let t = freq(RopeScaling::NtkAware { factor })[j];
            assert!(t < previous, "NTK j = {j}, factor = {factor}");
            previous = t;
        }
    }
}

#[test]
fn different_scaling_methods_are_distinguishable() {
    let config = RopeConfig::new(64, DEFAULT_BASE).unwrap();
    let x: Vec<f64> = (0..64).map(|k| (k as f64 * 0.21).sin()).collect();
    let outputs: Vec<Vec<f64>> = [
        RopeScaling::None,
        RopeScaling::Linear { factor: 4.0 },
        RopeScaling::NtkAware { factor: 4.0 },
        RopeScaling::yarn(4.0, 2048),
    ]
    .iter()
    .map(|&s| {
        RotaryEmbedding::with_scaling(config, s)
            .unwrap()
            .rotate(&x, pos(3000))
            .unwrap()
    })
    .collect();
    for a in 0..outputs.len() {
        for b in a + 1..outputs.len() {
            assert!(!all_close(&outputs[a], &outputs[b]), "methods {a} and {b}");
        }
    }
}

#[test]
fn invalid_scaling_configurations_return_useful_errors() {
    let config = RopeConfig::new(8, DEFAULT_BASE).unwrap();
    let error =
        RotaryEmbedding::with_scaling(config, RopeScaling::Linear { factor: 0.5 }).unwrap_err();
    assert_eq!(error, RopeError::InvalidScalingFactor { factor: 0.5 });
    assert!(error.to_string().contains("0.5"));
    let error = RotaryEmbedding::with_scaling(
        config,
        RopeScaling::PositionInterpolation {
            original_context: 8192,
            target_context: 2048,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("8192"), "{error}");
    assert!(RotaryEmbedding::with_scaling(config, RopeScaling::yarn(f64::NAN, 2048)).is_err());
    assert!(
        RotaryEmbedding::with_scaling(
            RopeConfig::new(2, DEFAULT_BASE).unwrap(),
            RopeScaling::NtkAware { factor: 2.0 }
        )
        .is_err()
    );
}

#[test]
fn scaling_works_with_both_layouts_partial_rope_and_tensors() {
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        let config = RopeConfig::new(8, DEFAULT_BASE)
            .unwrap()
            .with_layout(layout)
            .with_head_dimension(12)
            .unwrap();
        for scaling in all_methods() {
            let r = RotaryEmbedding::with_scaling(config, scaling).unwrap();
            let shape = TensorShape::new(2, 4, 3, 12);
            let (q, k) = (tensor(shape, 0.1), tensor(shape, 0.9));
            let (q2, k2) = r.rotate_query_key(&q, &k, pos(5000)).unwrap();
            assert_eq!(q2.shape(), shape);
            check_against_single_vector(&r, &q, &q2, 5000);
            check_against_single_vector(&r, &k, &k2, 5000);
            for b in 0..2 {
                for s in 0..4 {
                    for h in 0..3 {
                        // Pass-through components are never scaled or rotated.
                        assert_eq!(
                            &q2.vector(b, s, h).unwrap()[8..],
                            &q.vector(b, s, h).unwrap()[8..]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn yarn_attention_factor_scales_only_the_rotary_part() {
    let config = RopeConfig::new(4, DEFAULT_BASE)
        .unwrap()
        .with_head_dimension(6)
        .unwrap();
    let r = RotaryEmbedding::with_scaling(config, RopeScaling::yarn(4.0, 2048)).unwrap();
    let factor = 0.1 * 4f64.ln() + 1.0;
    assert!(approx_eq(r.attention_factor(), factor, DEFAULT_TOLERANCE));
    let x = [3.0, 4.0, 1.0, 0.0, 7.0, -7.0];
    let y = r.rotate(&x, Position::ZERO).unwrap(); // angle 0: only the factor acts
    assert!(approx_eq(y[0], 3.0 * factor, DEFAULT_TOLERANCE));
    assert!(approx_eq(y[1], 4.0 * factor, DEFAULT_TOLERANCE));
    assert_eq!(&y[4..], &[7.0, -7.0]);
}

// ===========================================================================
// Stage 9: sin/cos table, table rotation, multithreaded rotation (API)
// ===========================================================================

use rust_rope::rope::frequencies::SinCosTable;
use std::num::NonZeroUsize;

fn threads(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

#[test]
fn sin_cos_table_shape_and_lookup() {
    let theta = inverse_frequencies(&RopeConfig::new(8, 10_000.0).unwrap());
    let table = SinCosTable::new(PositionRange::new(10, 5).unwrap(), &theta).unwrap();
    assert_eq!(table.num_positions(), 5);
    assert_eq!(table.num_pairs(), 4);
    assert_eq!(table.positions().start(), pos(10));
    // 2 arrays · 5 rows · 4 pairs · 8 bytes.
    assert_eq!(table.size_in_bytes(), 2 * 5 * 4 * 8);
    // Positions outside 10..15 are not in the table.
    assert_eq!(table.sin_row(pos(9)), None);
    assert_eq!(table.cos_row(pos(15)), None);
    let (s, c) = (12.0 * theta[1]).sin_cos();
    assert_eq!(table.sin_row(pos(12)).unwrap()[1], s);
    assert_eq!(table.cos_row(pos(12)).unwrap()[1], c);
}

#[test]
fn sin_cos_table_rejects_invalid_frequencies() {
    let range = PositionRange::from_zero(4).unwrap();
    assert_eq!(
        SinCosTable::new(range, &[]),
        Err(RopeError::EmptyFrequencies)
    );
    assert!(SinCosTable::new(range, &[1.0, f64::NAN]).is_err());
    assert!(SinCosTable::new(range, &[1.0, -0.5]).is_err());
}

#[test]
fn sin_cos_table_from_the_embedding_uses_its_frequencies() {
    let rope = RotaryEmbedding::new(RopeConfig::new(16, 500_000.0).unwrap());
    let range = PositionRange::from_zero(32).unwrap();
    let from_rope = rope.sin_cos_table(range).unwrap();
    let direct = SinCosTable::new(range, rope.inverse_frequencies()).unwrap();
    assert_eq!(from_rope, direct);
}

#[test]
fn table_rotation_rejects_a_table_with_the_wrong_number_of_pairs() {
    let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0).unwrap());
    let other = RotaryEmbedding::new(RopeConfig::new(4, 10_000.0).unwrap());
    let table = other
        .sin_cos_table(PositionRange::from_zero(8).unwrap())
        .unwrap();
    let mut t =
        Tensor4::from_fn(TensorShape::new(1, 4, 1, 8), |_, s, _, d| (s + d) as f64).unwrap();
    let before = t.clone();
    assert_eq!(
        rope.rotate_tensor_with_table(&mut t, &table, Position::ZERO),
        Err(RopeError::TablePairCountMismatch {
            expected: 4,
            actual: 2
        })
    );
    assert_eq!(t, before);
}

#[test]
fn table_rotation_rejects_positions_outside_the_table() {
    let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0).unwrap());
    // The table contains the positions 100..110.
    let table = rope
        .sin_cos_table(PositionRange::new(100, 10).unwrap())
        .unwrap();
    let mut t = Tensor4::zeros(TensorShape::new(2, 4, 3, 8)).unwrap();

    // 100..104 and 106..110 are inside the table.
    assert!(
        rope.rotate_tensor_with_table(&mut t, &table, pos(100))
            .is_ok()
    );
    assert!(
        rope.rotate_tensor_with_table(&mut t, &table, pos(106))
            .is_ok()
    );
    // 99..103 starts before the table; 107..111 ends after it.
    for start in [99, 107, 0, 1_000] {
        assert_eq!(
            rope.rotate_tensor_with_table(&mut t, &table, pos(start)),
            Err(RopeError::TableDoesNotCoverPositions {
                table_start: 100,
                table_length: 10,
                start,
                length: 4
            }),
            "start = {start}"
        );
    }
    // The parallel function uses the same checks.
    assert!(
        rope.rotate_tensor_parallel(&mut t, &table, pos(107), threads(2))
            .is_err()
    );
}

#[test]
fn table_rotation_rejects_a_wrong_head_dimension() {
    let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0).unwrap());
    let table = rope
        .sin_cos_table(PositionRange::from_zero(4).unwrap())
        .unwrap();
    let mut t = Tensor4::zeros(TensorShape::new(1, 4, 1, 6)).unwrap();
    assert_eq!(
        rope.rotate_tensor_with_table(&mut t, &table, Position::ZERO),
        Err(RopeError::EmbeddingDimensionMismatch {
            expected: 8,
            actual: 6
        })
    );
}

#[test]
fn query_key_with_table_checks_both_tensors_first() {
    let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0).unwrap());
    let table = rope
        .sin_cos_table(PositionRange::from_zero(4).unwrap())
        .unwrap();
    let mut q = Tensor4::from_fn(TensorShape::new(1, 4, 2, 8), |_, s, h, d| {
        (s + h + d) as f64
    })
    .unwrap();
    let mut k = Tensor4::zeros(TensorShape::new(1, 3, 2, 8)).unwrap(); // S differs
    let q_before = q.clone();
    assert!(matches!(
        rope.rotate_query_key_with_table(&mut q, &mut k, &table, Position::ZERO),
        Err(RopeError::QueryKeyShapeMismatch { .. })
    ));
    assert_eq!(q, q_before);
}

#[test]
fn query_key_with_table_equals_the_stage_5_function() {
    let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0).unwrap());
    let table = rope
        .sin_cos_table(PositionRange::from_zero(64).unwrap())
        .unwrap();
    // Grouped-query attention: 4 query heads, 2 key heads.
    let q = Tensor4::from_fn(TensorShape::new(2, 5, 4, 8), |b, s, h, d| {
        (b + 2 * s + h) as f64 - d as f64 * 0.5
    })
    .unwrap();
    let k = Tensor4::from_fn(TensorShape::new(2, 5, 2, 8), |b, s, h, d| {
        (b * s + h * d) as f64
    })
    .unwrap();
    let (q_expected, k_expected) = rope.rotate_query_key(&q, &k, pos(7)).unwrap();
    let (mut q2, mut k2) = (q, k);
    rope.rotate_query_key_with_table(&mut q2, &mut k2, &table, pos(7))
        .unwrap();
    assert_eq!(q2, q_expected);
    assert_eq!(k2, k_expected);
}

#[test]
fn parallel_rotation_with_more_threads_than_tokens() {
    let rope = RotaryEmbedding::new(RopeConfig::new(4, 10_000.0).unwrap());
    let table = rope
        .sin_cos_table(PositionRange::from_zero(3).unwrap())
        .unwrap();
    // B·S = 3 tokens, but 16 threads: the function uses only 3 blocks.
    let input = Tensor4::from_fn(TensorShape::new(1, 3, 2, 4), |_, s, h, d| {
        (s * 7 + h * 3 + d) as f64
    })
    .unwrap();
    let mut expected = input.clone();
    rope.rotate_tensor_in_place(&mut expected, Position::ZERO)
        .unwrap();
    for n in [1, 2, 3, 16] {
        let mut t = input.clone();
        rope.rotate_tensor_parallel(&mut t, &table, Position::ZERO, threads(n))
            .unwrap();
        assert_eq!(t, expected, "threads = {n}");
    }
}
