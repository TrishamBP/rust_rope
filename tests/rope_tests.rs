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
//! The numerical properties (values, monotonicity, linearity) are in
//! `tests/numerical_tests.rs`.

use rust_rope::rope::config::{DEFAULT_BASE, RopeConfig};
use rust_rope::rope::frequencies::{AngleTable, angle, inverse_frequencies, inverse_frequency};
use rust_rope::rope::position::{Position, PositionRange};
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
