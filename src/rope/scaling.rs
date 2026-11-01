//! RoPE scaling methods for longer contexts (Stage 8).
//!
//! ## Implemented
//!
//! - [`RopeScaling`]: the scaling method, as an enum:
//!   - `None`: standard RoPE.
//!   - `PositionInterpolation { original_context, target_context }`
//!     (Chen et al., 2023): `m' = m · L/L'`.
//!   - `Linear { factor }`: `θ'ⱼ = θⱼ / s`. The same transformation as
//!     position interpolation, written with a factor `s = L'/L`.
//!   - `NtkAware { factor }` (static NTK-aware scaling, bloc97 2023, as
//!     written in the YaRN paper, section 3.1): `b' = b · s^(d/(d−2))`.
//!   - `Yarn { factor, original_context, alpha, beta }` (Peng et al., 2023,
//!     the paper's "NTK-by-parts" interpolation plus the attention factor
//!     `0.1·ln(s) + 1`).
//! - [`RopeScaling::validate`], [`scaled_inverse_frequencies`],
//!   [`RopeScaling::attention_factor`].
//!
//! ## Not implemented
//!
//! - Dynamic NTK scaling (the factor changes with the current sequence
//!   length).
//! - The Hugging Face discretization of YaRN (ramp over dimension indices
//!   with `floor`/`ceil`); see explain.md, Entry 16, section 3.7.
//! - LongRoPE and other methods.
//!
//! ## Design: scaling changes only the frequencies (and one factor)
//!
//! Every method in this module produces modified inverse frequencies
//! `θ'ⱼ`. [`crate::rope::embeddings::RotaryEmbedding::with_scaling`]
//! calculates them **one time**. The rotation loop does not know about
//! scaling: it reads `θ'ⱼ` exactly as it read `θⱼ`. YaRN also needs a
//! constant factor for the rotated components; the loop multiplies by it
//! (the factor is exactly 1.0 for all other methods, and `x · 1.0 = x`
//! exactly, so standard RoPE gives bit-identical results).
//!
//! Position interpolation scales positions, not frequencies. Because
//! `(m·s)·θ = m·(s·θ)` mathematically, we implement it as
//! `θ'ⱼ = θⱼ · L/L'`. Positions stay integers ([`crate::rope::position::Position`]).
//! The PyTorch reference implements it by scaling the positions, so the
//! comparison checks this equivalence.

use crate::rope::config::RopeConfig;
use crate::rope::frequencies::{inverse_frequencies, inverse_frequency_unchecked};
use crate::utils::validation::{
    RopeError, validate_context_lengths, validate_scaling_factor, validate_yarn_range,
};
use std::f64::consts::TAU;

/// The default YaRN ramp boundaries of the paper for LLaMA models:
/// `α = 1`, `β = 32` (Peng et al., 2023, section 3.2).
pub const YARN_DEFAULT_ALPHA: f64 = 1.0;
/// See [`YARN_DEFAULT_ALPHA`].
pub const YARN_DEFAULT_BETA: f64 = 32.0;

/// A RoPE scaling method.
///
/// Symbols: `s` = scaling factor (`s ≥ 1`), `L` = original (training)
/// context length, `L'` = target context length, `d` = rotary dimension,
/// `b` = base, `θⱼ = b^(−2j/d)`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RopeScaling {
    /// Standard RoPE: `θ'ⱼ = θⱼ`.
    #[default]
    None,
    /// Position interpolation: `m' = m · L/L'`, implemented as
    /// `θ'ⱼ = θⱼ · L/L'`. Requires `1 ≤ L ≤ L'`.
    PositionInterpolation {
        /// `L`, the context length of training.
        original_context: usize,
        /// `L'`, the desired context length.
        target_context: usize,
    },
    /// Linear frequency scaling: `θ'ⱼ = θⱼ / s`, `s ≥ 1`.
    Linear {
        /// `s`.
        factor: f64,
    },
    /// Static NTK-aware scaling: a new base `b' = b · s^(d/(d−2))`, then
    /// `θ'ⱼ = b'^(−2j/d)`. Requires `s ≥ 1` and `d ≥ 4` (for `d = 2` the
    /// exponent `d/(d−2)` is undefined).
    NtkAware {
        /// `s`.
        factor: f64,
    },
    /// YaRN (paper formulation):
    ///
    /// ```text
    /// λⱼ = 2π / θⱼ          r(j) = L / λⱼ
    /// γ(r) = 0 if r < α;  1 if r > β;  (r − α)/(β − α) otherwise
    /// θ'ⱼ = (1 − γ(r(j))) · θⱼ / s  +  γ(r(j)) · θⱼ
    /// attention factor = 0.1 · ln(s) + 1     (multiplies rotated q and k)
    /// ```
    Yarn {
        /// `s`.
        factor: f64,
        /// `L`, the context length of training.
        original_context: usize,
        /// `α`: below `α` rotations in `L`, a pair is fully interpolated.
        alpha: f64,
        /// `β`: above `β` rotations in `L`, a pair is not changed.
        beta: f64,
    },
}

impl RopeScaling {
    /// YaRN with the paper's default `α = 1`, `β = 32`.
    pub fn yarn(factor: f64, original_context: usize) -> Self {
        RopeScaling::Yarn {
            factor,
            original_context,
            alpha: YARN_DEFAULT_ALPHA,
            beta: YARN_DEFAULT_BETA,
        }
    }

    /// Checks the parameters of the method for the configuration `config`.
    ///
    /// # Errors
    ///
    /// - [`RopeError::InvalidScalingFactor`] if a factor is not finite or `< 1`.
    /// - [`RopeError::InvalidContextLengths`] if `L = 0` or `L' < L`.
    /// - [`RopeError::InvalidYarnRange`] if `α`, `β` are not finite or not `0 ≤ α < β`.
    /// - [`RopeError::NtkRequiresLargerDimension`] for `NtkAware` with `d = 2`.
    pub fn validate(&self, config: &RopeConfig) -> Result<(), RopeError> {
        match *self {
            RopeScaling::None => Ok(()),
            RopeScaling::PositionInterpolation {
                original_context,
                target_context,
            } => validate_context_lengths(original_context, target_context),
            RopeScaling::Linear { factor } => validate_scaling_factor(factor),
            RopeScaling::NtkAware { factor } => {
                validate_scaling_factor(factor)?;
                if config.rotary_dimension() < 4 {
                    return Err(RopeError::NtkRequiresLargerDimension {
                        rotary_dimension: config.rotary_dimension(),
                    });
                }
                Ok(())
            }
            RopeScaling::Yarn {
                factor,
                original_context,
                alpha,
                beta,
            } => {
                validate_scaling_factor(factor)?;
                validate_context_lengths(original_context, original_context)?;
                validate_yarn_range(alpha, beta)
            }
        }
    }

    /// The constant factor for the rotated components: `0.1·ln(s) + 1` for
    /// YaRN, exactly `1.0` for every other method.
    pub fn attention_factor(&self) -> f64 {
        match *self {
            RopeScaling::Yarn { factor, .. } => 0.1 * factor.ln() + 1.0,
            _ => 1.0,
        }
    }
}

/// The YaRN ramp `γ(r)`: 0 below `α`, 1 above `β`, linear in between.
fn yarn_ramp(r: f64, alpha: f64, beta: f64) -> f64 {
    if r < alpha {
        0.0
    } else if r > beta {
        1.0
    } else {
        (r - alpha) / (beta - alpha)
    }
}

/// Returns the scaled inverse frequencies `θ'ⱼ` for `config` and `scaling`.
///
/// The function allocates one `Vec` of `d/2` values (the same as
/// [`inverse_frequencies`]). It is called one time per
/// `RotaryEmbedding`, not per vector.
///
/// # Errors
///
/// The errors of [`RopeScaling::validate`].
///
/// # Example
///
/// ```
/// use rust_rope::rope::config::RopeConfig;
/// use rust_rope::rope::scaling::{scaled_inverse_frequencies, RopeScaling};
///
/// let config = RopeConfig::new(8, 10_000.0)?;
/// let pi = RopeScaling::PositionInterpolation { original_context: 2048, target_context: 8192 };
/// let theta = scaled_inverse_frequencies(&config, &pi)?;
/// assert_eq!(theta, vec![0.25, 0.025, 0.0025, 0.00025]); // θⱼ · 2048/8192
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
/// ```
pub fn scaled_inverse_frequencies(
    config: &RopeConfig,
    scaling: &RopeScaling,
) -> Result<Vec<f64>, RopeError> {
    scaling.validate(config)?;
    let d = config.rotary_dimension();
    let mut theta = inverse_frequencies(config);
    match *scaling {
        RopeScaling::None => {}
        RopeScaling::PositionInterpolation {
            original_context,
            target_context,
        } => {
            // m·(L/L')·θ = m·θ': scale each frequency by L/L' (≤ 1).
            let scale = original_context as f64 / target_context as f64;
            theta.iter_mut().for_each(|t| *t *= scale);
        }
        RopeScaling::Linear { factor } => {
            theta.iter_mut().for_each(|t| *t /= factor);
        }
        RopeScaling::NtkAware { factor } => {
            // b' = b · s^(d/(d−2)); then the standard formula with b'.
            let base = config.base() * factor.powf(d as f64 / (d as f64 - 2.0));
            for (j, t) in theta.iter_mut().enumerate() {
                *t = inverse_frequency_unchecked(j, d, base);
            }
        }
        RopeScaling::Yarn {
            factor,
            original_context,
            alpha,
            beta,
        } => {
            let l = original_context as f64;
            for t in theta.iter_mut() {
                let wavelength = TAU / *t; // λⱼ = 2π/θⱼ
                let rotations = l / wavelength; // r(j) = L/λⱼ
                let gamma = yarn_ramp(rotations, alpha, beta);
                *t = (1.0 - gamma) * (*t / factor) + gamma * *t;
            }
        }
    }
    Ok(theta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{DEFAULT_TOLERANCE, approx_eq};

    fn cfg(d: usize) -> RopeConfig {
        RopeConfig::new(d, 10_000.0).unwrap()
    }

    #[test]
    fn none_and_factor_one_give_the_standard_frequencies() {
        let c = cfg(16);
        let standard = inverse_frequencies(&c);
        for s in [
            RopeScaling::None,
            RopeScaling::Linear { factor: 1.0 },
            RopeScaling::NtkAware { factor: 1.0 },
            RopeScaling::yarn(1.0, 2048),
            RopeScaling::PositionInterpolation {
                original_context: 4096,
                target_context: 4096,
            },
        ] {
            let theta = scaled_inverse_frequencies(&c, &s).unwrap();
            for (a, b) in theta.iter().zip(&standard) {
                assert!(approx_eq(*a, *b, 1e-15), "{s:?}");
            }
            assert_eq!(s.attention_factor(), 1.0);
        }
    }

    #[test]
    fn position_interpolation_equals_linear_scaling() {
        let c = cfg(64);
        let pi = scaled_inverse_frequencies(
            &c,
            &RopeScaling::PositionInterpolation {
                original_context: 2048,
                target_context: 8192,
            },
        )
        .unwrap();
        let lin = scaled_inverse_frequencies(&c, &RopeScaling::Linear { factor: 4.0 }).unwrap();
        // ·0.25 and /4 are both exact operations here (powers of two).
        assert_eq!(pi, lin);
    }

    #[test]
    fn ntk_keeps_the_fastest_pair_and_interpolates_the_slowest() {
        // θ'₀ = 1 and θ'_(d/2−1) = θ_(d/2−1) / s: the derivation of b'.
        for d in [8, 64, 128] {
            let c = cfg(d);
            let theta = inverse_frequencies(&c);
            let ntk =
                scaled_inverse_frequencies(&c, &RopeScaling::NtkAware { factor: 4.0 }).unwrap();
            assert_eq!(ntk[0], 1.0);
            let last = d / 2 - 1;
            assert!(approx_eq(ntk[last], theta[last] / 4.0, 1e-13), "d = {d}");
            // All other pairs are between "unchanged" and "fully interpolated".
            for j in 1..last {
                assert!(
                    ntk[j] < theta[j] && ntk[j] > theta[j] / 4.0,
                    "d = {d}, j = {j}"
                );
            }
        }
    }

    #[test]
    fn yarn_bands() {
        // d = 8, L = 2048, s = 4: r = L/λ = [326, 32.6, 3.26, 0.326].
        let c = cfg(8);
        let theta = inverse_frequencies(&c);
        let y = scaled_inverse_frequencies(&c, &RopeScaling::yarn(4.0, 2048)).unwrap();
        assert_eq!(y[0], theta[0]); // r > β: unchanged
        assert_eq!(y[1], theta[1]); // r = 32.6 > β = 32: unchanged
        assert_eq!(y[3], theta[3] / 4.0); // r < α: fully interpolated
        let r2 = 2048.0 / (TAU / theta[2]);
        let g = (r2 - 1.0) / 31.0;
        assert!(approx_eq(
            y[2],
            (1.0 - g) * theta[2] / 4.0 + g * theta[2],
            DEFAULT_TOLERANCE
        ));
        assert!(approx_eq(
            RopeScaling::yarn(4.0, 2048).attention_factor(),
            0.1 * 4f64.ln() + 1.0,
            DEFAULT_TOLERANCE
        ));
    }

    #[test]
    fn invalid_parameters() {
        let c = cfg(8);
        for f in [0.5, 0.0, -2.0, f64::INFINITY] {
            assert_eq!(
                RopeScaling::Linear { factor: f }.validate(&c),
                Err(RopeError::InvalidScalingFactor { factor: f })
            );
        }
        assert!(
            RopeScaling::NtkAware { factor: f64::NAN }
                .validate(&c)
                .is_err()
        );
        assert_eq!(
            RopeScaling::PositionInterpolation {
                original_context: 4096,
                target_context: 2048
            }
            .validate(&c),
            Err(RopeError::InvalidContextLengths {
                original: 4096,
                target: 2048
            })
        );
        assert!(
            RopeScaling::PositionInterpolation {
                original_context: 0,
                target_context: 2048
            }
            .validate(&c)
            .is_err()
        );
        assert!(RopeScaling::yarn(4.0, 0).validate(&c).is_err());
        assert_eq!(
            RopeScaling::Yarn {
                factor: 4.0,
                original_context: 2048,
                alpha: 32.0,
                beta: 1.0
            }
            .validate(&c),
            Err(RopeError::InvalidYarnRange {
                alpha: 32.0,
                beta: 1.0
            })
        );
        assert_eq!(
            RopeScaling::NtkAware { factor: 2.0 }.validate(&cfg(2)),
            Err(RopeError::NtkRequiresLargerDimension {
                rotary_dimension: 2
            })
        );
    }
}
