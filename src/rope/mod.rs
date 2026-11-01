//! RoPE-specific logic.
//!
//! This module contains the parts of the implementation that are specific to
//! Rotary Position Embeddings. It uses the general primitives in
//! [`crate::math`].
//!
//! ## Status (Stage 3)
//!
//! - [`config`]: `RopeConfig` (dimension `d`, base `b`). Implemented.
//! - [`frequencies`]: inverse frequencies `θⱼ = b^(−2j/d)` and positional
//!   angles `m · θⱼ` (`AngleTable`). Implemented.
//! - [`position`]: `Position` and `PositionRange`. Implemented.
//! - [`rotations`], [`embeddings`]: planned (Stage 4).
//! - [`scaling`]: planned (Stage 5).
//!
//! ## Data flow
//!
//! ```text
//! RopeConfig (d, b) ──► inverse_frequencies ──► [θ₀, ..., θ_(d/2−1)]
//!                                                      │
//! PositionRange (start, L) ─────────────────────────────┤
//!                                                      ▼
//!                                AngleTable: φ(m, j) = m · θⱼ
//! ```

pub mod config;
pub mod embeddings;
pub mod frequencies;
pub mod position;
pub mod rotations;
pub mod scaling;
