//! RoPE-specific logic.
//!
//! This module contains the parts of the implementation that are specific to
//! Rotary Position Embeddings. It uses the general primitives in
//! [`crate::math`].
//!
//! ## Status (Stage 8)
//!
//! - [`config`]: `RopeConfig` (dimension `d`, base `b`). Implemented.
//! - [`frequencies`]: inverse frequencies `θⱼ = b^(−2j/d)` and positional
//!   angles `m · θⱼ` (`AngleTable`). Implemented.
//! - [`position`]: `Position` and `PositionRange`. Implemented.
//! - [`rotations`]: `rotate_pair`. Implemented (Stage 4).
//! - [`embeddings`]: `RotaryEmbedding` for vectors (Stage 4), query/key
//!   tensors (Stage 5), layouts and partial RoPE (Stage 7), scaling (Stage 8).
//! - [`scaling`]: `RopeScaling` (position interpolation, linear, NTK-aware,
//!   YaRN). Implemented (Stage 8).
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
