//! # rust_rope
//!
//! An educational implementation of Rotary Position Embeddings (RoPE)
//! in Rust, written from scratch.
//!
//! ## Status
//!
//! Stage 3 (RoPE fundamentals: frequencies, positions, angles) is complete.
//!
//! - Implemented: [`math`] (vectors, 2 × 2 matrices, rotations, complex
//!   numbers, Euler's formula, trigonometry, float comparison).
//! - Implemented: [`rope::config`], [`rope::frequencies`] (inverse
//!   frequencies and positional angles), [`rope::position`], and
//!   [`utils::validation`].
//! - Planned: [`rope::rotations`] and [`rope::embeddings`] (Stage 4),
//!   [`rope::scaling`] (Stage 5). These modules are empty.
//!
//! ## Module layout
//!
//! - [`math`]: general mathematical primitives (complex numbers,
//!   trigonometry, matrices and vectors). This module does not know about RoPE.
//! - [`rope`]: RoPE-specific logic. It uses [`math`].
//! - [`utils`]: shared helpers, for example input validation.
//!
//! The dependency direction is one-way: `rope -> math`. The `math` module
//! never depends on the `rope` module.

pub mod math;
pub mod rope;
pub mod utils;
