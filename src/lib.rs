//! # rust_rope
//!
//! An educational implementation of Rotary Position Embeddings (RoPE)
//! in Rust, written from scratch.
//!
//! ## Status
//!
//! Stage 2 (mathematical foundations) is complete.
//!
//! - Implemented: [`math`] (vectors, 2 × 2 matrices, rotations, complex
//!   numbers, Euler's formula, trigonometry, float comparison).
//! - Planned: [`rope`] (Stages 3–5) and [`utils`] (Stage 3). These modules
//!   are empty. Each module comment describes its planned responsibility.
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
