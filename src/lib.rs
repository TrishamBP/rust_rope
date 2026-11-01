//! # rust_rope
//!
//! An educational implementation of Rotary Position Embeddings (RoPE)
//! in Rust, written from scratch.
//!
//! ## Status
//!
//! Stage 8 (RoPE scaling) is complete. Stage 9 (CPU performance) is planned.
//!
//! - [`math`]: vectors, 2 × 2 matrices, rotations, complex numbers, Euler's
//!   formula, trigonometry, float comparison, the flat `[B, S, H, D]`
//!   tensor (`math::tensor`).
//! - [`rope`]: configuration (dimensions, base, layout, partial RoPE),
//!   frequencies and angles, positions, pair rotation, the RoPE
//!   transformation for vectors and query/key tensors, scaling methods.
//! - [`utils`]: validation and the error type `RopeError`.
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
