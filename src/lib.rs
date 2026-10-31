//! # rust_rope
//!
//! An educational implementation of Rotary Position Embeddings (RoPE)
//! in Rust, written from scratch.
//!
//! ## Status
//!
//! Stage 1 (project foundation). The modules exist, but they do not contain
//! algorithms yet. Each module comment describes its planned responsibility.
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
