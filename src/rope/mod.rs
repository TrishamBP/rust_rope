//! RoPE-specific logic.
//!
//! This module contains the parts of the implementation that are specific to
//! Rotary Position Embeddings. It uses the general primitives in
//! [`crate::math`].
//!
//! Status: planned. The submodules are empty in Stage 1.

pub mod config;
pub mod embeddings;
pub mod frequencies;
pub mod position;
pub mod rotations;
pub mod scaling;
