//! General mathematical primitives.
//!
//! This module contains mathematics that is not specific to RoPE.
//! It must not depend on [`crate::rope`]. This rule keeps the primitives
//! reusable and easy to test in isolation.
//!
//! Status: planned. The submodules are empty in Stage 1.

pub mod complex;
pub mod matrix;
pub mod trigonometry;
