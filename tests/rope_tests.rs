//! Integration tests for the RoPE API.
//!
//! Integration tests use the library from the outside, as a user does.
//! Cargo compiles each file in `tests/` as a separate crate.
//!
//! Status: placeholder. Tests are added together with each RoPE feature.

/// Confirms that the library crate links into an external crate, and that
/// the public module paths exist.
#[test]
fn library_modules_are_public() {
    // `use ... as _` imports a path without a name. It only checks that the
    // path exists. It does not execute code.
    #[allow(unused_imports)]
    use rust_rope::{math as _, rope as _, utils as _};
}
