//! Demonstration binary for `rust_rope`.
//!
//! This binary is intentionally thin. All logic lives in the library crate
//! (`src/lib.rs`), so that tests, examples and benchmarks can use it too.

fn main() {
    // `env!` reads a value at compile time. Cargo sets these variables.
    println!(
        "{} v{} — Stage 1: project foundation.",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    println!("No RoPE algorithms are implemented yet. See README.md for the roadmap.");
}
