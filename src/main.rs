//! Demonstration binary for `rust_rope`.
//!
//! This binary is intentionally thin. All logic lives in the library crate
//! (`src/lib.rs`), so that tests, examples and benchmarks can use it too.

fn main() {
    // `env!` reads a value at compile time. Cargo sets these variables.
    println!(
        "{} v{} — Stage 3: RoPE frequencies, positions and angles complete.",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    println!("Embedding rotation is not implemented yet (Stage 4). See README.md for the roadmap.");
    println!("Run `cargo run --example math_foundations` to see the Stage 2 mathematics.");
    println!(
        "Run `cargo run --example rope_frequencies` to see the Stage 3 frequencies and angles."
    );
}
