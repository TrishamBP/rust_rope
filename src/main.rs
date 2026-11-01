//! Demonstration binary for `rust_rope`.
//!
//! This binary is intentionally thin. All logic lives in the library crate
//! (`src/lib.rs`), so that tests, examples and benchmarks can use it too.

fn main() {
    // `env!` reads a value at compile time. Cargo sets these variables.
    println!(
        "{} v{} — Stage 8: RoPE with layouts, partial rotation and scaling complete.",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "Stage 9 (CPU performance engineering) is planned. See README.md and ROPE_Engineering.pdf."
    );
    println!("Run `cargo run --example basic_rope` to see the Stage 4 worked example.");
    println!("Run `cargo run --example math_foundations` to see the Stage 2 mathematics.");
    println!(
        "Run `cargo run --example rope_frequencies` to see the Stage 3 frequencies and angles."
    );
}
