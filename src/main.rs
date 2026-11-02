//! Demonstration binary for `rust_rope`.
//!
//! This binary is intentionally thin. All logic lives in the library crate
//! (`src/lib.rs`), so that tests, examples and benchmarks can use it too.

fn main() {
    // `env!` reads a value at compile time. Cargo sets these variables.
    println!(
        "{} v{} — Stage 9: RoPE with layouts, partial rotation, scaling and CPU optimizations.",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    );
    println!(
        "Run `cargo bench --bench rope_criterion` for the Stage 9 benchmarks (see README.md)."
    );
    println!("Run `cargo run --example basic_rope` to see the Stage 4 worked example.");
    println!("Run `cargo run --example math_foundations` to see the Stage 2 mathematics.");
    println!(
        "Run `cargo run --example rope_frequencies` to see the Stage 3 frequencies and angles."
    );
}
