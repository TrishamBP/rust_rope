//! Example: the RoPE transformation of one embedding vector (Stage 4).
//!
//! Run with: `cargo run --example basic_rope`
//!
//! This example prints the numerical example of `explain.md`, Entry 10:
//! an 8-dimensional embedding `x = [1, 2, 3, 4, 5, 6, 7, 8]`, rotated at the
//! token positions 0, 1 and 2 with `b = 10000`. It shows every pair, its
//! angle, `cos φ`, `sin φ`, the rotated pair, and the pair length before and
//! after. It uses only the public API of the library.

use rust_rope::math::complex::Complex;
use rust_rope::rope::config::RopeConfig;
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::Position;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0)?);
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    println!("x = {x:?}");
    println!("θ = {:?}", rope.inverse_frequencies());

    for m in [0, 1, 2] {
        let position = Position::new(m)?;
        let y = rope.rotate(&x, position)?;
        println!();
        println!("== Position m = {m} ==");
        for (j, (&theta, (pair, out))) in rope
            .inverse_frequencies()
            .iter()
            .zip(x.as_chunks::<2>().0.iter().zip(y.as_chunks::<2>().0))
            .enumerate()
        {
            let phi = position.as_f64() * theta;
            let (s, c) = phi.sin_cos();
            println!(
                "pair {j}: ({}, {}) φ = {phi:<6} cos = {c:.9} sin = {s:.9} -> ({:.9}, {:.9})  |p|² {} -> {:.12}",
                pair[0],
                pair[1],
                out[0],
                out[1],
                pair[0] * pair[0] + pair[1] * pair[1],
                out[0] * out[0] + out[1] * out[1],
            );
        }
        let norm2_in: f64 = x.iter().map(|v| v * v).sum();
        let norm2_out: f64 = y.iter().map(|v| v * v).sum();
        println!("‖x‖² = {norm2_in}, ‖x'‖² = {norm2_out:.12}");
        println!("x' = {y:?}");
    }

    println!();
    println!("== Complex view of pair 0 at m = 1 ==");
    let z = Complex::new(1.0, 2.0);
    let w = Complex::from_polar(1.0, 1.0);
    println!("(1 + 2i)·e^(i·1) = {:?}", z * w);

    println!();
    println!("== In place, no new vector ==");
    let mut v = x;
    rope.rotate_in_place(&mut v, Position::new(1)?)?;
    println!("{v:?}");

    println!();
    println!("== Invalid input is an error ==");
    match rope.rotate(&[1.0, 2.0, 3.0], Position::ZERO) {
        Ok(_) => println!("unexpected success"),
        Err(e) => println!("error: {e}"),
    }
    Ok(())
}
