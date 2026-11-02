//! Example: RoPE with a `sin`/`cos` table on query and key tensors (Stage 9).
//!
//! Run with: `cargo run --release --example cached_rope`
//!
//! This is the usage example of the README. It builds a LLaMA-like
//! configuration (`d = 128`, split-half layout, YaRN ×4 from a 4096-token
//! context), builds the `cos`/`sin` cache one time, and rotates a query
//! tensor with 32 heads and a key tensor with 8 heads (grouped-query
//! attention) for the tokens 1000..1512. Then it checks that the result is
//! bit-identical to the rotation without the table, and prints the times.

use rust_rope::math::tensor::{Tensor4, TensorShape};
use rust_rope::rope::config::{RopeConfig, RopeLayout};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::{Position, PositionRange};
use rust_rope::rope::scaling::RopeScaling;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A LLaMA-like head: d = 128, split-half layout, YaRN ×4 from a 4096 context.
    let config = RopeConfig::new(128, 10_000.0)?.with_layout(RopeLayout::SplitHalf);
    let rope = RotaryEmbedding::with_scaling(config, RopeScaling::yarn(4.0, 4096))?;

    // Build the cos/sin cache ONE time for the whole context (Stage 9).
    let started = Instant::now();
    let table = rope.sin_cos_table(PositionRange::from_zero(16_384)?)?;
    println!(
        "table: {} positions × {} pairs = {:.1} MiB, built in {:.2} ms",
        table.num_positions(),
        table.num_pairs(),
        table.size_in_bytes() as f64 / (1024.0 * 1024.0),
        started.elapsed().as_secs_f64() * 1e3
    );

    // Query [B=1, S=512, H=32, D=128] and key with 8 heads (grouped-query attention).
    let q0 = Tensor4::from_fn(TensorShape::new(1, 512, 32, 128), |_, s, h, d| {
        (s + h + d) as f64 * 1e-3
    })?;
    let k0 = Tensor4::from_fn(TensorShape::new(1, 512, 8, 128), |_, s, h, d| {
        (s * h + d) as f64 * 1e-3
    })?;

    // Tokens 1000..1512, e.g. the second chunk of a long prompt. No allocation.
    let start = Position::new(1000)?;
    let (mut q, mut k) = (q0.clone(), k0.clone());
    let started = Instant::now();
    rope.rotate_query_key_with_table(&mut q, &mut k, &table, start)?;
    let with_table = started.elapsed();

    // The same rotation without the table (Stage 5 function).
    let (mut q_direct, mut k_direct) = (q0, k0);
    let started = Instant::now();
    rope.rotate_query_key_in_place(&mut q_direct, &mut k_direct, start)?;
    let direct = started.elapsed();

    assert_eq!(q, q_direct, "the table must not change the result");
    assert_eq!(k, k_direct, "the table must not change the result");
    println!(
        "q + k rotation: direct {:.3} ms, with table {:.3} ms (one run, not a benchmark)",
        direct.as_secs_f64() * 1e3,
        with_table.as_secs_f64() * 1e3
    );
    println!("results are bit-identical: yes");
    Ok(())
}
