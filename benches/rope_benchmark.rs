//! Benchmarks for `rust_rope`.
//!
//! Run with: `cargo bench`
//!
//! `Cargo.toml` sets `harness = false` for this file. Thus, this file supplies
//! its own `main` function. The built-in `#[bench]` harness needs the nightly
//! compiler, and we use the stable compiler.
//!
//! ## Status (Stage 5)
//!
//! A **simple** timing benchmark with `std::time::Instant` and no external
//! crate. It measures RoPE on query/key tensors `[B, S, H, D]`:
//!
//! - `rotate_tensor_in_place`: no allocation.
//! - `rotate_query_key`: two tensor clones plus the rotation.
//!
//! Limits of this method: no statistical analysis, no outlier detection.
//! Each case runs a warm-up, then repeats the measurement and reports the
//! fastest and the median repetition. The planned performance stage
//! (Stage 9) will improve the methodology.

use rust_rope::math::tensor::{Tensor4, TensorShape};
use rust_rope::rope::config::{RopeConfig, RopeLayout};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::Position;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// Runs `f` `repetitions` times after `warmup` runs, and returns the
/// sorted durations.
fn measure(mut f: impl FnMut(), warmup: usize, repetitions: usize) -> Vec<Duration> {
    for _ in 0..warmup {
        f();
    }
    let mut times: Vec<Duration> = (0..repetitions)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed()
        })
        .collect();
    times.sort();
    times
}

fn report(name: &str, shape: TensorShape, times: &[Duration]) {
    let pairs = (shape.batch * shape.sequence * shape.heads * shape.features / 2) as f64;
    let best = times[0];
    let median = times[times.len() / 2];
    println!(
        "{name:<28} [{}, {}, {}, {}]  best {:>9.3} ms  median {:>9.3} ms  {:>6.2} ns/pair (best)",
        shape.batch,
        shape.sequence,
        shape.heads,
        shape.features,
        best.as_secs_f64() * 1e3,
        median.as_secs_f64() * 1e3,
        best.as_secs_f64() * 1e9 / pairs,
    );
}

fn main() {
    println!("rust_rope benchmark (Stage 5, std::time::Instant, release build)");
    let shapes = [
        TensorShape::new(1, 128, 8, 64),
        TensorShape::new(1, 512, 8, 64),
        TensorShape::new(2, 512, 16, 64),
    ];
    for shape in shapes {
        let rope = RotaryEmbedding::new(
            RopeConfig::new(shape.features, 10_000.0).expect("valid configuration"),
        );
        // Deterministic input; created outside the timed region.
        let q = Tensor4::from_fn(shape, |b, s, h, d| ((b + s + h + d) as f64 * 0.01).sin())
            .expect("valid shape");
        let k = q.clone();

        let mut work = q.clone();
        let times = measure(
            || {
                // Rotating the same buffer again is fine for timing: the
                // amount of work does not depend on the values.
                rope.rotate_tensor_in_place(black_box(&mut work), Position::ZERO)
                    .expect("valid input");
            },
            3,
            15,
        );
        report("rotate_tensor_in_place", shape, &times);

        let times = measure(
            || {
                let out = rope
                    .rotate_query_key(black_box(&q), black_box(&k), Position::ZERO)
                    .expect("valid input");
                black_box(out);
            },
            3,
            15,
        );
        report("rotate_query_key (2 clones)", shape, &times);
    }

    // Stage 7: layouts and partial RoPE, in place.
    println!();
    println!("Stage 7: layouts and partial rotation (rotate_tensor_in_place)");
    let shape = TensorShape::new(2, 512, 16, 128);
    let input = Tensor4::from_fn(shape, |b, s, h, d| ((b + s + h + d) as f64 * 0.01).sin())
        .expect("valid shape");
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        for rotary in [128, 64] {
            let config = RopeConfig::new(rotary, 10_000.0)
                .and_then(|c| c.with_layout(layout).with_head_dimension(128))
                .expect("valid configuration");
            let rope = RotaryEmbedding::new(config);
            let mut work = input.clone();
            let times = measure(
                || {
                    rope.rotate_tensor_in_place(black_box(&mut work), Position::ZERO)
                        .expect("valid input");
                },
                3,
                15,
            );
            report(&format!("{layout:?} r={rotary}"), shape, &times);
        }
    }
}
