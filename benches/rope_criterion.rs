//! Statistical CPU benchmarks for `rust_rope` (Stage 9).
//!
//! Run all groups:   `cargo bench --bench rope_criterion`
//! Run one group:    `cargo bench --bench rope_criterion -- tensor_shapes`
//! Quick smoke run:  `cargo bench --bench rope_criterion -- --quick`
//!
//! [Criterion](https://docs.rs/criterion) runs each function many times,
//! fits a linear model to the measurements, and reports a confidence
//! interval and outliers. It saves the results in `target/criterion/`. The
//! next run compares with the saved results and reports the change.
//!
//! ## Groups
//!
//! | Group | Question |
//! |---|---|
//! | `setup` | What do the frequencies and the `sin`/`cos` table cost? |
//! | `tensor_shapes` | Direct vs. table vs. multithreaded, for real model shapes |
//! | `positions` | Does the position (the angle size) change the cost? |
//! | `layouts` | Interleaved vs. split-half, full vs. partial rotation |
//! | `query_key` | Copying vs. in-place vs. table, for a query and a key tensor |
//! | `thread_scaling` | How does the speed change with the number of threads? |
//! | `shakespeare` | End-to-end: text → token IDs → embeddings → RoPE |
//!
//! ## Throughput
//!
//! Each group declares `Throughput::Elements(pairs)`. Criterion then also
//! reports "pairs per second". One pair = one 2D rotation.
//!
//! ## Input data
//!
//! RoPE speed does not depend on the **vector values** (no data-dependent
//! branches, no subnormal numbers). It **does** depend on the **positions**:
//! the cost of `sin_cos` grows with the size of the angle `m·θⱼ` (group
//! `positions`). Thus most groups use deterministic synthetic values, and
//! each case uses a realistic first position. The `shakespeare` group reads `data/shakespeare.txt` (Tiny
//! Shakespeare, 1 115 394 bytes) when it exists. Otherwise it uses a
//! synthetic text, so that `cargo bench` also works on a fresh clone.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rust_rope::math::tensor::{Tensor4, TensorShape};
use rust_rope::rope::config::{RopeConfig, RopeLayout};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::frequencies::inverse_frequencies;
use rust_rope::rope::position::{Position, PositionRange};
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// The number of 2D rotations in one tensor: `B · S · H · d/2`.
fn pairs(shape: TensorShape, rotary_dimension: usize) -> u64 {
    (shape.batch * shape.sequence * shape.heads * rotary_dimension / 2) as u64
}

/// Deterministic input values in about `[−1, 1]`. Created outside the
/// timed region.
fn input(shape: TensorShape) -> Tensor4 {
    Tensor4::from_fn(shape, |b, s, h, d| {
        ((b * 7919 + s * 131 + h * 17 + d) as f64 * 0.001).sin()
    })
    .expect("valid shape")
}

fn rope(rotary: usize, head: usize, layout: RopeLayout) -> RotaryEmbedding {
    let config = RopeConfig::new(rotary, 10_000.0)
        .expect("valid dimension")
        .with_layout(layout)
        .with_head_dimension(head)
        .expect("valid head dimension");
    RotaryEmbedding::new(config)
}

/// The number of hardware threads (16 on the reference computer).
fn hardware_threads() -> NonZeroUsize {
    std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
}

fn shape_label(s: TensorShape) -> String {
    format!("{}x{}x{}x{}", s.batch, s.sequence, s.heads, s.features)
}

/// A small deterministic pseudo-random generator (SplitMix64). We need
/// reproducible "random" embedding values without an external crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `[−1, 1)`: the top 53 bits give an exact `f64`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Group 1: setup costs
// ---------------------------------------------------------------------------

fn bench_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup");
    let config = RopeConfig::new(128, 10_000.0).expect("valid configuration");
    group.bench_function("inverse_frequencies d=128", |b| {
        b.iter(|| inverse_frequencies(black_box(&config)))
    });
    let rope = RotaryEmbedding::new(config);
    for length in [2048_usize, 8192] {
        let range = PositionRange::from_zero(length).expect("valid range");
        // Throughput: one `sin_cos` call per table entry.
        group.throughput(Throughput::Elements((length * 64) as u64));
        group.bench_with_input(
            BenchmarkId::new("sin_cos_table d=128", length),
            &range,
            |b, &range| b.iter(|| rope.sin_cos_table(black_box(range)).expect("valid")),
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2: direct vs. table vs. threads, for real model shapes
// ---------------------------------------------------------------------------

fn bench_tensor_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_shapes");
    group.measurement_time(Duration::from_secs(4));
    let threads = hardware_threads();
    // (shape, first position). A prefill starts at position 0. A decode step
    // processes one new token at a large position: the cost of `sin_cos`
    // depends on the size of the angle (see the `positions` group), so a
    // decode step at position 0 would be unrealistically cheap.
    let cases = [
        // One decode step of a LLaMA-7B-like model: 1 token, 32 heads, D = 128,
        // at position 4095 (the last position of a 4096-token context).
        (TensorShape::new(1, 1, 32, 128), 4095),
        // A short prompt for a small model (GPT-2-like head size 64).
        (TensorShape::new(1, 128, 8, 64), 0),
        // The Stage 5 shape, for comparison with the old results.
        (TensorShape::new(2, 512, 16, 64), 0),
        // A LLaMA-7B-like prefill of 2048 tokens: 8.4 M values, 64 MiB.
        (TensorShape::new(1, 2048, 32, 128), 0),
    ];
    for (shape, start) in cases {
        let d = shape.features;
        let rope = rope(d, d, RopeLayout::Interleaved);
        let table = rope
            .sin_cos_table(PositionRange::from_zero(start + shape.sequence).expect("valid"))
            .expect("valid table");
        let mut work = input(shape);
        let label = if start == 0 {
            shape_label(shape)
        } else {
            format!("{}@{start}", shape_label(shape))
        };
        let start = Position::new(start).expect("valid position");
        group.throughput(Throughput::Elements(pairs(shape, d)));

        // Rotating the same buffer again and again is valid for timing: the
        // amount of work does not depend on the values, and a rotation keeps
        // the lengths, so the values never overflow.
        group.bench_function(BenchmarkId::new("direct", &label), |b| {
            b.iter(|| {
                rope.rotate_tensor_in_place(black_box(&mut work), start)
                    .expect("valid")
            })
        });
        group.bench_function(BenchmarkId::new("table", &label), |b| {
            b.iter(|| {
                rope.rotate_tensor_with_table(black_box(&mut work), &table, start)
                    .expect("valid")
            })
        });
        group.bench_function(
            BenchmarkId::new(format!("table+{}threads", threads.get()), &label),
            |b| {
                b.iter(|| {
                    rope.rotate_tensor_parallel(black_box(&mut work), &table, start, threads)
                        .expect("valid")
                })
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 2b: the cost of sin_cos depends on the position
// ---------------------------------------------------------------------------

/// The same decode-step tensor at different positions. The vector values do
/// not change the cost, but the **angles** `m·θⱼ` do: `sin_cos` must reduce
/// a large angle into `[−π/4, π/4]` first, and `sin_cos(0)` is a fast path.
/// The table path does no trigonometry, so its cost does not depend on `m`.
fn bench_positions(c: &mut Criterion) {
    let mut group = c.benchmark_group("positions");
    let shape = TensorShape::new(1, 1, 32, 128);
    let rope = rope(128, 128, RopeLayout::Interleaved);
    let mut work = input(shape);
    group.throughput(Throughput::Elements(pairs(shape, 128)));
    for m in [0_usize, 1, 4095, 1_000_000] {
        let position = Position::new(m).expect("valid position");
        group.bench_with_input(BenchmarkId::new("direct", m), &position, |b, &position| {
            b.iter(|| {
                rope.rotate_tensor_in_place(black_box(&mut work), black_box(position))
                    .expect("valid")
            })
        });
        let table = rope
            .sin_cos_table(PositionRange::new(m, 1).expect("valid"))
            .expect("valid table");
        group.bench_with_input(BenchmarkId::new("table", m), &position, |b, &position| {
            b.iter(|| {
                rope.rotate_tensor_with_table(black_box(&mut work), &table, black_box(position))
                    .expect("valid")
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 3: layouts and partial rotation
// ---------------------------------------------------------------------------

fn bench_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("layouts");
    let shape = TensorShape::new(2, 512, 16, 128);
    let base = input(shape);
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        for rotary in [128, 64] {
            let rope = rope(rotary, 128, layout);
            let table = rope
                .sin_cos_table(PositionRange::from_zero(shape.sequence).expect("valid"))
                .expect("valid table");
            let mut work = base.clone();
            // Throughput counts only the rotated pairs.
            group.throughput(Throughput::Elements(pairs(shape, rotary)));
            let label = format!("{layout:?} d={rotary}");
            group.bench_function(BenchmarkId::new("direct", &label), |b| {
                b.iter(|| {
                    rope.rotate_tensor_in_place(black_box(&mut work), Position::ZERO)
                        .expect("valid")
                })
            });
            group.bench_function(BenchmarkId::new("table", &label), |b| {
                b.iter(|| {
                    rope.rotate_tensor_with_table(black_box(&mut work), &table, Position::ZERO)
                        .expect("valid")
                })
            });
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 4: query and key together
// ---------------------------------------------------------------------------

fn bench_query_key(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_key");
    // Grouped-query attention, as in LLaMA-2-70B / LLaMA-3: 32 query heads
    // and 8 key heads share each position.
    let q_shape = TensorShape::new(1, 512, 32, 128);
    let k_shape = TensorShape::new(1, 512, 8, 128);
    let rope = rope(128, 128, RopeLayout::Interleaved);
    let table = rope
        .sin_cos_table(PositionRange::from_zero(512).expect("valid"))
        .expect("valid table");
    let (q, k) = (input(q_shape), input(k_shape));
    group.throughput(Throughput::Elements(
        pairs(q_shape, 128) + pairs(k_shape, 128),
    ));

    group.bench_function("copying (2 clones)", |b| {
        b.iter(|| {
            rope.rotate_query_key(black_box(&q), black_box(&k), Position::ZERO)
                .expect("valid")
        })
    });
    let (mut qw, mut kw) = (q.clone(), k.clone());
    group.bench_function("in place", |b| {
        b.iter(|| {
            rope.rotate_query_key_in_place(black_box(&mut qw), black_box(&mut kw), Position::ZERO)
                .expect("valid")
        })
    });
    group.bench_function("in place + table", |b| {
        b.iter(|| {
            rope.rotate_query_key_with_table(
                black_box(&mut qw),
                black_box(&mut kw),
                &table,
                Position::ZERO,
            )
            .expect("valid")
        })
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 5: thread scaling
// ---------------------------------------------------------------------------

fn bench_thread_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_scaling");
    let shape = TensorShape::new(1, 2048, 32, 128);
    let rope = rope(128, 128, RopeLayout::Interleaved);
    let table = rope
        .sin_cos_table(PositionRange::from_zero(shape.sequence).expect("valid"))
        .expect("valid table");
    let mut work = input(shape);
    group.throughput(Throughput::Elements(pairs(shape, 128)));
    let max = hardware_threads().get();
    let counts: Vec<usize> = [1, 2, 4, 8, 16, 32]
        .into_iter()
        .filter(|&n| n <= 2 * max)
        .collect();
    for n in counts {
        let threads = NonZeroUsize::new(n).expect("n ≥ 1");
        group.bench_with_input(BenchmarkId::new("table", n), &threads, |b, &threads| {
            b.iter(|| {
                rope.rotate_tensor_parallel(black_box(&mut work), &table, Position::ZERO, threads)
                    .expect("valid")
            })
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Group 6: end-to-end on Tiny Shakespeare
// ---------------------------------------------------------------------------

/// The text, the character vocabulary and the token IDs.
struct Corpus {
    source: &'static str,
    tokens: Vec<u8>,
    vocabulary: usize,
}

/// Loads `data/shakespeare.txt`, or a synthetic text if the file is missing.
///
/// Tokenizer: one token per byte (character-level, as in Karpathy's
/// char-rnn and nanoGPT "shakespeare_char"). The vocabulary is the sorted
/// set of bytes that occur in the text (65 for Tiny Shakespeare).
fn load_corpus() -> Corpus {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/shakespeare.txt");
    let (bytes, source) = match std::fs::read(path) {
        Ok(bytes) => (bytes, "data/shakespeare.txt"),
        Err(_) => {
            let text = "To be, or not to be, that is the question.\n".repeat(30_000);
            (
                text.into_bytes(),
                "synthetic text (data/shakespeare.txt not found)",
            )
        }
    };
    // `seen[byte]` marks the bytes that occur. A byte has 256 values, so a
    // fixed array on the stack is enough: no HashMap.
    let mut seen = [false; 256];
    for &byte in &bytes {
        seen[byte as usize] = true;
    }
    let mut id_of = [0u8; 256];
    let mut vocabulary = 0;
    for byte in 0..256 {
        if seen[byte] {
            id_of[byte] = vocabulary as u8;
            vocabulary += 1;
        }
    }
    let tokens = bytes.iter().map(|&byte| id_of[byte as usize]).collect();
    Corpus {
        source,
        tokens,
        vocabulary,
    }
}

/// Fills a `[B, S, H, D]` tensor from consecutive text windows: row `b`
/// holds the tokens `b·S .. b·S + S`. Each token copies its embedding row
/// (`H · D` values) from the table.
///
/// A real model computes `q = x · W_q` with a trained matrix `W_q`. We have
/// no trained weights, so the embedding table plays the role of `x · W_q`.
/// This does not change the RoPE work.
fn embed(tokens: &[u8], table: &[f64], tensor: &mut Tensor4) {
    let shape = tensor.shape();
    let token_len = shape.heads * shape.features;
    for (t, row) in tensor
        .as_mut_slice()
        .chunks_exact_mut(token_len)
        .enumerate()
    {
        let id = tokens[t] as usize;
        row.copy_from_slice(&table[id * token_len..(id + 1) * token_len]);
    }
}

fn bench_shakespeare(c: &mut Criterion) {
    let corpus = load_corpus();
    // A small character-level GPT: 6 heads × D = 64 (d_model = 384, as in
    // nanoGPT "shakespeare_char"), context S = 256, batch B = 64.
    let shape = TensorShape::new(64, 256, 6, 64);
    let token_len = shape.heads * shape.features;
    let window = shape.batch * shape.sequence; // 16 384 tokens per batch
    assert!(corpus.tokens.len() >= window, "the corpus is too short");
    println!(
        "shakespeare: source = {}, {} tokens, vocabulary = {}, one batch = {} tokens",
        corpus.source,
        corpus.tokens.len(),
        corpus.vocabulary,
        window
    );

    // Fixed-seed embedding table: vocabulary × (H·D) values in [−0.1, 0.1).
    let mut rng = SplitMix64(0x0005_EED0_FA11);
    let embedding: Vec<f64> = (0..corpus.vocabulary * token_len)
        .map(|_| rng.next_f64() * 0.1)
        .collect();

    let rope = rope(64, 64, RopeLayout::Interleaved);
    let table = rope
        .sin_cos_table(PositionRange::from_zero(shape.sequence).expect("valid"))
        .expect("valid table");
    let tokens = &corpus.tokens[..window];
    let mut q = Tensor4::zeros(shape).expect("valid shape");
    let mut k = Tensor4::zeros(shape).expect("valid shape");

    let mut group = c.benchmark_group("shakespeare");
    group.throughput(Throughput::Elements(2 * pairs(shape, 64)));
    group.bench_function("embed only (q, k)", |b| {
        b.iter(|| {
            embed(black_box(tokens), &embedding, &mut q);
            embed(black_box(tokens), &embedding, &mut k);
        })
    });
    group.bench_function("embed + rope direct", |b| {
        b.iter(|| {
            embed(black_box(tokens), &embedding, &mut q);
            embed(black_box(tokens), &embedding, &mut k);
            rope.rotate_query_key_in_place(&mut q, &mut k, Position::ZERO)
                .expect("valid")
        })
    });
    group.bench_function("embed + rope table", |b| {
        b.iter(|| {
            embed(black_box(tokens), &embedding, &mut q);
            embed(black_box(tokens), &embedding, &mut k);
            rope.rotate_query_key_with_table(&mut q, &mut k, &table, Position::ZERO)
                .expect("valid")
        })
    });
    let threads = hardware_threads();
    group.bench_function(
        format!("embed + rope table+{}threads", threads.get()),
        |b| {
            b.iter(|| {
                embed(black_box(tokens), &embedding, &mut q);
                embed(black_box(tokens), &embedding, &mut k);
                rope.rotate_tensor_parallel(&mut q, &table, Position::ZERO, threads)
                    .expect("valid");
                rope.rotate_tensor_parallel(&mut k, &table, Position::ZERO, threads)
                    .expect("valid")
            })
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_setup,
    bench_tensor_shapes,
    bench_positions,
    bench_layouts,
    bench_query_key,
    bench_thread_scaling,
    bench_shakespeare
);
criterion_main!(benches);
