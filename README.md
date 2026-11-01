# Rust ROPE

An educational, from-scratch implementation of **Rotary Position Embeddings
(RoPE)** in Rust.

> **Status:** Stage 8 complete (review pending) — RoPE scaling. The library
> rotates vectors and `[B, S, H, D]` query/key tensors with interleaved or
> split-half layouts, full or partial rotation, and four scaling methods
> (position interpolation, linear, NTK-aware, YaRN). An independent PyTorch
> reference validates 288 cases. Stage 9 (CPU performance) is planned and
> waits for approval. Full guide: `ROPE_Engineering.pdf`.

---

## 1. Project overview

This project has two equal objectives:

1. **Learn Rust** at a systems level: ownership, borrowing, lifetimes, memory
   allocation, traits, generics, iterators, error handling and performance.
2. **Learn the mathematics of RoPE**: trigonometry, rotation matrices, complex
   numbers and their use in Transformer models.

The code is the vehicle for learning. Each milestone has a detailed
explanation in [`explain.md`](explain.md). The permanent project rules are in
[`CLAUDE.md`](CLAUDE.md).

## 2. What is Rotary Position Embedding?

A **Transformer** is a neural network architecture. It processes a sequence of
tokens (words or parts of words). Each token is a vector of numbers. This
vector is the **embedding**.

**Rotary Position Embedding (RoPE)** puts the position of a token into its
vectors with a rotation. RoPE divides a vector into pairs of numbers. It
treats each pair as a point in a two-dimensional plane. Then it rotates each
point by an angle. The angle is the token position `m` multiplied by a
frequency `θ_i`. Each pair `i` has a different frequency.

```text
(x, y)  →  (x·cos(mθ_i) − y·sin(mθ_i),  x·sin(mθ_i) + y·cos(mθ_i))
```

RoPE was introduced in the RoFormer paper (Su et al., 2021). Many modern
language models use it, for example LLaMA.

## 3. Why do Transformers need positional information?

The **attention** operation compares each token with every other token. It
calculates a dot product between a "query" vector and a "key" vector.

Attention by itself does not know the order of the tokens. If you shuffle the
input tokens, each token gets the same attention result as before, only in a
shuffled order. Thus, "dog bites man" and "man bites dog" look the same to
attention without positional information. A Transformer needs an extra
mechanism that tells it where each token is.

## 4. Why is RoPE different from traditional positional embeddings?

| Method                                     | How it adds position                                 | Limitation                                           |
| ------------------------------------------ | ---------------------------------------------------- | ---------------------------------------------------- |
| Sinusoidal absolute (Vaswani et al., 2017) | **Adds** a fixed sine/cosine vector to the embedding | The model must learn to extract relative distance    |
| Learned absolute                           | **Adds** a trained vector for each position          | Fixed maximum length; no direct relative information |
| Relative bias (for example, T5)            | Adds a learned bias to the attention scores          | Extra parameters and extra calculation in attention  |
| **RoPE**                                   | **Rotates** the query and key vectors                | Needs scaling methods for very long contexts         |

The key property of RoPE: a rotation keeps the length of a vector. And the
dot product of a query at position `m` and a key at position `n` depends only
on the **relative distance** `m − n`. RoPE encodes absolute positions, but
attention sees relative positions. RoPE adds no trainable parameters.

## 5. Learning objectives

- Understand Rust ownership, borrowing, lifetimes and memory layout through
  real numerical code.
- Understand the stack, the heap, and when Rust allocates memory.
- Use traits, generics, iterators and error handling with a clear purpose.
- Derive the two-dimensional rotation matrix and its link to complex numbers.
- Derive the RoPE frequencies and prove the relative-position property.
- Understand context-extension methods: position interpolation, NTK-aware
  scaling and YaRN.
- Measure performance: allocation, cache locality and SIMD (Single
  Instruction, Multiple Data) opportunities.
- Validate numerical code with tests and mathematical invariants.

## 6. Planned implementation milestones (learning roadmap)

Legend: ✅ done · 🔄 in progress · ⏳ planned

### Stage 1: Project foundation — ✅

- Rust project initialization. ✅
- Module structure. ✅
- Rust development conventions. ✅
- Documentation setup. ✅

### Stage 2: Mathematical foundations — ✅

- Vectors and matrices. ✅ `math::matrix::{Vector, Matrix2}`
- Complex numbers. ✅ `math::complex::Complex`
- Trigonometric functions. ✅ `math::trigonometry`
- Two-dimensional rotations. ✅ `rotate_vector`, `Complex::rotate`
- Rotation matrices. ✅ `Matrix2::rotation`
- Euler's formula. ✅ `Complex::from_polar`
- Shared float comparison. ✅ `math::approx_eq`

### Stage 3: RoPE fundamentals — ✅

- Positional frequencies. ✅ `rope::config::RopeConfig`
- Inverse frequency calculations. ✅ `rope::frequencies::{inverse_frequency, inverse_frequencies}`
- Position indexing. ✅ `rope::position::{Position, PositionRange}`
- Position-dependent angles. ✅ `rope::frequencies::{angle, AngleTable}`
- Input validation. ✅ `utils::validation::RopeError`
- Applying rotations to embedding dimensions. ⏳ Moved to Stage 4 (the
  Stage 3 scope excludes vector rotation).

> **Roadmap change (2026-11-01):** the project owner redefined Stages 5–9.
> The old Stage 5 (advanced RoPE) is now Stage 8; the old Stage 6
> (performance) is now Stage 9; the old Stage 7 (validation) became
> Stage 6.

### Stage 4: RoPE implementation — ✅

- Pairwise dimension rotations. ✅ `rope::rotations::rotate_pair`
- Embedding transformations. ✅ `rope::embeddings::RotaryEmbedding::{rotate, rotate_in_place}`
- Numerical validation (invariants A–E). ✅ `tests/numerical_tests.rs`

### Stage 5: Query and key tensors — ✅

- Flat `[B, S, H, D]` tensor. ✅ `math::tensor::Tensor4`
- RoPE on queries and keys, values unchanged. ✅ `rotate_query_key[_in_place]`, `rotate_tensor_in_place`
- Relative-position property in scores. ✅ tested
- Simple benchmark. ✅ `benches/rope_benchmark.rs`

### Stage 6: PyTorch reference validation — ✅

- Independent reference. ✅ `reference/pytorch_rope.py`
- float64 and float32 comparison, error statistics. ✅ `reference/compare_rope.py`

### Stage 7: Layouts and partial RoPE — ✅

- Interleaved and split-half layouts. ✅ `rope::config::RopeLayout`
- Partial rotation. ✅ `RopeConfig::with_head_dimension`

### Stage 8: RoPE scaling — ✅

- Position interpolation, linear, NTK-aware, YaRN. ✅ `rope::scaling::RopeScaling`
- Not implemented: dynamic NTK, the Hugging Face YaRN discretization, LongRoPE.

### Stage 9: CPU performance engineering — ⏳ (waiting for approval)

- Criterion benchmarks, allocation analysis, `cos`/`sin` caches, loop
  variants, SIMD, Rayon, build settings.

## 7. Technology stack

- **Language:** Rust, edition 2024, stable toolchain.
- **Build tool:** Cargo.
- **Dependencies:** none. We use only the Rust standard library.
  Each future dependency must have a written justification in `explain.md`.
- **Hardware target:** CPU only. No GPU dependencies.

## 8. Build and execution

Install Rust with [rustup](https://rustup.rs). Then run these commands from
the project root:

```bash
cargo check --all-targets      # Type-check all code without a full build
cargo build                    # Build the library and the binary
cargo run                      # Run the demonstration binary (src/main.rs)
cargo test                     # Run unit tests and integration tests
cargo run --example math_foundations # Print the Stage 2 numerical examples
cargo run --example rope_frequencies # Print the Stage 3 frequencies and angles
cargo run --example basic_rope       # The Stage 4 worked example
cargo run --example export_reference_cases           # Stage 6: export cases
reference/.venv/Scripts/python reference/compare_rope.py  # Stage 6: PyTorch comparison
scripts/.venv/Scripts/python scripts/build_pdf.py        # rebuild ROPE_Engineering.pdf
cargo bench                    # Run the benchmarks (optimized build)
cargo fmt                      # Format the code
cargo clippy --all-targets     # Run the linter
cargo doc --open               # Build and open the API documentation
```

## Documentation

- [`explain.md`](explain.md) — the learning journal, one entry per milestone.
- [`docs/mathematics.md`](docs/mathematics.md) — mathematical notes.
- [`docs/architecture.md`](docs/architecture.md) — module design.
- [`docs/references.md`](docs/references.md) — papers and resources.
- [`ROPE_Engineering.pdf`](ROPE_Engineering.pdf) — the complete engineering guide (source:
  [`docs/ROPE_Engineering.md`](docs/ROPE_Engineering.md), diagrams in `docs/diagrams/`,
  generator `scripts/build_pdf.py`; setup: `python -m venv scripts/.venv` and
  `scripts/.venv/Scripts/python -m pip install -r scripts/requirements.txt`; needs Chrome or Edge).
