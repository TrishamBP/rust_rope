# explain.md — Learning Journal for `rust_rope`

This file is the learning journal of the project.
Each implementation milestone adds one new entry.

**Rules for this file** (from `CLAUDE.md`):

- Never delete or overwrite previous entries. Append new entries.
- If an old entry becomes incorrect, add a dated correction note below it.
- Each entry uses the same 10-section structure.
- Each entry clearly separates **implemented** features from **planned**
  features.

---

## Entry index

| Entry | Date | Stage | Topic |
|---|---|---|---|
| 1 | 2026-10-31 | Stage 1 | Project foundation: structure, documentation, conventions |

---

# Entry 1 — Project foundation

**Date:** 2026-10-31
**Stage:** 1 (Project foundation)

### Implementation status of this entry

| Item | Status |
|---|---|
| Cargo package (library + binary) | ✅ Implemented |
| Module tree (`math`, `rope`, `utils`) | ✅ Implemented (empty modules) |
| Placeholder example, benchmark and tests | ✅ Implemented |
| Documentation (`README.md`, `CLAUDE.md`, `docs/`, this file) | ✅ Implemented |
| Any RoPE mathematics | ⏳ Planned (Stages 2–5) |
| Frequencies, rotations, scaling | ⏳ Planned (Stages 3–5) |
| Real benchmarks | ⏳ Planned (Stage 6) |

---

## 1. What did we build?

We built the **skeleton** of the project. A skeleton is a structure with no
functional content yet. It contains:

- A **Cargo package**. Cargo is the Rust build tool and package manager.
  A package is a folder with a `Cargo.toml` file.
- A **library crate** (`src/lib.rs`). A crate is the unit that the Rust
  compiler compiles. The library crate will contain all the logic.
- A **binary crate** (`src/main.rs`). It prints a status message.
- Three **modules**: `math`, `rope` and `utils`, with ten submodules.
  A module is a named container for code inside a crate.
- One example, one benchmark and two integration test files. Each is a
  placeholder.
- The documentation: `README.md`, `CLAUDE.md`, this file, and three files in
  `docs/`.

The project compiles, and all tests pass. No algorithm exists yet.

## 2. Why did we build it?

A clear structure before code has three advantages:

1. **Each future feature has a known place.** When we implement the
   frequencies in Stage 3, the code goes into `src/rope/frequencies.rs`.
   We do not need to decide this again.
2. **The rules exist before the code.** `CLAUDE.md` defines how we write,
   test and explain. Thus, every stage follows the same standard.
3. **The learning record starts at the beginning.** This file records the
   reasons for each decision, from the first day.

### 2.1 Why we selected Rust

Rust is a systems programming language. It gives control over memory, like C
and C++. But the compiler checks memory safety at compile time.

These properties make Rust a good teacher:

- **Explicit ownership.** Each value has one owner. The compiler shows you
  when a value moves, when it is borrowed, and when it is freed. In Python,
  this happens without your knowledge.
- **Explicit allocation.** You see when code uses the heap (for example
  `Vec<f32>`) and when it uses the stack (for example `[f32; 4]`).
- **No garbage collector.** A garbage collector is a runtime system that
  frees memory automatically, at unpredictable times. Rust frees memory at a
  known point: when the owner goes out of scope. This makes performance
  predictable.
- **Zero-cost abstractions.** Iterators and generics compile to code that is
  usually as fast as hand-written loops. We can measure this in Stage 6.

### 2.2 Why we implement RoPE from scratch

Libraries such as PyTorch apply RoPE in a few lines of code. Those lines
hide the important ideas: the frequency formula, the pairing of dimensions,
and the reason for the relative-position property.

If we implement each step ourselves, we must understand each step. Also,
RoPE is a good size for a learning project. It is small enough to finish,
but it contains trigonometry, linear algebra, complex numbers, numerical
precision and performance questions.

## 3. Mathematical foundation

**Status: no mathematics is implemented in this entry.**

This entry only defines the **notation**. We define it now, so that all later
entries use the same symbols. The full table is in `docs/mathematics.md`.

The core idea of RoPE, in one sentence: *rotate each pair of embedding
dimensions by an angle that grows with the token position.* In symbols, for
pair `i` at position `m`, the angle is:

```text
φ = m · θ_i
```

- `m` is the position of the token (0, 1, 2, ...).
- `θ_i` is the frequency of pair `i`, in radians per position.
- `φ` is the rotation angle, in radians.

**A small preview example.** Let `θ_0 = 1` radian per position.

| Position `m` | Angle `φ = m · θ_0` | In degrees (approx.) |
|---|---|---|
| 0 | 0 rad | 0° |
| 1 | 1 rad | 57.3° |
| 2 | 2 rad | 114.6° |

The token at position 0 is not rotated. The token at position 2 is rotated
twice as much as the token at position 1.

We will derive the rotation matrix in Stage 2, and the frequency formula
`θ_i = b^(−2i/d)` in Stage 3. We do not skip these derivations.

## 4. Rust concepts learned

### 4.1 Packages, crates and targets

- A **package** is a folder with `Cargo.toml`.
- A **crate** is one compilation unit.
- A **target** is something Cargo can build from a crate: a library, a
  binary, a test, an example or a benchmark.

Our package has several targets. Cargo finds most of them automatically by
their location:

```text
src/lib.rs        → library target
src/main.rs       → binary target
tests/*.rs        → one integration-test target per file
examples/*.rs     → one example target per file
benches/*.rs      → one benchmark target per file
```

**Why this matters:** each integration test, example and benchmark is a
**separate crate**. It can use only the **public** items of the library.
It cannot use `main.rs`. For this reason, we put all logic in the library.

### 4.2 Modules: `mod` and `pub mod`

A module declaration tells the compiler to include another file.

```rust
// In src/lib.rs
pub mod math;
```

The compiler interprets this line as: "Find `src/math.rs` or
`src/math/mod.rs`, compile it, and name its contents `math`." We use the
`mod.rs` form, because each module has submodules in the same folder.

The keyword `pub` makes the module **public**. Without `pub`, only code
inside the crate can use it. Rust items are **private by default**. This
default forces you to decide which items are part of the public interface.

Small independent example:

```rust
mod shapes {
    pub fn area(w: f64, h: f64) -> f64 { w * h } // public
    fn helper() {}                                // private
}

fn main() {
    println!("{}", shapes::area(2.0, 3.0)); // OK: prints 6
    // shapes::helper();                     // Error: `helper` is private
}
```

**Important:** a file that exists, but that no `mod` line declares, is not
compiled. Rust does not find modules by scanning folders. The module tree is
always explicit.

### 4.3 Documentation comments: `//!` and `///`

- `//!` documents the item that **contains** the comment. At the top of a
  file, it documents the module.
- `///` documents the item that **follows** the comment (a function, a
  struct, and so on).

`cargo doc` converts these comments into HTML documentation. Code blocks in
`///` comments become **doc-tests**: `cargo test` compiles and runs them.
In Stage 1 we have zero doc-tests, so the result shows "running 0 tests".

### 4.4 The `env!` macro

`src/main.rs` uses `env!("CARGO_PKG_VERSION")`. A **macro** is code that
generates code at compile time. The `env!` macro reads an environment
variable **when the compiler runs**, not when the program runs. Cargo sets
`CARGO_PKG_NAME` and `CARGO_PKG_VERSION` from `Cargo.toml`. The result is a
`&'static str`: a string slice that lives for the full program. It is stored
in the read-only data of the executable.

### 4.5 `harness = false` for benchmarks

Rust has a built-in benchmark attribute, `#[bench]`. But it needs the
**nightly** compiler (the unstable version). We use the **stable** compiler.

In `Cargo.toml`, `harness = false` tells Cargo: "Do not generate a test
runner for this benchmark. The file has its own `main` function." This keeps
the benchmark on the stable compiler. In Stage 6, we will decide if we
measure with `std::time::Instant` or with a benchmark library.

### 4.6 `use path as _`

The test `library_modules_are_public` contains:

```rust
use rust_rope::{math as _, rope as _, utils as _};
```

`as _` imports a path, but gives it no name. The compiler still checks that
the path exists and is public. Thus, the test fails at compile time if a
module becomes private or is removed. It executes no code.

## 5. Memory and execution

There is almost no runtime behaviour in Stage 1. But we can still analyse it.

**The binary.** `main` calls `println!` two times.

- The format strings are string literals. A string literal has the type
  `&'static str`. The compiler stores it in the executable. **No heap
  allocation** occurs for the literal.
- `println!` takes its arguments **by reference**. It borrows them. It does
  not move them. Thus, the values are still usable after the call.
- `println!` writes to standard output through a lock and a buffer. The
  standard library owns that buffer. Our code does not allocate it.
- When `main` returns, its stack frame is removed. There are no owned values
  to free.

**Empty modules.** An empty module produces no machine code. Modules are a
compile-time naming structure. They have **no runtime cost**.

**Tests.** `cargo test` builds one test executable per test target. It runs
each executable. Each test function runs in its own thread by default.

## 6. Engineering decisions

### 6.1 We separated mathematical operations from RoPE logic

`src/math/` contains general mathematics: complex numbers, trigonometry,
matrices. `src/rope/` contains RoPE-specific logic. The dependency goes in
one direction only: `rope → math`.

Reasons:

1. **Testability.** We can test a 2 × 2 rotation alone, before RoPE uses it.
   If a RoPE test fails, but the `math` tests pass, the error is probably in
   the RoPE logic.
2. **Clarity.** A complex number does not depend on token positions. If we
   mix the two, the code hides which part is general and which part is
   specific.
3. **Learning order.** Stage 2 builds `math`. Stage 3 builds `rope` on top of
   it. The code structure follows the learning structure.

### 6.2 We created individual modules

Each module has **one responsibility** (see `docs/architecture.md`).
Small modules have these advantages:

- A short file is easier to read and to explain.
- Each `explain.md` entry can point to a specific file.
- Privacy rules apply per module. Each module can hide its internal helpers.

We did **not** add traits, generics or abstract interfaces. There is no
code yet that needs them. We add an abstraction only when a real need exists.

### 6.3 We created `CLAUDE.md`

`CLAUDE.md` contains the permanent rules of the project: the writing
standard, the explanation format, the architecture rules, the dependency
rules, the code conventions and the task workflow. The AI assistant reads it
before each task.

Without a permanent rule file, each task can follow different standards.
`CLAUDE.md` makes the quality of explanations consistent across all stages.

### 6.4 ASD-STE100 is our explanation standard

ASD-STE100 (Simplified Technical English) is a controlled writing standard.
The aerospace industry created it for maintenance documentation. In that
industry, an unclear sentence can cause a dangerous error.

The standard uses short sentences, the active voice, one topic per sentence,
and defined terms. These rules help us because:

- Systems programming and mathematics are already difficult. The language
  must not add more difficulty.
- Short sentences make each logical step visible. A hidden step in an
  explanation is often a hidden step in understanding.
- Defined terms prevent ambiguity. For example, "dimension" can mean the
  vector length `d` or one coordinate. We define which meaning we use.

### 6.5 We maintain `explain.md`

Code shows **what** the program does. It does not show **why** we selected
this design, or **what else** we considered. `explain.md` records this
information. It is a learning journal that we can read again later.

We never delete old entries. Old entries show how our understanding
developed. If an entry is wrong, a correction note shows the mistake and the
fix. A visible mistake is also a learning resource.

### 6.6 How this project teaches Rust and mathematics together

Each milestone has a **mathematical part** and a **Rust part**, and they
depend on each other:

| Mathematics topic | Rust topic that it motivates |
|---|---|
| Vectors and dot products (Stage 2) | Slices `&[f32]`, borrowing, iterators |
| Complex numbers (Stage 2) | `struct`, `Copy` types, operator traits (`Mul`, `Add`) |
| Rotation matrices (Stage 2) | Fixed-size arrays on the stack |
| Frequencies `θ_i` (Stage 3) | `Vec<f32>` on the heap, `Result` errors |
| Rotating embeddings (Stage 4) | Mutable borrows `&mut [f32]`, in-place updates |
| Scaling methods (Stage 5) | `enum`, `match`, traits |
| Batched RoPE (Stage 6) | Cache locality, data layout, SIMD |

The mathematics gives a reason for each Rust feature. The Rust code makes
each equation concrete and testable.

### 6.7 Initial dependency decisions

**We added no external dependencies.** `[dependencies]` and
`[dev-dependencies]` are both empty.

Reasons:

1. **The standard library is sufficient for Stage 1.** We need no external
   code to create modules or placeholder tests.
2. **Learning.** If we write the complex number type ourselves, we learn
   operator traits. A library would hide this.
3. **Cost.** Each dependency increases compile time and adds code that we
   must trust.

**Possible future dependencies** (not added; each needs a justification in a
future entry):

| Candidate | Possible purpose | Stage |
|---|---|---|
| `criterion` (dev-dependency) | Statistically reliable benchmarks | 6 |
| `approx` (dev-dependency) | Floating-point comparisons in tests | 7 (we may write our own) |

We will add a dependency only if the standard library solution is clearly
insufficient.

## 7. Alternative approaches

| Decision | Alternative | Why we did not select the alternative |
|---|---|---|
| One package with `lib.rs` + `main.rs` | A Cargo **workspace** with several packages | A workspace adds configuration with no benefit at this size. |
| `mod.rs` files in folders | `src/rope.rs` + `src/rope/` folder (the newer style) | Both work. The `mod.rs` style keeps each module and its submodules in one folder, as the task specifies. |
| `harness = false` with own `main` | `#[bench]` attribute | `#[bench]` needs the nightly compiler. |
| No dependencies | Add `ndarray` or `num-complex` now | They would hide the concepts that we want to learn. |
| Separate `math` and `rope` | One flat module | A flat module mixes general and specific code, and is harder to test. |

## 8. Performance considerations

Stage 1 has no computation. Thus, there is no runtime complexity to analyse.

Relevant facts for later stages:

- **Modules have zero runtime cost.** They exist only at compile time.
- **Debug and release builds are very different.** `cargo build` and
  `cargo test` use the `dev` profile: no optimization, with debug
  information. `cargo bench` and `cargo build --release` use optimization.
  Never measure performance with a debug build.
- **Compile time.** Zero dependencies keep compile time below one second.

## 9. Practical example

Run the binary:

```bash
cargo run
```

Expected output:

```text
rust_rope v0.1.0 — Stage 1: project foundation.
No RoPE algorithms are implemented yet. See README.md for the roadmap.
```

Run the tests:

```bash
cargo test
```

Expected result (summary):

```text
Running tests\numerical_tests.rs
test numerical_test_harness_runs ... ok
Running tests\rope_tests.rs
test library_modules_are_public ... ok
test result: ok.
```

## 10. What we learned

**Rust:**

- A package contains crates. Each crate becomes one or more targets.
- Integration tests, examples and benchmarks are separate crates. They can
  use only the public interface of the library.
- `mod` declarations build an explicit module tree. Items are private by
  default.
- `//!` documents the enclosing item. `///` documents the next item.
- `env!` reads values at compile time.
- `harness = false` lets a benchmark run on the stable compiler.
- String literals are `&'static str` and do not use the heap.

**Mathematics:**

- RoPE rotates pairs of dimensions by the angle `φ = m · θ_i`.
- We defined the notation: `d`, `i`, `m`, `n`, `b`, `θ_i`, `φ`, `q`, `k`.

**Engineering:**

- General mathematics and RoPE-specific logic are separate.
- Permanent rules (`CLAUDE.md`) and a permanent journal (`explain.md`)
  keep the project consistent.

---

## Complete project roadmap

Legend: ✅ done · 🔄 in progress · ⏳ planned

| Stage | Topic | Content | Status |
|---|---|---|---|
| 1 | Project foundation | Initialization, module structure, conventions, documentation | ✅ Done (this entry) |
| 2 | Mathematical foundations | Vectors and matrices, complex numbers, trigonometric functions, 2-D rotations, rotation matrices | ⏳ Planned |
| 3 | RoPE fundamentals | Positional frequencies, inverse frequency calculations, position-dependent angles, rotations of embedding dimensions | ⏳ Planned |
| 4 | RoPE implementation | Pairwise dimension rotations, dimension layouts, embedding transformations, numerical validation | ⏳ Planned |
| 5 | Advanced RoPE | Frequency scaling, extended context, NTK-aware scaling, YaRN, comparison of variants | ⏳ Planned |
| 6 | Performance engineering | Allocation analysis, stack versus heap, cache locality, SIMD, benchmarking, CPU optimization | ⏳ Planned |
| 7 | Validation and experimentation | Unit tests, numerical tests, invariants, reference comparison, performance reports | ⏳ Planned |

**Next stage (Stage 2):** we implement the mathematical primitives in
`src/math/`. We derive the 2-D rotation matrix, prove that it keeps vector
length, connect it to complex multiplication, and implement each primitive
with tests.
