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
| 2 | 2026-10-31 | Stage 2 | Vector operations (`Vector`, dot product, magnitude, `approx_eq`) |
| 3 | 2026-10-31 | Stage 2 | Complex numbers (`Complex`, operator traits, conjugate, phase) |
| 4 | 2026-10-31 | Stage 2 | Trigonometry (radians, unit circle, periodicity, precision) |
| 5 | 2026-10-31 | Stage 2 | 2 × 2 matrices and rotation matrices (`Matrix2`, `R(θ)`) |
| 6 | 2026-10-31 | Stage 2 | Euler's formula and rotation by complex multiplication |
| 7 | 2026-11-01 | Stage 3 | Configuration, validation and inverse frequencies (`RopeConfig`, `RopeError`, `θⱼ = b^(−2j/d)`) |
| 8 | 2026-11-01 | Stage 3 | Position indexing (`Position`, `PositionRange`, the `2^53` limit) |
| 9 | 2026-11-01 | Stage 3 | Positional angles `φ(m, j) = m·θⱼ` and the `AngleTable`; link to `R(m, j)` |
| 10 | 2026-11-01 | Stage 4 | The RoPE transformation of a vector (`rotate_pair`, `RotaryEmbedding`, layouts) |
| 11 | 2026-11-01 | Stage 5 | Attention tensor shapes `[B, S, H, D]` and the flat `Tensor4` |
| 12 | 2026-11-01 | Stage 5 | RoPE on query and key tensors; values unchanged; relative position in scores |
| 13 | 2026-11-01 | Stage 6 | Independent PyTorch reference validation; IEEE 754 and error analysis |
| 14 | 2026-11-01 | Stage 7 | Interleaved and split-half layouts (`RopeLayout`) |
| 15 | 2026-11-01 | Stage 7 | Partial RoPE, tensor integration, Stage 7 validation results |
| 16 | 2026-11-01 | Stage 8 | RoPE scaling: position interpolation, linear, NTK-aware, YaRN |

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


---

# Entry 2 — Vector operations (Stage 2, Component 1)

**Date:** 2026-10-31
**Stage:** 2 (Mathematical foundations)
**Files:** `src/math/matrix.rs` (the `Vector` part), `src/math/mod.rs`
(`approx_eq`)

### Implementation status of this entry

| Item | Status |
|---|---|
| `Vector` struct with `new`, `from_slice`, `dimension`, `is_empty`, `components`, `into_components` | ✅ Implemented |
| `checked_add`, `checked_sub`, `dot`, `magnitude` | ✅ Implemented |
| `DimensionMismatch` error type | ✅ Implemented |
| `approx_eq` and `DEFAULT_TOLERANCE` (shared float comparison) | ✅ Implemented |
| Scalar multiplication, normalization | ⏳ Not needed yet. Not implemented. |
| Overflow-safe magnitude | ⏳ Discussed in section 8. Not implemented. |

---

## 1. What did we build?

We built a type `Vector`. It represents a list of `n` real numbers. With it,
we can:

- create a vector from a list of numbers,
- read its dimension `n`,
- add and subtract two vectors,
- calculate the dot product of two vectors,
- calculate the length (magnitude) of a vector.

If two vectors have different dimensions, addition, subtraction and the dot
product return an error of type `DimensionMismatch`.

We also built `approx_eq`. It compares two floating-point numbers with a
tolerance. All tests in the project use it.

## 2. Why did we build it?

In a Transformer, every token is a vector. The vectors that RoPE changes are
the **query** vector `q` and the **key** vector `k`. Attention compares them
with the dot product `q · k`. A large dot product means "this key is
relevant to this query".

RoPE rotates `q` and `k` before the dot product. Thus, to understand RoPE,
we must first understand:

- what a vector is,
- what the dot product measures,
- what the length of a vector is (because a rotation does not change it).

## 3. Mathematical foundation

### 3.1 What is a vector?

A **vector** is an ordered list of numbers:

```text
v = [v₁, v₂, ..., vₙ]
```

- `v` is the name of the vector.
- `vᵢ` (read "v sub i") is the component at position `i`. The small number
  below the letter is the **index**.
- `n` is the **dimension**: the number of components.

**Example.** `v = [3, 4]` has dimension `n = 2`. `v₁ = 3` and `v₂ = 4`.

**Geometric interpretation.** In two dimensions, `[3, 4]` is an arrow. It
starts at the origin `(0, 0)`. It ends at the point `(3, 4)`. In higher
dimensions we cannot draw the arrow, but the same rules apply.

**Note on indices.** Mathematics usually counts from 1 (`v₁` is first).
Rust counts from 0 (`v[0]` is first). Thus, `vᵢ` in an equation is
`v[i - 1]` in Rust. In later stages, we write equations with indices that
start at 0, to match the code.

### 3.2 Addition and subtraction

We add two vectors component by component:

```text
a + b = [a₁ + b₁, a₂ + b₂, ..., aₙ + bₙ]
a − b = [a₁ − b₁, a₂ − b₂, ..., aₙ − bₙ]
```

This definition needs **the same dimension `n`** for `a` and `b`. If `a` has
3 components and `b` has 2, the third sum `a₃ + b₃` does not exist. For this
reason, the operation is **not defined**, and our code returns an error.

**Geometric interpretation.** To calculate `a + b`, put the start of arrow
`b` at the end of arrow `a`. The sum goes from the start of `a` to the end
of `b`. The difference `a − b` is the arrow from the end of `b` to the end
of `a`.

**Numerical example.**

```text
a = [1, 2, 3],  b = [4, −5, −3]
a + b = [1 + 4, 2 + (−5), 3 + (−3)] = [5, −3, 0]
```

### 3.3 The dot product

The **dot product** of `a` and `b` is one number:

```text
a · b = Σᵢ aᵢ·bᵢ = a₁·b₁ + a₂·b₂ + ... + aₙ·bₙ
```

The symbol `Σ` (the Greek capital letter "sigma") means "sum". `Σᵢ` means
"calculate the expression for each index `i`, and add the results".

**Numerical example.**

```text
a = [1, 2, 3],  b = [4, 5, 6]
a · b = 1·4 + 2·5 + 3·6 = 4 + 10 + 18 = 32
```

**Geometric meaning.** The dot product also has this form:

```text
a · b = ‖a‖ · ‖b‖ · cos(α)
```

- `‖a‖` is the length of `a` (section 3.4).
- `α` (the Greek letter "alpha") is the angle between the two arrows.

**Derivation of the geometric form (two dimensions).** Write each vector
with its length and its direction angle. Let `a` have length `r` and angle
`φ`. Let `b` have length `s` and angle `ψ`. On the unit circle (Entry 4), a
direction with angle `φ` is the point `(cos(φ), sin(φ))`. Thus:

```text
a = [r·cos(φ), r·sin(φ)],   b = [s·cos(ψ), s·sin(ψ)]
a · b = r·s·(cos(φ)·cos(ψ) + sin(φ)·sin(ψ))
```

The angle-difference identity is
`cos(ψ − φ) = cos(ψ)·cos(φ) + sin(ψ)·sin(φ)`. Thus:

```text
a · b = r·s·cos(ψ − φ)
```

The angle between the arrows is `α = ψ − φ`. This proves the formula in
two dimensions. The same result is true in all dimensions.

**What this tells us:**

| Angle `α` | `cos(α)` | Dot product | Meaning |
|---|---|---|---|
| 0° | 1 | largest positive value | same direction |
| 90° | 0 | 0 | perpendicular ("orthogonal") |
| 180° | −1 | largest negative value | opposite directions |

**Example.** `[1, 0] · [0, 1] = 1·0 + 0·1 = 0`. The x-axis and the y-axis
are perpendicular.

**Why this is important for RoPE.** Attention uses `q · k`. The dot product
depends on the lengths **and the angle** between `q` and `k`. RoPE changes
only the angle, by rotation. Thus, RoPE changes the attention score in a
controlled way. Entry 5 and Stage 3 show this in detail.

### 3.4 The magnitude (Euclidean norm)

The **magnitude** (also: length, Euclidean norm) of `v` is:

```text
‖v‖ = √(Σᵢ vᵢ²) = √(v₁² + v₂² + ... + vₙ²)
```

The double bars `‖ ‖` mean "length of". `√` is the square root.

**Derivation (two dimensions).** The vector `[x, y]` is the hypotenuse
(the longest side) of a right triangle. The other two sides have the
lengths `x` and `y`. The theorem of Pythagoras says:

```text
length² = x² + y²   →   length = √(x² + y²)
```

In three dimensions, apply the theorem two times. First, the diagonal of
the floor is `√(x² + y²)`. Then this diagonal and the height `z` make a new
right triangle:

```text
length = √( (√(x² + y²))² + z² ) = √(x² + y² + z²)
```

Each new dimension adds one more square under the root.

**Link to the dot product.** `v · v = Σᵢ vᵢ·vᵢ = Σᵢ vᵢ² = ‖v‖²`.
Thus, `‖v‖ = √(v · v)`. Our test
`dot_product_with_itself_is_magnitude_squared` checks this identity.

**Numerical example.** `‖[3, 4]‖ = √(9 + 16) = √25 = 5`.
`‖[−3, −4]‖ = 5` too: squares remove the signs.

### 3.5 Floating-point precision

A computer stores a real number as an **`f64`**: a 64-bit binary
floating-point number (IEEE 754 "double precision"). It has 53 binary
digits of precision. This is about 15 to 17 decimal digits.

Many decimal numbers have no exact binary form. Example: `0.1` in binary is
`0.000110011001100...` with no end. The computer rounds it. Thus:

```text
0.1 + 0.2 = 0.30000000000000004   (in f64)
```

Our test `floating_point_addition_needs_a_tolerance` shows this.

**Consequence:** do not compare calculated floating-point values with `==`.
Compare them with a tolerance. Section 6.2 explains our `approx_eq` helper.

**Where exact comparison is correct.** Some results are exact in binary.
Small integers have an exact `f64` form. IEEE 754 requires that `+`, `−`,
`×`, `÷` and `√` give the correctly rounded result. Thus `3² + 4² = 25`
exactly, `√25 = 5` exactly, and the test for `‖[3, 4]‖` correctly uses `==`.

## 4. Rust concepts learned

### 4.1 Structs

A **struct** groups named values (fields) into one new type.

```rust
pub struct Vector {
    components: Vec<f64>,
}
```

The field `components` has no `pub`. Thus, it is **private**. Code outside
the `matrix` module cannot read it or change it directly. It must use the
methods.

**Why a private field?** A wrapper type with one private field is a
**newtype**. It lets us control all operations. Later, we can change the
internal storage without breaking the code that uses `Vector`.

Small independent example:

```rust
mod bank {
    pub struct Account { balance: i64 }      // private field
    impl Account {
        pub fn new() -> Self { Account { balance: 0 } }
        pub fn balance(&self) -> i64 { self.balance }
    }
}
// let a = bank::Account { balance: 100 }; // Error: the field is private
let a = bank::Account::new();               // OK
```

### 4.2 `Vec<f64>`

`Vec<T>` is the standard growable array. `Vec<f64>` holds `f64` values.

A `Vec` has two parts:

1. **On the stack:** three machine words. On a 64-bit computer this is
   3 × 8 = 24 bytes: a pointer to the data, the **capacity** (the number of
   slots allocated) and the **length** (the number of slots in use).
2. **On the heap:** the data itself: `capacity × 8` bytes, one after the
   other ("contiguous").

```text
 Stack (Vector)                Heap
┌──────────────┐          ┌─────┬─────┬─────┐
│ ptr      ────┼────────► │ 1.0 │ 2.0 │ 3.0 │
│ capacity = 3 │          └─────┴─────┴─────┘
│ length   = 3 │             8 B   8 B   8 B
└──────────────┘
```

`Vector` has only one field. Thus, `Vector` has exactly the same size as
`Vec<f64>`: 24 bytes. The wrapper costs nothing.

**Why `f64` and not `f32`?** `f64` has about 16 decimal digits. `f32` has
about 7. In Stage 2 we learn the mathematics, and we want small errors.
Real models often use `f32` or smaller types for speed. We study this in
Stage 6.

### 4.3 Associated functions and methods

Code in an `impl Vector { ... }` block belongs to the type.

- An **associated function** has no `self` parameter. You call it with
  `Type::name(...)`. Constructors are associated functions:
  `Vector::new(vec![1.0])`.
- A **method** has a `self` parameter. You call it with `value.name(...)`:
  `v.magnitude()`.

The form of `self` decides the ownership:

| Parameter | Meaning | Example in our code |
|---|---|---|
| `&self` | borrow, read only | `dimension`, `dot`, `magnitude` |
| `&mut self` | borrow, read and change | not used yet |
| `self` | take ownership (move) | `into_components` |

`Self` (capital S) is an alias for the type of the `impl` block. Here,
`Self` means `Vector`.

### 4.4 Ownership, moves and borrows

**Ownership rules:**

1. Each value has one owner.
2. When the owner goes out of scope, Rust drops the value (frees its memory).
3. Assignment or a function call **moves** the value, unless the type is
   `Copy` (Entry 3) or you pass a reference.

**`new` moves:**

```rust
let data = vec![1.0, 2.0];
let v = Vector::new(data);     // `data` moves into `v`
// println!("{:?}", data);     // Error: value used after move
```

The move copies only the 24 stack bytes (pointer, capacity, length). The
heap data stays where it is. No allocation occurs. A move is cheap.

**`from_slice` borrows:**

```rust
let data = [1.0, 2.0];
let v = Vector::from_slice(&data);   // `&data` is a borrow
println!("{:?}", data);              // OK: we still own `data`
```

`&[f64]` is a **slice**: a reference to a sequence of `f64` values. It has
two words: a pointer and a length. `values.to_vec()` allocates a new heap
buffer and copies the values. The caller keeps the original.

**`checked_add` borrows both operands:**

```rust
pub fn checked_add(&self, other: &Vector) -> Result<Vector, DimensionMismatch>
```

Both inputs are references. Thus, the caller can use `a` and `b` again
after `a.checked_add(&b)`. The result is a new `Vector` that the caller
owns.

**`into_components` moves `self`:**

```rust
let v = Vector::new(vec![7.0, 8.0]);
let raw: Vec<f64> = v.into_components();  // `v` is consumed
// v.dimension();                          // Error: `v` was moved
```

In Rust, the name prefix `into_` means "consume `self` and convert it".

**Mutable references (`&mut`).** A mutable reference allows a change. Rust
permits **either** one `&mut` reference **or** any number of `&` references
at the same time, never both. This rule prevents data races and the use of
invalid pointers. We do not need `&mut` yet, because all Component 1
operations return new values. Stage 4 uses `&mut [f64]` to rotate
embeddings in place.

Small independent example:

```rust
let mut v = vec![1.0, 2.0];
let r = &mut v;
r.push(3.0);            // OK: a change through the mutable reference
// let s = &v;          // Error if `r` is used after this line
println!("{:?}", v);    // OK: `r` is no longer used
```

### 4.5 Iterators

`magnitude` uses an **iterator chain**:

```rust
self.components.iter().map(|x| x * x).sum::<f64>().sqrt()
```

Step by step:

1. `.iter()` creates an iterator. It yields `&f64` (a reference to each
   component). It does not copy the vector.
2. `.map(|x| x * x)` applies a **closure** (an anonymous function) to each
   element. `|x| x * x` takes `x` and returns `x * x`. The standard library
   implements `&f64 * &f64`, so this works with references.
3. `.sum::<f64>()` adds all values. `::<f64>` (the "turbofish") tells the
   compiler the type of the result. Without it, the compiler cannot select
   the type for `.sqrt()`.
4. `.sqrt()` calculates the square root.

`dot` and `checked_add` use `.zip(...)`. `zip` combines two iterators into
one iterator of pairs `(aᵢ, bᵢ)`. This is a direct translation of
`Σᵢ aᵢ·bᵢ`.

`checked_add` ends with `.collect()`. `collect` puts the iterator results
into a collection. The compiler finds the collection type from the field
type: `components` is `Vec<f64>`, so `collect` creates a `Vec<f64>`.

**Iterators are lazy.** `map` and `zip` do not calculate anything when you
call them. The calculation happens when `sum` or `collect` consumes the
iterator. The compiler then usually converts the chain into one simple
loop. This is a **zero-cost abstraction**.

**Why not an index loop?** In a loop `for i in 0..n { a[i] + b[i] }`, each
`a[i]` has a bounds check (it panics if `i` is too large). The `zip` version
cannot go out of bounds, so the compiler does not need bounds checks.

### 4.6 Error handling with `Result`

`Result<T, E>` is an `enum` with two variants:

```rust
enum Result<T, E> {
    Ok(T),   // success, contains the value
    Err(E),  // failure, contains the error
}
```

`dot` returns `Result<f64, DimensionMismatch>`. The caller must handle both
cases. The compiler shows a warning if the caller ignores a `Result`.

**The `?` operator.** In `checked_add`:

```rust
self.check_same_dimension(other)?;
```

If the result is `Err(e)`, `?` returns `Err(e)` from `checked_add`
immediately. If the result is `Ok(())`, execution continues. `()` (the
"unit" type) means "no value".

**Why `Result` and not `panic!`?** A dimension mismatch is an error that
the **caller** causes. The caller can correct it. `panic!` stops the
thread. It is for bugs in our own code. `CLAUDE.md` section 6 states this
rule.

**The error type.** `DimensionMismatch` contains the two dimensions. Thus,
the error message can tell the user exactly what is wrong. It implements:

- `Debug` (with `#[derive]`): print with `{:?}`. `assert_eq!` needs it.
- `Display` (by hand): a readable message with `{}` or `.to_string()`.
- `std::error::Error`: the standard error trait. Other error-handling code
  can accept our error.
- `Copy`, `Clone`, `PartialEq`, `Eq`: the error is two `usize` values.
  It is cheap to copy and easy to compare in tests.

### 4.7 `#[derive]`

`#[derive(Debug, Clone, PartialEq)]` tells the compiler to generate trait
implementations automatically.

- `Debug`: print the struct for debugging.
- `Clone`: `v.clone()` makes a **deep copy**: a new heap buffer with the
  same values.
- `PartialEq`: `a == b` compares all components.

`Vector` is **not** `Copy`. A `Copy` type must be copyable bit by bit. A
bitwise copy of a `Vec` would create two owners of the same heap buffer.
Both would free it, and the program would free the memory two times. Rust
forbids `Copy` for types that own heap memory.

### 4.8 Unit tests and `#[cfg(test)]`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn name() { /* ... */ }
}
```

- `#[cfg(test)]` compiles this module **only** for `cargo test`. The test
  code is not in the normal build.
- `use super::*;` imports all items of the parent module, also the private
  items. Thus, unit tests can test private functions.
- `#[test]` marks a test function.

## 5. Memory and execution

**Example call:** `let s = a.checked_add(&b)?;` with `a = [1, 2, 3]` and
`b = [4, 5, 6]`.

1. `a` and `b` each are 24 bytes on the stack. Each points to 24 bytes of
   heap data.
2. The call passes `&a` (as `self`) and `&b`. Each reference is 8 bytes.
   No vector data is copied.
3. `check_same_dimension` reads the two lengths. Both are 3. It returns
   `Ok(())`. No allocation occurs.
4. `iter().zip().map()` creates a small iterator struct **on the stack**.
5. `collect()` allocates **one** heap buffer. The iterator knows its exact
   length (3), so `collect` allocates the correct capacity one time.
6. The new `Vec` moves into a new `Vector`, then into `Ok(...)`, then to the
   caller. Each step is a move of 24 bytes. The heap buffer does not move.
7. `a` and `b` are still valid. When the caller's scope ends, Rust drops
   `s`, `b` and `a` in the reverse order of declaration. Each drop frees one
   heap buffer.

**Memory summary of each operation:**

| Operation | Heap allocations | Ownership |
|---|---|---|
| `Vector::new(vec)` | 0 (reuses the `Vec` buffer) | moves the `Vec` in |
| `Vector::from_slice(&s)` | 1 | borrows `s`, copies |
| `dimension`, `components` | 0 | borrows |
| `checked_add`, `checked_sub` | 1 (the result) | borrows both |
| `dot`, `magnitude` | 0 | borrows |
| `into_components` | 0 | moves `self` out |
| `clone` | 1 | borrows, deep copy |

## 6. Engineering decisions

### 6.1 Named methods `checked_add` and `checked_sub`, not the `+` operator

We could implement the `Add` trait so that `&a + &b` works. But `Add::add`
must return one output type. There are two choices:

- Return `Vector` and **panic** on a dimension mismatch. This breaks our
  error rule.
- Return `Result<Vector, _>`. Then `&a + &b` has the type `Result`, and
  `(&a + &b) + &c` does not compile. This is surprising for an operator.

Thus, we use named methods. The prefix `checked_` follows the standard
library convention: `i32::checked_add` returns `None` on overflow instead
of a panic. Complex numbers (Entry 3) cannot fail, so they use real
operators.

### 6.2 A shared `approx_eq` helper in `src/math/mod.rs`

Many tests compare floating-point values. One shared helper:

- gives all tests the same, documented rule,
- is public, so the integration tests in `tests/` can use it too.

We put it in `math`, because it is a general numerical tool, and `math`
must not depend on other modules.

The rule:

```text
|a − b| ≤ tolerance × max(1, |a|, |b|)
```

- **Absolute** near zero. This is necessary, because `cos(π/2)` in `f64` is
  `6.1e-17`, not `0`. A pure relative test against 0 would never pass.
- **Relative** for large values, because floating-point errors grow with
  the size of the values.

**A bug that the tests found.** The first version returned `true` for
`approx_eq(+∞, −∞, …)`. The reason: `|+∞ − (−∞)| = ∞` and `scale = ∞`, so
the test was `∞ ≤ ∞`, which is true. We added an explicit check: two
infinities that are not exactly equal are different, and an infinity is
different from every finite number. The test `special_values` now covers
these cases. This shows why we test edge cases.

**Why `1e-12`?** See the comment on `DEFAULT_TOLERANCE`. In short: it is
about 4500 × `f64::EPSILON`. Rounding errors of short calculations are
smaller. Real mistakes are much larger.

### 6.3 A struct with a private field, not a bare `Vec<f64>`

A bare `Vec<f64>` would work. But then any code could push values and
change the dimension. The `Vector` type names the concept, keeps its
methods together, and costs zero extra bytes.

### 6.4 `DimensionMismatch` is in `matrix.rs`

The error belongs to the vector operations. `utils/validation.rs` is for
RoPE configuration validation (Stage 3). If we put the error there, `math`
would depend on `utils`. We prefer that `math` stays independent.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage | Decision |
|---|---|---|---|
| Fixed-size array `[f64; N]` with const generics (`Vector<const N: usize>`) | No heap. The compiler checks the dimensions, so no runtime error | `N` must be known at compile time. RoPE reads `d` from a configuration | Not for general vectors. We use `[f64; 2]` for 2-D (Entry 5) |
| Free functions on slices `&[f64]` | No wrapper type. Works with any data | No type for the concept. Dimension rules are spread out | Possible in Stage 4 for in-place work |
| `Add` trait with panic | Short syntax `&a + &b` | Panics on caller errors | Rejected |
| External crate (`ndarray`, `nalgebra`) | Complete and fast | Hides the concepts we want to learn. Adds a dependency | Rejected (project rule) |
| Generic over `f32`/`f64` (`Vector<T>`) | One code for both types | Needs a numeric trait (not in the standard library) or much extra code | Not yet. Maybe Stage 6 |

## 8. Performance considerations

**Complexity.** For dimension `n`:

| Operation | Time | Extra memory |
|---|---|---|
| `dimension` | O(1) | 0 |
| `checked_add`, `checked_sub` | O(n) | n × 8 bytes (the result) |
| `dot`, `magnitude` | O(n) | 0 |

"O(n)" means: the time grows in proportion to `n`. "O(1)" means: the time
does not depend on `n`.

**Cache locality.** The components are contiguous in memory. The processor
reads memory in blocks of 64 bytes ("cache lines"). One block contains 8
`f64` values. A sequential pass is the fastest memory access pattern.

**SIMD.** SIMD (Single Instruction, Multiple Data) processes several numbers
with one instruction. The compiler can automatically use SIMD for
`checked_add`. For `dot` and `magnitude`, it usually **cannot**. The reason:
the sum must add the values in the exact order of the source code, because
floating-point addition is not associative. `(a + b) + c` can be different
from `a + (b + c)`. Stage 6 studies this.

**Overflow.** `magnitude` squares each component. If a component is larger
than about `1.34e154`, its square is larger than `f64::MAX ≈ 1.8e308` and
becomes infinity. The test `magnitude_overflows_for_very_large_components`
documents this. A safe method first divides by the largest absolute
component, then multiplies the result back. It is slower. Embedding values
are usually smaller than 100, so we use the simple formula. For 2-D, the
standard library has `f64::hypot`, which is overflow-safe. Entry 3 uses it.

**Allocation.** `checked_add` allocates for each result. In a loop with
many additions, this is slow. A later version can add an in-place method
(`&mut self`) if a real need exists.

## 9. Practical example

```rust
use rust_rope::math::matrix::Vector;

let a = Vector::new(vec![1.0, 2.0, 3.0]);
let b = Vector::new(vec![4.0, 5.0, 6.0]);

println!("{:?}", a.checked_add(&b).unwrap().components());
println!("{:?}", a.checked_sub(&b).unwrap().components());
println!("{:?}", a.dot(&b));
println!("{}", Vector::new(vec![3.0, 4.0]).magnitude());
println!("{}", a.dot(&Vector::new(vec![1.0, 2.0])).unwrap_err());
```

Expected output:

```text
[5.0, 7.0, 9.0]
[-3.0, -3.0, -3.0]
Ok(32.0)
5
dimension mismatch: the left vector has 3 components, the right vector has 2
```

`examples/math_foundations.rs` contains a runnable version (Entry 6).

## 10. What we learned

**Mathematics:**

- A vector is an ordered list of `n` numbers. It is also an arrow.
- Addition and subtraction work component by component. They need equal
  dimensions.
- The dot product `Σ aᵢbᵢ` equals `‖a‖‖b‖cos(α)`. It measures length and
  alignment. Attention uses it.
- The magnitude `√(Σ vᵢ²)` is the theorem of Pythagoras in `n` dimensions.
  `‖v‖² = v · v`.
- `f64` values are rounded. Compare them with a tolerance.

**Rust:**

- A struct with a private field (a newtype) controls all access.
- `Vec<f64>`: 24 bytes on the stack, the data on the heap.
- `&self` borrows, `self` moves. A move of a `Vec` copies 24 bytes, not the
  data.
- Iterators (`iter`, `zip`, `map`, `sum`, `collect`) are lazy. They compile
  to simple loops.
- `Result` and `?` handle errors that the caller causes.
- `#[derive]` generates `Debug`, `Clone` and `PartialEq`. A type that owns
  heap memory cannot be `Copy`.
- Edge-case tests find real bugs (the infinity bug in `approx_eq`).


---

# Entry 3 — Complex numbers (Stage 2, Component 2)

**Date:** 2026-10-31
**Stage:** 2 (Mathematical foundations)
**File:** `src/math/complex.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `Complex` struct, `new`, `re`, `im`, constants `ZERO`, `ONE`, `I` | ✅ Implemented |
| `+`, `−`, `×` with the `Add`, `Sub`, `Mul` traits | ✅ Implemented |
| `conjugate`, `magnitude`, `phase` | ✅ Implemented |
| Polar form and rotation (Euler's formula) | ⏳ Entry 6 (Component 5) |
| Division, negation, scalar multiplication | ⏳ Not needed. Not implemented. |

---

## 1. What did we build?

We built a type `Complex`. It represents a complex number `z = a + b·i`.
We can:

- create a complex number from its two components,
- read the real component `a` and the imaginary component `b`,
- add, subtract and multiply complex numbers with `+`, `−` and `*`,
- calculate the conjugate, the magnitude and the phase.

## 2. Why did we build it?

RoPE rotates pairs of numbers. There are two equivalent ways to describe
such a rotation:

1. multiplication by a 2 × 2 rotation matrix (Entry 5),
2. multiplication by a complex number of length 1 (Entry 6).

The RoFormer paper uses the complex form to prove the most important
property of RoPE: the attention score depends only on the **relative**
position of two tokens. Many implementations (for example the original
LLaMA code) also use complex numbers. Thus, we must understand complex
numbers before we implement RoPE.

## 3. Mathematical foundation

### 3.1 What is a complex number?

The equation `x² = −1` has no real solution, because the square of a real
number is never negative. Mathematicians define a new number `i` with this
property:

```text
i² = −1
```

`i` is the **imaginary unit**. A **complex number** is a number of this form:

```text
z = a + b·i
```

- `a` is a real number. It is the **real component** (`Re(z)`).
- `b` is a real number. It is the **imaginary component** (`Im(z)`).
  Note: the imaginary component is `b`, not `b·i`.

**Examples.** `3 + 4i` has `a = 3`, `b = 4`. The real number `2` is
`2 + 0i`. The number `i` is `0 + 1i`.

### 3.2 Why two floating-point values are sufficient

Two complex numbers are equal **only if** their real components are equal
**and** their imaginary components are equal. Thus, the pair `(a, b)`
identifies each complex number exactly. There is no other information.

This gives a picture: the **complex plane**. The horizontal axis is the
real axis. The vertical axis is the imaginary axis. The number `a + b·i` is
the point `(a, b)`.

```text
          Im
           │
         4 ┤        • 3 + 4i  = point (3, 4)
           │
           │
───────────┼──────────── Re
           0        3
```

`a` and `b` are real numbers. A computer approximates a real number with an
`f64`. Thus, a complex number is two `f64` values: 16 bytes.

### 3.3 Addition and subtraction

Add the real parts and the imaginary parts separately:

```text
(a + bi) + (c + di) = (a + c) + (b + d)i
(a + bi) − (c + di) = (a − c) + (b − d)i
```

**Derivation.** Use the normal rules of algebra. Put the terms without `i`
together and the terms with `i` together:
`a + bi + c + di = (a + c) + (b + d)i`.

**Geometric interpretation.** This is exactly vector addition of the points
`(a, b)` and `(c, d)` (Entry 2).

**Numerical example.** `(1 + 2i) + (3 − 5i) = 4 − 3i`.

### 3.4 Multiplication

```text
(a + bi)(c + di) = (ac − bd) + (ad + bc)i
```

**Derivation.** Multiply each term of the first factor by each term of the
second factor (the distributive law):

```text
(a + bi)(c + di) = a·c + a·di + bi·c + bi·di
                 = ac + adi + bci + bd·i²
```

Now use `i² = −1`. Thus `bd·i² = −bd`:

```text
                 = ac + adi + bci − bd
                 = (ac − bd) + (ad + bc)i
```

**Numerical example.**

```text
(1 + 2i)(3 + 4i):  a = 1, b = 2, c = 3, d = 4
real part:      ac − bd = 1·3 − 2·4 = 3 − 8  = −5
imaginary part: ad + bc = 1·4 + 2·3 = 4 + 6  = 10
result:         −5 + 10i
```

**A preview of rotation.** Multiply `1` by `i`: `i·1 = i`. The point `(1, 0)`
moves to `(0, 1)`. Multiply again: `i·i = −1`. The point moves to `(−1, 0)`.
Each multiplication by `i` turns the point by 90° around the origin. The
test `multiplication_by_i_rotates_by_a_quarter_turn` checks this. Entry 6
generalizes this to any angle.

### 3.5 The conjugate

```text
z̄ = a − b·i     (for z = a + b·i)
```

The bar over `z` means "conjugate". The conjugate changes the sign of the
imaginary component.

**Geometric interpretation.** The conjugate reflects the point `(a, b)`
across the real axis, to `(a, −b)`. In terms of angles, it changes the angle
`φ` to `−φ`. It reverses a rotation.

**Important identity.**

```text
z · z̄ = (a + bi)(a − bi) = a² − abi + abi − b²i² = a² + b²
```

The result is a real number: the square of the length. The test
`z_times_conjugate_is_magnitude_squared` checks `(3 + 4i)(3 − 4i) = 25`.

**Where RoPE uses the conjugate.** The RoFormer paper writes the attention
score as `Re[q_m · conj(k_n)]`. The conjugate of the key reverses the key's
rotation. Thus, only the difference of the two rotation angles remains. We
derive this in Stage 3.

### 3.6 The magnitude

```text
|z| = √(a² + b²)
```

The vertical bars `| |` mean "magnitude" (also: modulus, absolute value).

**Derivation.** The point `(a, b)` is the end of a right triangle with sides
`a` and `b`. The theorem of Pythagoras gives the distance from the origin:
`√(a² + b²)`. This is the 2-D Euclidean norm of Entry 2. From 3.5:
`|z| = √(z · z̄)`.

**Numerical example.** `|3 + 4i| = √(9 + 16) = 5`.

**Property.** `|z·w| = |z|·|w|`. The magnitudes multiply.
Example: `|3 + 4i| = 5` and `|5 + 12i| = 13`, so `|(3 + 4i)(5 + 12i)| = 65`.
The test `magnitudes_multiply` checks this. **Consequence for RoPE:** if
`|w| = 1`, multiplication by `w` does not change the length of `z`.

### 3.7 The phase (argument)

The **phase** `arg(z)` is the angle from the positive real axis to the
arrow from `0` to `z`. It is measured in **radians** (Entry 4) and
counterclockwise. The symbol is often `φ` (the Greek letter "phi").

From the unit circle (Entry 4), a point with distance `r` and angle `φ` is:

```text
a = r·cos(φ),   b = r·sin(φ)
```

Thus `b / a = sin(φ) / cos(φ) = tan(φ)`, and a first idea is
`φ = arctan(b / a)`. But this idea fails:

- `1 + i` and `−1 − i` both have `b / a = 1`. `arctan(1) = π/4` for both.
  But `−1 − i` is in the opposite direction. Its phase is `−3π/4`.
- For `a = 0`, the division `b / a` is not defined.

The function **`atan2(b, a)`** solves both problems. It uses the signs of
`a` and `b` separately to find the correct quadrant. It returns a value in
the range `[−π, π]`. In Rust the call is `b.atan2(a)`: note the order, `b`
first.

| `z` | `atan(b/a)` | `atan2(b, a)` = correct phase |
|---|---|---|
| `1 + i` | π/4 | π/4 (45°) |
| `−1 − i` | π/4 (wrong) | −3π/4 (−135°) |
| `0 + i` | undefined | π/2 (90°) |
| `−1 + 0i` | −0 (wrong) | π (180°) |

**The phase of zero.** `0 + 0i` has no direction. Its phase is not defined.
`atan2(0, 0)` returns `0` by convention. Our test
`phase_of_zero_is_zero_by_convention` documents this.

**Property.** `arg(z·w) = arg(z) + arg(w)`. The phases add (if the sum stays
in `[−π, π]`; else the result wraps around by `2π`). Example:
`arg(1 + i) = π/4`, `arg(2i) = π/2`, and `(1 + i)(2i) = −2 + 2i` has phase
`3π/4`. The test `phases_add_under_multiplication` checks this.

**Summary: multiplication multiplies lengths and adds angles.** This is the
key to Entry 6.

### 3.8 Precision considerations

- `+`, `−`, `×` on `f64` are correctly rounded. Results with small integers
  are exact, so many tests correctly use `==`.
- A product `ac − bd` can lose precision when `ac` and `bd` are almost
  equal. The subtraction removes the equal leading digits
  ("cancellation"), and the remaining digits contain the rounding errors.
- `hypot(a, b)` avoids overflow and underflow. The naive `√(a² + b²)`
  overflows when `a² > f64::MAX`, for example `a = 1e200`. The test
  `magnitude_does_not_overflow_for_large_components` checks this.
- `atan2` is accurate to about one unit in the last place, but `π/4` and
  other results are irrational. Thus phase tests use `approx_eq`.

## 4. Rust concepts learned

### 4.1 `impl` blocks, `const` items and `const fn`

```rust
impl Complex {
    pub const I: Complex = Complex::new(0.0, 1.0);
    pub const fn new(re: f64, im: f64) -> Self { Self { re, im } }
}
```

- An **associated constant** (`pub const I`) belongs to the type. You use it
  as `Complex::I`.
- A **`const fn`** can run at compile time. The compiler calculates
  `Complex::new(0.0, 1.0)` during compilation and puts the result directly
  into the program. A constant must have a value at compile time, so it can
  only call `const fn` functions.
- `Self { re, im }` is the **field init shorthand**. It means
  `Self { re: re, im: im }`.

`magnitude` and `phase` are not `const fn`, because `hypot` and `atan2` are
not `const` in the standard library.

### 4.2 `Copy` and `Clone`

```rust
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex { re: f64, im: f64 }
```

- **`Clone`** gives an explicit `.clone()` method. A clone can do any work,
  for example allocate heap memory (as `Vector::clone` does).
- **`Copy`** is a **marker trait**. It has no methods. It tells the
  compiler: "a bitwise copy of this value is a complete, independent copy".
  Then an assignment **copies** the value instead of a move. The original
  stays usable.

```rust
let z = Complex::new(1.0, 2.0);
let w = z;           // copy (Complex is Copy)
println!("{z:?}");   // OK

let v = vec![1.0];
let u = v;           // move (Vec is not Copy)
// println!("{v:?}"); // Error: `v` was moved
```

**Rule.** A type can be `Copy` only if all its fields are `Copy` and it has
no `Drop` code. `f64` is `Copy`, so `Complex` can be `Copy`. `Vec` is not
`Copy`, so `Vector` cannot be `Copy` (Entry 2, section 4.7). Every `Copy`
type must also be `Clone`; this is why we derive both.

**Why `Copy` for `Complex`?** It is 16 bytes, like two `u64` values. It is
cheaper to copy 16 bytes than to manage a reference to them. With `Copy`,
we can write `z * w` and still use `z` and `w` afterwards.

**`Default`** gives `Complex::default()`, which is `0 + 0i`, because the
default `f64` is `0.0`.

### 4.3 Methods that take `self` by value

```rust
pub const fn conjugate(self) -> Self
pub fn magnitude(self) -> f64
```

These methods take `self`, not `&self`. For a non-`Copy` type, this would
**move** the value into the method. For a `Copy` type, it **copies** the
value. The caller can still use it.

Why not `&self`? A reference is 8 bytes and needs a memory read through a
pointer. The value itself is 16 bytes and fits in two processor registers.
For small `Copy` types, `self` by value is the idiomatic choice. The
standard library does the same: `f64::sqrt(self)`.

### 4.4 Traits and operator overloading

A **trait** defines a set of methods that a type can implement. It is
similar to an "interface" in other languages.

The standard library defines the operator traits in `std::ops`. The
compiler translates operators to trait method calls:

| Expression | Compiler translation |
|---|---|
| `z + w` | `Add::add(z, w)` |
| `z - w` | `Sub::sub(z, w)` |
| `z * w` | `Mul::mul(z, w)` |

The definition of `Add` (simplified):

```rust
pub trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}
```

- `Rhs` ("right-hand side") is the type of the right operand. The default
  is `Self`. Thus `impl Add for Complex` means `Complex + Complex`.
- `type Output` is an **associated type**. It names the result type. For
  `Complex + Complex`, the result is `Complex`.
- `add` takes `self` **by value**. Because `Complex` is `Copy`, the operands
  are copied, not moved.

Our implementation:

```rust
impl Mul for Complex {
    type Output = Complex;
    fn mul(self, other: Complex) -> Complex {
        let (a, b) = (self.re, self.im);
        let (c, d) = (other.re, other.im);
        Complex::new(a * c - b * d, a * d + b * c)
    }
}
```

The local names `a`, `b`, `c`, `d` are the same as in the equation. This
makes it easy to compare the code with the mathematics. The
`let (a, b) = (...)` syntax is **destructuring**: it unpacks a tuple into
two variables.

**Why operators for `Complex`, but named methods for `Vector`?** Complex
operations cannot fail. Every `Complex + Complex` has a result. Vector
operations can fail on a dimension mismatch (Entry 2, section 6.1).

**Static dispatch.** The compiler knows the exact type at compile time. It
calls `Complex::mul` directly, and it usually **inlines** it (it copies the
function body into the caller). There is no runtime cost for the trait.

### 4.5 Private fields with accessor methods

The fields `re` and `im` are private. The methods `re()` and `im()` return
them. This keeps the same style as `Vector`. It also allows a later change of
the internal representation (for example, to `[f64; 2]`) without breaking
code that uses `Complex`.

Note: in Rust, a field and a method can have the same name. `z.re` is the
field (only visible inside the module). `z.re()` is the method.

### 4.6 Floating-point methods

| Method | Mathematics |
|---|---|
| `x.hypot(y)` | `√(x² + y²)` without overflow |
| `y.atan2(x)` | the angle of the point `(x, y)`, in `[−π, π]` |
| `x.sqrt()` | `√x` |

These are methods of the primitive type `f64`. The standard library
implements them with the platform's mathematics library.

### 4.7 A test helper with a custom panic message

```rust
fn assert_complex_eq(actual: Complex, expected: Complex) {
    assert!(cond, "expected {expected:?}, got {actual:?}");
}
```

`assert!` accepts a format message after the condition. `{expected:?}`
is **inline format arguments**: the variable name inside the braces. If the
test fails, the message shows both values.

## 5. Memory and execution

**Layout.** `Complex` is two `f64` values: 16 bytes, alignment 8. There are
no pointers and no heap data.

```text
 Stack (or two registers)
┌──────────┬──────────┐
│ re: f64  │ im: f64  │
└──────────┴──────────┘
   8 bytes    8 bytes
```

Note: the default Rust layout (`repr(Rust)`) does not guarantee the field
order. For two fields of the same type, the order does not matter for us.
If we needed a fixed order (for example, to share memory with C code), we
would add `#[repr(C)]`.

**Execution of `z * w`:**

1. The compiler copies `z` and `w` into the arguments of `Mul::mul`. With
   inlining, they usually stay in registers. No memory is used.
2. The processor calculates four multiplications (`ac`, `bd`, `ad`, `bc`),
   one subtraction and one addition.
3. The result is a new `Complex`, returned in registers.
4. `z` and `w` are unchanged and still usable.

**Ownership summary:**

| Operation | Allocation | Ownership |
|---|---|---|
| `Complex::new` | none | creates a value |
| `z + w`, `z - w`, `z * w` | none | copies `z` and `w` |
| `conjugate`, `magnitude`, `phase` | none | copies `self` |

**Scope.** When a `Complex` goes out of scope, nothing happens. There is no
heap memory to free and no `Drop` code.

## 6. Engineering decisions

1. **A struct with two `f64` fields.** It matches the mathematics `(a, b)`.
   It has a clear name and methods.
2. **`Copy`.** The type is small and has no heap data. Copy semantics make
   the arithmetic code natural.
3. **Operator traits only for `+`, `−`, `×`.** These are the operations
   that the task needs. We did not add division or negation, because no
   later step needs them yet. (No speculative features.)
4. **`hypot` instead of `(a*a + b*b).sqrt()`.** It is overflow-safe. In
   Entry 2, `Vector::magnitude` uses the simple formula. We kept both:
   for `n` dimensions the standard library has no `hypot`, and an extra
   scaling pass costs time.
5. **`atan2` instead of `atan`.** It finds the correct quadrant and handles
   `a = 0`.
6. **No dependency on `num-complex`.** The project rule is to implement the
   primitives ourselves.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| Tuple `(f64, f64)` | No new type | No methods, no operators, no meaning in the type |
| Public fields `pub re`, `pub im` | Shorter code (`z.re`). The `num-complex` crate does this | Users can change components directly. Less control |
| Polar storage `(r, φ)` | Multiplication is cheap (`r₁r₂`, `φ₁ + φ₂`) | Addition needs `sin`/`cos`. Angles need wrapping. More rounding |
| Generic `Complex<T>` | Works with `f32` and `f64` | Needs a numeric trait. More complex code. Not needed now |
| `num-complex` crate | Complete, tested | Hides the concepts. Adds a dependency |
| `impl Add<&Complex> for &Complex` | Works without `Copy` | Not needed: `Complex` is `Copy` |

## 8. Performance considerations

- **Time.** Every operation is O(1). Multiplication needs 4 multiplications
  and 2 additions. Modern processors can also use FMA (fused
  multiply-add, `a·c − b·d` in one instruction with one rounding) if the
  compiler enables it.
- **Memory.** 16 bytes, no heap. An array of `n` complex numbers uses
  `16·n` bytes, contiguous.
- **Inlining.** The trait methods are small. The compiler inlines them in
  optimized builds. There is no function call overhead.
- **`hypot` versus `sqrt`.** `hypot` is slower than `(a*a + b*b).sqrt()`,
  because it must avoid overflow. In a hot loop, the simple formula can be
  faster. Stage 6 can measure this.
- **`atan2`** is a relatively expensive function (tens of processor cycles).
  RoPE does not need `phase` in its main calculation. It needs `cos` and
  `sin` of known angles.

## 9. Practical example

```rust
use rust_rope::math::complex::Complex;

let z = Complex::new(1.0, 2.0);
let w = Complex::new(3.0, 4.0);

println!("{:?}", z + w);
println!("{:?}", z * w);
println!("{:?}", Complex::I * Complex::I);
println!("{:?}", w.conjugate());
println!("{}", w.magnitude());
println!("{}", Complex::new(-1.0, -1.0).phase());
```

Expected output:

```text
Complex { re: 4.0, im: 6.0 }
Complex { re: -5.0, im: 10.0 }
Complex { re: -1.0, im: 0.0 }
Complex { re: 3.0, im: -4.0 }
5
-2.356194490192345
```

`−2.356194490192345` is `−3π/4` (−135°).

## 10. What we learned

**Mathematics:**

- `i² = −1`. A complex number `a + bi` is a point `(a, b)` in the plane.
- Addition is vector addition.
- Multiplication: `(ac − bd) + (ad + bc)i`. We derived it from the
  distributive law and `i² = −1`.
- Multiplication **multiplies the magnitudes** and **adds the phases**.
  Multiplication by `i` is a 90° rotation.
- The conjugate reflects across the real axis. `z·z̄ = |z|²`.
- `atan2` gives the correct phase in all four quadrants.

**Rust:**

- `Copy` types are copied, not moved. Only types without heap data can be
  `Copy`.
- Operator overloading uses the traits `Add`, `Sub`, `Mul` with an
  associated `Output` type.
- `const fn` and associated constants run at compile time.
- Small `Copy` types take `self` by value, not `&self`.
- Trait calls on known types use static dispatch and inlining: zero cost.


---

# Entry 4 — Trigonometry (Stage 2, Component 3)

**Date:** 2026-10-31
**Stage:** 2 (Mathematical foundations)
**File:** `src/math/trigonometry.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `degrees_to_radians`, `radians_to_degrees` | ✅ Implemented |
| `sine`, `cosine`, `sin_cos` | ✅ Implemented (calls to the standard library) |
| `unit_circle_point` | ✅ Implemented |
| Precomputed sine/cosine tables | ⏳ Planned (Stage 3 or 6, a RoPE cache) |
| Our own sine/cosine algorithm (for example a Taylor series) | ❌ Not planned. See section 7. |

---

## 1. What did we build?

We built a small module with six functions:

| Function | Calculation |
|---|---|
| `degrees_to_radians(deg)` | `deg × π / 180` |
| `radians_to_degrees(rad)` | `rad × 180 / π` |
| `sine(θ)` | `sin(θ)` |
| `cosine(θ)` | `cos(θ)` |
| `sin_cos(θ)` | `(sin(θ), cos(θ))` in one call |
| `unit_circle_point(θ)` | `[cos(θ), sin(θ)]` |

## 2. Why did we build it?

RoPE rotates each pair of numbers by an angle. To rotate by an angle `θ`,
we need `cos(θ)` and `sin(θ)` (Entry 5). Every RoPE calculation therefore
calls sine and cosine. In a real model, it calls them millions of times.

This module gives one central place for these calls. The rotation matrix
(Entry 5) and the polar form of complex numbers (Entry 6) both use it.

## 3. Mathematical foundation

### 3.1 Angles: degrees and radians

An **angle** measures a rotation. There are two common units.

**Degrees.** A full turn is 360 degrees (360°). This number is historical.
It has no special mathematical meaning.

**Radians.** Draw a circle with radius `r`. Walk along the circle from the
positive `x` axis. Let `s` be the length of the path along the circle (the
**arc length**). Then the angle in radians is:

```text
θ = s / r
```

A full turn walks around the full circumference `2πr`. Thus a full turn is
`2πr / r = 2π` radians.

```text
360° = 2π rad    →    180° = π rad    →    1° = π/180 rad
```

From this, we get the conversion formulas:

```text
θ_radians = θ_degrees × π / 180
θ_degrees = θ_radians × 180 / π
```

**Numerical examples:**

| Degrees | Radians (exact) | Radians (decimal) |
|---|---|---|
| 0° | 0 | 0 |
| 30° | π/6 | 0.5236 |
| 45° | π/4 | 0.7854 |
| 90° | π/2 | 1.5708 |
| 180° | π | 3.1416 |
| 270° | 3π/2 | 4.7124 |
| 360° | 2π | 6.2832 |

Example calculation: `90 × π / 180 = π / 2 ≈ 1.5708`.

### 3.2 Why computers (and mathematics) use radians

Radians are the **natural** unit. Three reasons:

1. **Arc length is simple.** With radians, the arc length is `s = r·θ`. With
   degrees, it is `s = r·θ·π/180`.
2. **The derivatives are simple.** The derivative (the rate of change) of
   `sin(θ)` is `cos(θ)` **only** when `θ` is in radians. In degrees, it is
   `(π/180)·cos(θ)`. Many formulas, for example the Taylor series
   `sin(θ) = θ − θ³/6 + ...`, are only correct in radians.
3. **Small angles.** For a small `θ` in radians, `sin(θ) ≈ θ`. Example:
   `sin(0.01) = 0.0099998...`. In degrees, this simple approximation does
   not exist.

For these reasons, the standard library functions `f64::sin` and
`f64::cos` expect radians. RoPE also uses radians: the angle of pair `i` at
position `m` is `m·θᵢ` radians.

### 3.3 The unit circle

The **unit circle** is the circle with radius 1 and its centre at the
origin `(0, 0)`.

Start at the point `(1, 0)`. Turn counterclockwise by the angle `θ`. You
arrive at a point `P` on the circle. By definition:

```text
x = cos(θ)     (the horizontal coordinate of P)
y = sin(θ)     (the vertical coordinate of P)
```

```text
              y
              │   P = (cos θ, sin θ)
         1 ┤  │  •
              │ /│
              │/ │ sin θ
──────────────┼──┴──────── x
              │cos θ   1
```

This is the definition of sine and cosine for **all** angles, also for
negative angles and angles larger than 2π.

**The four quarter turns:**

| θ (degrees) | θ (radians) | Point `(cos θ, sin θ)` |
|---|---|---|
| 0° | 0 | (1, 0) |
| 90° | π/2 | (0, 1) |
| 180° | π | (−1, 0) |
| 270° | 3π/2 | (0, −1) |

**Link to right triangles.** Draw a vertical line from `P` to the `x` axis.
This makes a right triangle with hypotenuse 1. The side next to the angle
has the length `cos(θ)`. The opposite side has the length `sin(θ)`. For a
triangle with hypotenuse `r`, multiply by `r`: the sides are `r·cos(θ)` and
`r·sin(θ)`. We used this in Entries 2 and 3.

### 3.4 The Pythagorean identity

`P` is on the unit circle, so its distance from the origin is 1. The
theorem of Pythagoras gives:

```text
cos²(θ) + sin²(θ) = 1
```

(`sin²(θ)` means `(sin(θ))²`.) The test `pythagorean_identity` checks this
for many angles. The test `unit_circle_points_have_length_one` checks the
same fact with `hypot`.

**Consequence.** `−1 ≤ sin(θ) ≤ 1` and `−1 ≤ cos(θ) ≤ 1`.

### 3.5 Symmetry

Turn by `−θ` (clockwise). The point is the reflection of `P` across the `x`
axis: `(cos θ, −sin θ)`. Thus:

```text
cos(−θ) =  cos(θ)    (cosine is an "even" function)
sin(−θ) = −sin(θ)    (sine is an "odd" function)
```

The test `sine_is_odd_and_cosine_is_even` checks this.

### 3.6 Periodicity

A full turn (2π radians) returns to the same point. Thus:

```text
sin(θ + 2πk) = sin(θ)
cos(θ + 2πk) = cos(θ)        for every integer k
```

The **period** of sine and cosine is `2π`. The test
`periodicity_with_period_two_pi` checks this.

**Why periodicity is important for RoPE.** The rotation angle of pair `i`
at position `m` is `m·θᵢ`. Because of periodicity, the rotation repeats
after `2π / θᵢ` positions. This number is the **wavelength** of the pair.
RoPE uses many different frequencies `θᵢ` so that the combination of all
pairs does not repeat for a very long distance. Stage 3 studies this.

### 3.7 Two angle-addition formulas (for later entries)

```text
cos(α + β) = cos(α)·cos(β) − sin(α)·sin(β)
sin(α + β) = sin(α)·cos(β) + cos(α)·sin(β)
```

`α` and `β` are two angles. **Derivation** (with the result of Entry 3:
complex multiplication adds the phases). The unit-circle points
`u = cos α + i·sin α` and `v = cos β + i·sin β` both have magnitude 1.
Their product `u·v` has magnitude `1·1 = 1` and phase `α + β`. Thus
`u·v = cos(α + β) + i·sin(α + β)`. Now multiply with the formula of
Entry 3, `(ac − bd) + (ad + bc)i`, where `a = cos α`, `b = sin α`,
`c = cos β`, `d = sin β`:

```text
real part:      cos α·cos β − sin α·sin β   = cos(α + β)
imaginary part: cos α·sin β + sin α·cos β   = sin(α + β)
```

Entry 5 uses these formulas to derive the rotation matrix.

### 3.8 Floating-point precision

This is the most important precision topic of Stage 2. We measured these
values on this computer:

| Calculation | Mathematical value | `f64` result |
|---|---|---|
| `cos(π/2)` | 0 | `6.123233995736766e-17` |
| `sin(π)` | 0 | `1.2246467991473532e-16` |
| `cos(3π/2)` | 0 | `-1.8369701987210297e-16` |
| `sin(2π)` | 0 | `-2.4492935982947064e-16` |
| `sin(π/6)` | 0.5 | `0.49999999999999994` |
| `cos(π/3)` | 0.5 | `0.5000000000000001` |
| `sin(2π · 10⁶)` | 0 | `-4.463824362721742e-10` |

**Why is `sin(π)` not zero?** The `sin` function is not wrong. The
**input** is wrong. `π` is irrational: it has infinitely many digits. The
constant `std::f64::consts::PI` is the nearest `f64` value:

```text
PI (f64) = π − δ,    δ ≈ 1.2246e-16
```

Near `π`, `sin(π − δ) = sin(δ) ≈ δ` (the small-angle rule from 3.2). Thus
`sin(PI)` returns `1.2246e-16`. This is the correct sine of the number that
we gave it.

**The error grows with the angle.** An `f64` has about 16 significant
decimal digits. For a value near `6.28 · 10⁶`, the spacing between two
neighbouring `f64` values is about `10⁻⁹`. Thus the angle itself has a
rounding error of up to half this spacing, about `5 · 10⁻¹⁰`. The measured
result `−4.46e-10` is this rounding error. `sin` returns it.

The rule: **the absolute error of an angle in `f64` is about
`|θ| × 1.1 × 10⁻¹⁶`.** Large angles have large absolute errors.

**Why this matters for RoPE.** The angle is `m·θᵢ`. For the first pair,
`θ₀ = 1`, so at position `m = 100 000` the angle is `100 000` radians. In
`f64`, the error is about `10⁻¹¹`. This is acceptable. In `f32`
(7 significant digits) the error is about `0.004` radians. This is much
larger. This is one reason that real implementations calculate the angles
carefully. Stage 3 and Stage 6 study it. The test
`precision_loss_for_large_angles` documents the effect.

**Conversion precision.** `degrees_to_radians(180.0)` returns exactly
`PI`, because the rounding errors cancel for this value. But in general,
the conversion has a rounding error of about one unit in the last digit.
`radians_to_degrees(degrees_to_radians(x))` is usually equal to `x`
within the tolerance, but not always bit for bit.

## 4. Rust concepts learned

### 4.1 Free functions in a module

The functions in this module are **free functions**: they are not methods
of a type. A free function is the natural choice when there is no data
type that owns the operation. Callers use the module path:

```rust
use rust_rope::math::trigonometry::{sin_cos, degrees_to_radians};
let (s, c) = sin_cos(degrees_to_radians(30.0));
```

### 4.2 Methods on primitive types

`f64` is a primitive type, but it has methods: `theta.sin()`,
`theta.cos()`, `theta.sin_cos()`. The standard library defines them in an
`impl f64` block. `theta.sin()` and `f64::sin(theta)` are the same call. We
use the second form in a test: `f64::to_radians(degrees)`.

### 4.3 Constants from `std::f64::consts`

`PI`, `TAU` (`2π`), `FRAC_PI_2` (`π/2`), `FRAC_PI_4` (`π/4`),
`FRAC_PI_6` (`π/6`), `FRAC_1_SQRT_2` (`1/√2`) and `SQRT_2` (`√2`) are
predefined constants. Each is the nearest `f64` value to the exact
mathematical value. Use them instead of writing `3.14159...` by hand.

### 4.4 Constant folding

```rust
degrees * (PI / 180.0)
```

`PI / 180.0` contains only constants. The compiler calculates it during
compilation (**constant folding**). The program contains only one
multiplication. The parentheses matter: `degrees * PI / 180.0` means
`(degrees * PI) / 180.0`. That is one multiplication **and** one division
at run time, and it can round differently.

### 4.5 Tuples

`sin_cos` returns a **tuple** `(f64, f64)`. A tuple groups values of fixed
number and type, without names. The caller unpacks it with destructuring:

```rust
let (s, c) = sin_cos(theta);
```

A tuple of two `f64` values is 16 bytes. It is returned in registers. No
allocation occurs.

### 4.6 Fixed-size arrays `[f64; 2]`

`unit_circle_point` returns `[f64; 2]`: an array with exactly two `f64`
values.

- The size is part of the **type**. `[f64; 2]` and `[f64; 3]` are different
  types. Thus the compiler knows that a 2-D point always has 2 components.
  No dimension error is possible at run time.
- The array is stored **inline**: on the stack, or in registers. No heap.
- `[f64; 2]` is `Copy`, because `f64` is `Copy`.

Compare with `Vector` (Entry 2): `Vec<f64>` has a size that is known only
at run time, so it needs the heap and a runtime dimension check.

| | `[f64; 2]` | `Vector` (`Vec<f64>`) |
|---|---|---|
| Size known at | compile time | run time |
| Storage | inline (stack) | heap |
| Size in bytes | 16 | 24 (stack) + 8·n (heap) |
| `Copy` | yes | no |
| Dimension check | by the compiler | at run time (`Result`) |

**Array patterns.** `let [x, y] = unit_circle_point(theta);` unpacks the
array. The compiler accepts this pattern only for an array with exactly 2
elements.

### 4.7 `#[inline]`

`#[inline]` is a hint to the compiler: "copy the body of this function into
the caller". Without the attribute, the compiler can inline a function only
inside its own crate (unless link-time optimization is on). With
`#[inline]`, other crates (our tests, examples and benchmarks) can also
inline it. For a one-line wrapper around `f64::sin`, inlining removes all
cost of the wrapper.

### 4.8 Tests with loops and arrays of cases

```rust
for (theta, [ex, ey]) in expected { ... }
```

Many tests iterate over an array of test cases. One test checks a property
for many input values. The pattern `(theta, [ex, ey])` unpacks each tuple
and its inner array.

## 5. Memory and execution

- Every function takes `f64` values **by value**. `f64` is `Copy`. Nothing
  is borrowed or moved in a way that the caller can see.
- No function allocates. All values live in registers or on the stack.
- **Execution of `sin_cos(θ)`:** the standard library calls the platform
  mathematics library. The algorithm first reduces `θ` into a small range
  near zero, with the periodicity of section 3.6 ("range reduction"). Then
  it evaluates a polynomial that approximates sine and cosine in that range.
  `sin_cos` can share the range reduction for both results.
- Nothing needs to be freed when the values go out of scope.

## 6. Engineering decisions

1. **We wrap the standard library functions.** The task permits standard
   library methods. They are accurate (usually within one unit in the last
   place) and fast. Our own implementation would be slower and less
   accurate.
2. **We wrote the conversion formulas ourselves,** because the task asks us
   to learn the equation. We used the same expression as the standard
   library, `x * (π / 180)`. The test
   `conversion_matches_the_standard_library_exactly` proves that the results
   are bit-for-bit equal.
3. **`sin_cos` returns `(sin, cos)`.** This is the order of `f64::sin_cos`.
   A different order would confuse users who know the standard library.
4. **`unit_circle_point` returns `[cos, sin]`.** This is the order `(x, y)`.
   Note the difference from `sin_cos`. The function name tells the
   meaning: a point is `(x, y)`.
5. **Radians everywhere.** Only `degrees_to_radians` accepts degrees. All
   other functions expect radians. This prevents a mix of units.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| Our own Taylor series for sine | We learn the approximation | Less accurate, slower, and needs range reduction. Not the goal of this project |
| A lookup table of sine values | Very fast for repeated angles | Needs memory, and has interpolation errors. RoPE uses a different form: a table of `cos(m·θᵢ)` for all positions (Stage 3) |
| A newtype `Radians(f64)` | The compiler prevents unit mistakes | Extra syntax at every call. Not needed in a small project |
| Call `f64::sin` directly in all modules | No wrapper | The mathematics is spread out. No central place for documentation |
| `f32` instead of `f64` | Two times less memory, faster SIMD | 7 digits. Large angles lose much precision (section 3.8) |

## 8. Performance considerations

- **Conversion:** one multiplication. About 1 processor cycle.
- **`sin`, `cos`:** about 20 to 100 cycles each, depending on the platform
  and on the size of the angle. Large angles need a more expensive range
  reduction.
- **`sin_cos`:** can be faster than two calls. Rust does not guarantee it.
  Stage 6 can measure it.
- **Implication for RoPE:** sine and cosine are expensive compared to
  multiplication. RoPE needs `cos(m·θᵢ)` and `sin(m·θᵢ)` for all positions
  `m` and pairs `i`. These values do not depend on the input data. Thus
  real implementations calculate them **one time** and store them in a
  table ("cache"). We plan this in Stage 3.

## 9. Practical example

```rust
use rust_rope::math::trigonometry::*;

for degrees in [0.0, 90.0, 180.0, 270.0] {
    let theta = degrees_to_radians(degrees);
    let [x, y] = unit_circle_point(theta);
    println!("{degrees:>5}° = {theta:.4} rad -> ({x:e}, {y:e})");
}
```

Expected output:

```text
    0° = 0.0000 rad -> (1e0, 0e0)
   90° = 1.5708 rad -> (6.123233995736766e-17, 1e0)
  180° = 3.1416 rad -> (-1e0, 1.2246467991473532e-16)
  270° = 4.7124 rad -> (-1.8369701987210297e-16, -1e0)
```

The values `6.1e-17`, `1.2e-16` and `−1.8e-16` are the mathematical zeros.
Section 3.8 explains them. `{:e}` prints in scientific notation, so the
small errors are visible. `{:.4}` prints 4 decimal places.

## 10. What we learned

**Mathematics:**

- Radians measure an angle by arc length: `θ = s / r`. A full turn is `2π`.
  `θ_rad = θ_deg × π / 180`.
- Radians are the natural unit: the formulas for derivatives, series and
  small angles are simple only in radians.
- On the unit circle, the point at angle `θ` is `(cos θ, sin θ)`.
- `sin² + cos² = 1`. Cosine is even, sine is odd. Both have the period `2π`.
- Periodicity creates the "wavelength" `2π / θᵢ` of each RoPE pair.
- The angle-addition formulas follow from complex multiplication.
- `sin(PI) ≠ 0` because `PI` is not exactly `π`. The absolute error of an
  angle grows with its size. `f32` is much worse than `f64` for large
  angles.

**Rust:**

- Free functions, methods on primitives, and `std::f64::consts`.
- Constant folding: `x * (PI / 180.0)` is one multiplication.
- Tuples and fixed-size arrays are stored inline and are `Copy`.
- `[f64; 2]` puts the dimension into the type. The compiler checks it.
- `#[inline]` allows inlining across crates.


---

# Entry 5 — 2 × 2 matrices and rotation matrices (Stage 2, Component 4)

**Date:** 2026-10-31
**Stage:** 2 (Mathematical foundations)
**File:** `src/math/matrix.rs` (the `Vector2`, `Matrix2` and
`rotate_vector` part)

### Implementation status of this entry

| Item | Status |
|---|---|
| `Vector2` (type alias for `[f64; 2]`) | ✅ Implemented |
| `Matrix2`: `new`, `from_rows`, `rows`, `get`, `IDENTITY` | ✅ Implemented |
| `transpose`, `determinant` | ✅ Implemented |
| `mul_vector` (matrix-vector), `mul_matrix` (matrix-matrix) | ✅ Implemented |
| `Matrix2::rotation(θ)`, `rotate_vector(v, θ)` | ✅ Implemented |
| General `n × n` matrices | ❌ Not planned. RoPE needs only 2 × 2 blocks. |
| Rotation of RoPE dimension pairs | ⏳ Planned (Stages 3–4) |

---

## 1. What did we build?

We built:

- `Vector2`: a 2-D vector `[x, y]` as a fixed-size array `[f64; 2]`.
- `Matrix2`: a 2 × 2 matrix, with construction, element access, transpose,
  determinant, matrix-vector multiplication and matrix-matrix
  multiplication.
- `Matrix2::rotation(θ)`: the rotation matrix `R(θ)`.
- `rotate_vector(v, θ)`: rotates `v` around the origin by `θ` radians.

We did not duplicate the vector operations of Entry 2. The tests calculate
the lengths and dot products of `Vector2` values with the existing `Vector`
type.

## 2. Why did we build it?

The rotation matrix is the core operation of RoPE. RoPE takes a vector with
`d` components and divides it into `d/2` pairs. It rotates each pair with
its own `R(θ)`. Thus, RoPE is "many 2-D rotations at the same time". If we
understand one 2-D rotation completely, we understand the core of RoPE.

The general matrix operations (multiplication, transpose) are necessary to
prove the important properties: a rotation keeps lengths, and two rotations
combine into one rotation by the sum of the angles.

## 3. Mathematical foundation

### 3.1 What is a matrix?

A **matrix** is a rectangular table of numbers. A 2 × 2 matrix has 2 rows
and 2 columns:

```text
M = [ a  b ]      row 1: a, b
    [ c  d ]      row 2: c, d
```

A matrix describes a **linear transformation**: a rule that moves every
point of the plane to a new point. "Linear" means that the rule keeps
straight lines straight and keeps the origin fixed.

### 3.2 Matrix-vector multiplication

```text
[ a  b ] [ x ]   [ a·x + b·y ]
[ c  d ] [ y ] = [ c·x + d·y ]
```

Each component of the result is the **dot product** of one row with the
vector (Entry 2). Row 1 · `[x, y]` = `a·x + b·y`.

**Numerical example.**

```text
[ 1  2 ] [ 5 ]   [ 1·5 + 2·6 ]   [ 17 ]
[ 3  4 ] [ 6 ] = [ 3·5 + 4·6 ] = [ 39 ]
```

**The column interpretation (important for the derivation below).** Write
the product in another way:

```text
M·[x, y] = x·[a, c] + y·[b, d]
```

The vector `[1, 0]` gives `M·[1, 0] = [a, c]`: the **first column**. The
vector `[0, 1]` gives `M·[0, 1] = [b, d]`: the **second column**. Thus:

> **The columns of `M` are the images of the two axis vectors.**
> Column 1 shows where `[1, 0]` goes. Column 2 shows where `[0, 1]` goes.

### 3.3 Matrix-matrix multiplication

```text
[ a  b ] [ e  f ]   [ a·e + b·g   a·f + b·h ]
[ c  d ] [ g  h ] = [ c·e + d·g   c·f + d·h ]
```

The element in row `r`, column `c` of the result is the dot product of row
`r` of the first matrix with column `c` of the second matrix.

**Derivation.** We want a matrix `P` with `P·v = M·(N·v)` for all `v`:
"first apply `N`, then apply `M`". Calculate `N·v = [e·x + f·y, g·x + h·y]`.
Then apply `M`:

```text
first component:  a·(e·x + f·y) + b·(g·x + h·y) = (a·e + b·g)·x + (a·f + b·h)·y
second component: c·(e·x + f·y) + d·(g·x + h·y) = (c·e + d·g)·x + (c·f + d·h)·y
```

The coefficients of `x` and `y` are exactly the elements of `M·N`. The test
`product_matrix_applies_right_factor_first` checks `(M·N)·v = M·(N·v)`.

**Numerical example.**

```text
[ 1  2 ] [ 5  6 ]   [ 1·5 + 2·7   1·6 + 2·8 ]   [ 19  22 ]
[ 3  4 ] [ 7  8 ] = [ 3·5 + 4·7   3·6 + 4·8 ] = [ 43  50 ]
```

**The order is important.** In the opposite order, `N·M` gives
`[[23, 34], [31, 46]]`. Matrix multiplication is **not commutative**: in
general `M·N ≠ N·M`. (Exception: two rotation matrices commute, because
`R(α)·R(β) = R(α + β) = R(β + α)`.)

### 3.4 The identity, the transpose and the determinant

- The **identity** `I = [[1, 0], [0, 1]]` changes nothing: `I·v = v` and
  `I·M = M·I = M`.
- The **transpose** `Mᵀ` changes rows into columns:
  `[[a, b], [c, d]]ᵀ = [[a, c], [b, d]]`.
- The **determinant** `det(M) = a·d − b·c` is one number. `|det(M)|` is the
  factor by which `M` scales areas. If `det(M) < 0`, `M` also reflects the
  plane (it changes "counterclockwise" into "clockwise"). Example:
  `det([[1, 2], [3, 4]]) = 1·4 − 2·3 = −2`.

### 3.5 Derivation of the rotation matrix — method 1: the axis vectors

We want the matrix `R(θ)` that rotates every vector counterclockwise by the
angle `θ` around the origin. From 3.2, we only need to know where the two
axis vectors go.

- `[1, 0]` is the point at angle 0 on the unit circle. Rotate it by `θ`. It
  arrives at the point at angle `θ`: `[cos θ, sin θ]` (Entry 4). This is
  **column 1**.
- `[0, 1]` is the point at angle `π/2`. Rotate it by `θ`. It arrives at
  angle `θ + π/2`: `[cos(θ + π/2), sin(θ + π/2)]`. Use the angle-addition
  formulas (Entry 4, section 3.7) with `cos(π/2) = 0` and `sin(π/2) = 1`:

```text
cos(θ + π/2) = cos θ·0 − sin θ·1 = −sin θ
sin(θ + π/2) = sin θ·0 + cos θ·1 =  cos θ
```

  This is **column 2**: `[−sin θ, cos θ]`.

Put the two columns together:

```text
R(θ) = [ cos θ   −sin θ ]
       [ sin θ    cos θ ]
```

### 3.6 Derivation of the rotation matrix — method 2: polar coordinates

Write the vector `v = [x, y]` with its length `r` and its angle `φ`
(Entry 4, section 3.3):

```text
x = r·cos φ,    y = r·sin φ
```

A rotation by `θ` keeps the length `r` and adds `θ` to the angle. The new
vector `[x', y']` (read "x prime", "y prime") is:

```text
x' = r·cos(φ + θ),    y' = r·sin(φ + θ)
```

Expand with the angle-addition formulas:

```text
x' = r·(cos φ·cos θ − sin φ·sin θ) = (r·cos φ)·cos θ − (r·sin φ)·sin θ
y' = r·(sin φ·cos θ + cos φ·sin θ) = (r·cos φ)·sin θ + (r·sin φ)·cos θ
```

Replace `r·cos φ` by `x`, and `r·sin φ` by `y`:

```text
x' = x·cos θ − y·sin θ
y' = x·sin θ + y·cos θ
```

This is exactly `R(θ)·[x, y]` from 3.2. Both methods give the same matrix.

### 3.7 Why `R(θ)` rotates around the origin

A transformation is a rotation around the origin if it has three
properties. We prove each property.

**(1) It keeps the origin fixed.** `R(θ)·[0, 0] = [0, 0]`. Every matrix does
this. (Test: `rotating_the_zero_vector_gives_zero`.)

**(2) It keeps lengths.** Calculate `x'² + y'²`:

```text
x'² = x²cos²θ − 2xy·cosθ·sinθ + y²sin²θ
y'² = x²sin²θ + 2xy·sinθ·cosθ + y²cos²θ
sum = x²(cos²θ + sin²θ) + y²(sin²θ + cos²θ)    (the 2xy terms cancel)
    = x² + y²                                  (because sin² + cos² = 1)
```

Thus `‖R(θ)·v‖ = ‖v‖`. (Test: `rotation_keeps_the_length`.) More generally,
`R` keeps dot products, so it also keeps the angles between vectors. (Test:
`rotation_keeps_dot_products`.)

**(3) It does not reflect.**
`det(R(θ)) = cos θ·cos θ − (−sin θ)·sin θ = cos²θ + sin²θ = 1`. The
determinant is positive, so there is no reflection. It is 1, so areas do
not change. (Test: `rotation_matrix_has_determinant_one`.) A reflection,
for example `[[1, 0], [0, −1]]`, also keeps lengths, but its determinant is
`−1`.

### 3.8 Rotations at the four quarter turns

Put `cos` and `sin` of each angle into `R(θ)`:

| θ | `cos θ` | `sin θ` | `R(θ)` | `[x, y]` goes to | `[2, 3]` goes to |
|---|---|---|---|---|---|
| 0 | 1 | 0 | `[[1, 0], [0, 1]]` | `[x, y]` | `[2, 3]` |
| π/2 | 0 | 1 | `[[0, −1], [1, 0]]` | `[−y, x]` | `[−3, 2]` |
| π | −1 | 0 | `[[−1, 0], [0, −1]]` | `[−x, −y]` | `[−2, −3]` |
| 3π/2 | 0 | −1 | `[[0, 1], [−1, 0]]` | `[y, −x]` | `[3, −2]` |

Compare `R(π/2)` with Entry 3: multiplication by `i` changes `x + yi` into
`−y + xi`. This is the same transformation.

### 3.9 Properties of rotation matrices

**Composition.** Two rotations, one after the other, are one rotation by the
sum of the angles:

```text
R(α)·R(β) = R(α + β)
```

Proof: multiply the matrices (3.3) and use the angle-addition formulas.
Element (1,1): `cos α·cos β − sin α·sin β = cos(α + β)`. The other three
elements are similar. (Test: `rotations_compose_by_adding_angles`.)

**The inverse is the transpose.**

```text
R(θ)ᵀ = [  cos θ   sin θ ] = [ cos(−θ)  −sin(−θ) ] = R(−θ)
        [ −sin θ   cos θ ]   [ sin(−θ)   cos(−θ) ]
```

We used `cos(−θ) = cos θ` and `sin(−θ) = −sin θ` (Entry 4). Thus
`R(θ)ᵀ·R(θ) = R(−θ + θ) = R(0) = I`. A matrix with `Mᵀ·M = I` is
**orthogonal**. (Tests: `transpose_is_the_inverse_rotation`,
`rotation_matrix_is_orthogonal`.)

**The relative-rotation identity.** Combine the two results:

```text
R(α)ᵀ·R(β) = R(−α)·R(β) = R(β − α)
```

(Test: `relative_rotation_identity`.) This identity is the mathematical
heart of RoPE.

### 3.10 Preview: how RoPE uses this (not implemented)

Let `q` be a query vector of a token at position `m`, and `k` a key vector
of a token at position `n`. For one pair of dimensions, RoPE calculates
`R(m·θ)·q` and `R(n·θ)·k`. Attention takes the dot product. A dot product
can be written as `uᵀ·w`. Thus:

```text
(R(mθ)·q)ᵀ · (R(nθ)·k) = qᵀ · R(mθ)ᵀ · R(nθ) · k = qᵀ · R((n − m)θ) · k
```

(We used `(A·q)ᵀ = qᵀ·Aᵀ`.) The result depends only on the **difference**
`n − m` of the positions, not on `m` and `n` separately. This is why RoPE
gives relative position information to attention.

For a full vector with `d` dimensions, RoPE uses a **block-diagonal**
matrix: `d/2` rotation blocks on the diagonal, each with its own frequency
`θᵢ`, and zeros everywhere else:

```text
[ R(mθ₀)    0     ...        0          ]
[   0     R(mθ₁)  ...        0          ]
[  ...     ...    ...       ...         ]
[   0       0     ...  R(mθ_{d/2−1})    ]
```

Each pair is rotated independently. The real implementation never builds
this large matrix. It rotates each pair directly, with 4 multiplications
and 2 additions. Stage 3 derives the frequencies `θᵢ`. Stage 4 implements
the pairwise rotation.

### 3.11 Floating-point errors in rotations

The test `quarter_turn_result_is_not_exact` shows:

```text
rotate_vector([1, 0], π/2) = [6.123233995736766e-17, 1.0]
```

The exact result is `[0, 1]`. The `x` component is
`1·cos(π/2) − 0·sin(π/2) = cos(PI/2)`. As Entry 4 explains, `PI/2` is not
exactly `π/2`, so `cos(PI/2) = 6.1e-17`. The `y` component is `sin(PI/2)`.
Near the top of the sine curve, the curve is flat, so the tiny input error
has no visible effect: the result is exactly `1.0`.

Other sources of error:

- Each multiplication and addition rounds the result (a relative error of
  up to about `1.1e-16` each).
- `R(α)·R(β)` and `R(α + β)` are mathematically equal. In `f64` they differ
  in the last digits, because the rounding steps are different.
- **Repeated rotations accumulate errors.** If you rotate a vector 1000
  times by a small angle, the length slowly drifts away from the original
  length. It is better to calculate one rotation by the total angle. RoPE
  does exactly this: it calculates `R(m·θ)` directly, not `m` rotations by
  `θ`.

All tests that compare rotation results use `approx_eq` with the tolerance
`1e-12`.

## 4. Rust concepts learned

### 4.1 Type aliases

```rust
pub type Vector2 = [f64; 2];
```

A **type alias** gives a new name to an existing type. It does **not**
create a new type. `Vector2` and `[f64; 2]` are fully interchangeable. The
alias makes the signatures readable: `fn rotate_vector(v: Vector2, ...)`.

Compare with a **newtype** (`struct Vector2([f64; 2])`). A newtype is a new,
different type. It can have its own methods and trait implementations, but
the caller must wrap and unwrap the values. We do not need methods on
`Vector2` (the operations are on `Matrix2`), so the alias is sufficient.

### 4.2 Nested arrays and their layout

`Matrix2` stores `[[f64; 2]; 2]`: an array of 2 rows, each an array of 2
`f64` values. In memory, this is 4 contiguous `f64` values (32 bytes) in the
order `m₁₁, m₁₂, m₂₁, m₂₂`. This is **row-major order**.

```text
offset:  0       8       16      24
        ┌───────┬───────┬───────┬───────┐
        │  m11  │  m12  │  m21  │  m22  │
        └───────┴───────┴───────┴───────┘
           row 0           row 1
```

There are no pointers and no heap. `Matrix2` is `Copy`.

### 4.3 Destructuring patterns for arrays

```rust
let [[a, b], [c, d]] = self.rows;
let [x, y] = v;
```

A **pattern** on the left of `let` unpacks the value. The compiler checks
that the pattern matches the type exactly. `[[a, b], [c, d]]` matches only a
2 × 2 array. There is no runtime check, and no possible out-of-bounds
access. Then the code uses the same letters as the equations:
`[a * x + b * y, c * x + d * y]`.

### 4.4 `const fn` with arrays

`Matrix2::new`, `from_rows`, `rows`, `get` and `transpose` are `const fn`.
Thus `Matrix2::IDENTITY` is a compile-time constant. `rotation` cannot be
`const`, because `sin_cos` is not `const`.

`get` uses array indexing `self.rows[row][col]`. Array indexing checks the
bounds at run time and **panics** if the index is too large. The test
`get_out_of_range_panics` uses `#[should_panic]`: the test passes only if
the code panics.

**Why panic here and `Result` for `Vector`?** For `Vector`, the dimensions
come from the caller's data and can be wrong at run time. For `Matrix2`, the
valid indices are always 0 and 1. A wrong index is a bug in the caller's
code, not a data problem. `CLAUDE.md` section 6 says: "Use `panic!` only for
internal logic errors (bugs)". The documentation of `get` has a `# Panics`
section for this reason.

### 4.5 Methods that take `self` by value (again)

All `Matrix2` methods take `self`, not `&self`. `Matrix2` is 32 bytes and
`Copy`. In optimized builds, the compiler keeps the values in registers and
inlines the methods. The caller can still use the matrix after the call.

### 4.6 Named methods instead of the `*` operator

We named the products `mul_vector` and `mul_matrix`. We could implement
`Mul<Vector2> for Matrix2` and `Mul<Matrix2> for Matrix2`. Rust permits
this, because `Matrix2` is our own type. But:

- Two different products behind one symbol `*` are harder to read for a
  learner.
- The method names say exactly which product happens.

This is a style choice. Section 7 lists the alternative.

### 4.7 Module dependencies inside `math`

```rust
use crate::math::trigonometry::sin_cos;
```

`crate::` starts the path at the root of our library. `matrix` uses
`trigonometry`. Both are inside `math`. The architecture rule (`math` must
not use `rope`) is still kept.

### 4.8 Closures in tests

```rust
let dot = |u: Vector2, w: Vector2| Vector::from_slice(&u).dot(&Vector::from_slice(&w));
```

A **closure** is an anonymous function that you can store in a variable.
Here it reuses the `Vector` dot product (Entry 2) for `Vector2` values.
This avoids a second dot-product implementation. In a test, the two heap
allocations of `from_slice` do not matter.

## 5. Memory and execution

**Execution of `rotate_vector([2.0, 3.0], θ)`:**

1. The caller copies the array `[2.0, 3.0]` (16 bytes) into the parameter
   `v`. `Vector2` is `Copy`, so the caller keeps its own copy.
2. `Matrix2::rotation(θ)` calls `sin_cos(θ)`. It returns `(sin, cos)` in
   registers.
3. It builds a `Matrix2` value (32 bytes) on the stack or in registers.
4. `mul_vector` copies the matrix (by value), destructures it, and
   calculates 4 multiplications and 2 additions.
5. It returns a new `[f64; 2]`. The temporary matrix goes out of scope.
   Nothing is freed, because there is no heap memory.

In an optimized build, the compiler usually inlines all of this. The result
is a call to the sine/cosine routine, then 6 arithmetic instructions. There
is no allocation.

**Memory summary:**

| Value | Size | Location | `Copy` |
|---|---|---|---|
| `Vector2` | 16 bytes | inline (stack/registers) | yes |
| `Matrix2` | 32 bytes | inline (stack/registers) | yes |
| `Vector` (Entry 2) | 24 bytes + 8·n heap | stack + heap | no |

**Ownership:** all types in this component are `Copy`. Thus values are
copied, never moved, and nothing is borrowed.

## 6. Engineering decisions

1. **Fixed-size types for 2-D.** RoPE always rotates **pairs**. The size 2 is
   known at compile time. `[f64; 2]` and `[[f64; 2]; 2]` give zero heap
   allocation and compile-time dimension checks.
2. **A type alias, not a newtype, for `Vector2`.** No extra methods are
   needed. The alias keeps the call syntax simple:
   `rotate_vector([1.0, 0.0], θ)`.
3. **Row-major storage.** It matches the way we write matrices on paper and
   the order of the arguments in `Matrix2::new`.
4. **Explicit formulas instead of loops.** For a 2 × 2 matrix, four explicit
   expressions are clearer than three nested loops, and they match the
   equations letter by letter.
5. **No duplicate norm or dot product.** The tests use `Vector` for these.
6. **`transpose` and `determinant` are included.** They are not in the task
   list. We added them because they are necessary to *prove* the rotation
   properties in tests (orthogonality, no reflection), and the
   relative-rotation identity uses the transpose. They are small.
7. **`rotate_vector` builds the matrix.** A direct formula
   (`[x·c − y·s, x·s + y·c]`) is equivalent. Building the matrix shows the
   link between the mathematics and the code. In optimized builds, the
   compiler produces the same machine code.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| `impl Mul for Matrix2` (`m * n`, `m * v`) | Short, mathematical syntax | Two products behind one symbol. Less explicit for learners |
| Column-major storage | The convention of Fortran, BLAS and many GPU libraries | Different from the paper order. No benefit for 2 × 2 |
| Generic `Matrix<const R: usize, const C: usize>` | All sizes with one type | More complex code. RoPE needs only 2 × 2 |
| Store only `(cos θ, sin θ)` for rotations | 16 bytes instead of 32. RoPE implementations do this | A separate type for rotations. We prefer to show the full matrix first |
| `nalgebra` or `glam` crate | Complete, fast, SIMD | Hides the concepts. Adds a dependency |
| Rotate with complex multiplication | Same result (Entry 6) | A different view. We implement both to compare them |

## 8. Performance considerations

| Operation | Multiplications | Additions | Other |
|---|---|---|---|
| `mul_vector` | 4 | 2 | — |
| `mul_matrix` | 8 | 4 | — |
| `transpose` | 0 | 0 | copies values |
| `determinant` | 2 | 1 | — |
| `rotation(θ)` | 0 | 0 | 1 `sin_cos` call (expensive) |
| `rotate_vector` | 4 | 2 | 1 `sin_cos` call |

- Every operation is O(1). No allocation.
- **The expensive part is `sin_cos`**, not the arithmetic. If many vectors
  are rotated by the same angle, calculate `R(θ)` one time and reuse it. In
  RoPE, the angle `m·θᵢ` is the same for every vector at position `m`, so a
  precomputed table of `cos` and `sin` values is the standard optimization
  (Stage 3/6).
- **A full block-diagonal `d × d` matrix would be very wasteful.** It has
  `d²` elements, but only `2d` of them are not zero. The matrix-vector
  product would need `d²` multiplications. The pairwise rotation needs only
  `2d` multiplications. For `d = 128`, this is 16 384 versus 256.
- **SIMD.** The pairwise rotation applies the same 6 operations to many
  independent pairs. This pattern is ideal for SIMD (Stage 6).

## 9. Practical example

```rust
use rust_rope::math::matrix::{rotate_vector, Matrix2};
use std::f64::consts::{FRAC_PI_2, PI};

let m = Matrix2::new(1.0, 2.0, 3.0, 4.0);
println!("{:?}", m.mul_vector([5.0, 6.0]));
println!("{:?}", m.mul_matrix(Matrix2::new(5.0, 6.0, 7.0, 8.0)).rows());

for theta in [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2] {
    let [x, y] = rotate_vector([1.0, 0.0], theta);
    println!("θ = {theta:.4}: ({x:e}, {y:e})");
}
```

Expected output:

```text
[17.0, 39.0]
[[19.0, 22.0], [43.0, 50.0]]
θ = 0.0000: (1e0, 0e0)
θ = 1.5708: (6.123233995736766e-17, 1e0)
θ = 3.1416: (-1e0, 1.2246467991473532e-16)
θ = 4.7124: (-1.8369701987210297e-16, -1e0)
```

The small values (`6.1e-17` and others) are the mathematical zeros, with
the rounding error of section 3.11.

## 10. What we learned

**Mathematics:**

- A matrix is a linear transformation. Its columns are the images of the
  axis vectors.
- Matrix-vector and matrix-matrix products are row-times-column dot
  products. `M·N` means "first `N`, then `M`". In general `M·N ≠ N·M`.
- We derived `R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]` in two ways: from
  the images of the axis vectors, and from polar coordinates with the
  angle-addition formulas.
- `R(θ)` is a rotation: it fixes the origin, keeps lengths (proof with
  `sin² + cos² = 1`) and has determinant 1 (no reflection).
- `R(α)R(β) = R(α + β)`, `R(θ)ᵀ = R(−θ)`, and `R(α)ᵀR(β) = R(β − α)`. The
  last identity is the reason that RoPE encodes relative positions.
- RoPE applies one 2-D rotation to each pair of dimensions. It never builds
  the full `d × d` matrix.

**Rust:**

- Type aliases (`type Vector2 = [f64; 2]`) versus newtypes.
- Nested fixed-size arrays: inline, contiguous, row-major, `Copy`.
- Array destructuring patterns: checked at compile time, no bounds checks.
- `#[should_panic]` tests, and when `panic!` is correct (caller bugs with
  fixed indices) versus `Result` (data problems).
- `crate::` paths for dependencies between modules.


---

# Entry 6 — Euler's formula and rotation by complex multiplication (Stage 2, Component 5)

**Date:** 2026-10-31
**Stage:** 2 (Mathematical foundations)
**Files:** `src/math/complex.rs` (`from_polar`, `rotate`),
`tests/numerical_tests.rs`, `examples/math_foundations.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `Complex::from_polar(r, φ)` = `r·e^(iφ)` | ✅ Implemented |
| `Complex::rotate(θ)` = `z·e^(iθ)` | ✅ Implemented |
| Integration tests: complex rotation = matrix rotation, relative-position property | ✅ Implemented |
| Example program `examples/math_foundations.rs` | ✅ Implemented |
| A general complex exponential `exp(a + bi)` | ❌ Not implemented. See section 6. |
| RoPE with complex numbers | ⏳ Planned (Stages 3–4) |

---

## 1. What did we build?

We added two functions to `Complex`:

- `Complex::from_polar(r, φ)` creates the complex number with magnitude `r`
  and phase `φ`: `r·(cos φ + i·sin φ)`. With `r = 1`, this is `e^(iφ)`.
- `z.rotate(θ)` rotates `z` by the angle `θ`. It multiplies `z` by `e^(iθ)`.

We also added integration tests that connect all Stage 2 modules, and an
example program that prints all numerical examples of Entries 2–6.

## 2. Why did we build it?

Entry 5 showed rotation with a matrix. This entry shows the same rotation
with complex multiplication. Both views are important:

- The RoFormer paper (Su et al., 2021) **derives** RoPE with complex
  numbers. The proof of the relative-position property is very short in
  complex form.
- Many implementations (for example the original LLaMA code) **calculate**
  RoPE with complex numbers: they view each pair `(x, y)` as `x + iy` and
  multiply it by `e^(i·m·θⱼ)`.

If we understand that both views give the same result, we can read any RoPE
implementation.

## 3. Mathematical foundation

### 3.1 The exponential function `e^x`

`e ≈ 2.71828` is a mathematical constant (Euler's number). The function
`e^x` grows at a rate equal to its own value: its derivative is `e^x`.

`e^x` has a **power series** (Taylor series): an infinite sum of powers of
`x`. It is correct for every real number `x`:

```text
e^x = 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5! + ...
```

`n!` (read "n factorial") is the product `1·2·3·…·n`. Examples: `2! = 2`,
`3! = 6`, `4! = 24`, `5! = 120`.

Sine and cosine also have power series (for `θ` in radians; Entry 4,
section 3.2):

```text
cos θ = 1 − θ²/2! + θ⁴/4! − θ⁶/6! + ...
sin θ = θ − θ³/3! + θ⁵/5! − θ⁷/7! + ...
```

`cos` has only the even powers. `sin` has only the odd powers. The signs
alternate.

### 3.2 Derivation of Euler's formula

What is `e` to the power of an **imaginary** number `iθ`? Repeated
multiplication ("`e` times itself `iθ` times") has no meaning. Thus we use
the power series as the definition, and put `x = iθ` into it.

First, calculate the powers of `i`. They repeat with period 4:

```text
i⁰ = 1,   i¹ = i,   i² = −1,   i³ = i²·i = −i,   i⁴ = i²·i² = 1,   i⁵ = i, ...
```

Now put `x = iθ` into the series. Use `(iθ)ⁿ = iⁿ·θⁿ`:

```text
e^(iθ) = 1 + iθ + (iθ)²/2! + (iθ)³/3! + (iθ)⁴/4! + (iθ)⁵/5! + ...
       = 1 + iθ −   θ²/2!  − i·θ³/3! +   θ⁴/4!  + i·θ⁵/5! − ...
```

Collect the terms without `i` and the terms with `i`:

```text
e^(iθ) = (1 − θ²/2! + θ⁴/4! − ...)  +  i·(θ − θ³/3! + θ⁵/5! − ...)
       =          cos θ             +  i·          sin θ
```

This is **Euler's formula**:

```text
e^(iθ) = cos θ + i·sin θ
```

- `e` is Euler's number.
- `i` is the imaginary unit (`i² = −1`).
- `θ` is a real number: an angle in radians.

**Geometric interpretation.** `e^(iθ)` is the complex number with real part
`cos θ` and imaginary part `sin θ`. This is the point `(cos θ, sin θ)` on the
**unit circle** (Entry 4). Its magnitude is `√(cos²θ + sin²θ) = 1`, and its
phase is `θ`.

**Values at the four quarter turns:**

| θ | `e^(iθ)` | point |
|---|---|---|
| 0 | `1` | (1, 0) |
| π/2 | `i` | (0, 1) |
| π | `−1` | (−1, 0) |
| 3π/2 | `−i` | (0, −1) |

The row `θ = π` gives **Euler's identity**: `e^(iπ) + 1 = 0`. (Test:
`eulers_identity`.)

### 3.3 The polar form of a complex number

Every complex number `z = a + bi` has a magnitude `r = |z|` and a phase
`φ = arg(z)` (Entry 3). From the unit circle, `a = r·cos φ` and
`b = r·sin φ`. Thus:

```text
z = r·cos φ + i·r·sin φ = r·(cos φ + i·sin φ) = r·e^(iφ)
```

This is the **polar form**. `from_polar(r, φ)` implements it. `magnitude`
and `phase` go in the opposite direction. (Test: `polar_round_trip`.)

### 3.4 Why multiplication by `e^(iθ)` rotates

Multiply `z = r·e^(iφ)` by `e^(iθ)`. The exponent rule `e^x·e^y = e^(x+y)`
is also correct for complex exponents (it follows from the power series).
Thus:

```text
z·e^(iθ) = r·e^(iφ)·e^(iθ) = r·e^(i(φ + θ))
```

The result has:

- the **same magnitude** `r`, and
- the phase `φ + θ`: **the angle increased by `θ`**.

This is exactly a rotation by `θ` around the origin. Entry 3 found the same
rule from the other direction: multiplication multiplies the magnitudes
(`r · 1 = r`) and adds the phases (`φ + θ`).

### 3.5 The link to the rotation matrix

Calculate the product in components. Let `z = x + iy` and
`e^(iθ) = cos θ + i·sin θ`. With the multiplication formula of Entry 3
(`a = x`, `b = y`, `c = cos θ`, `d = sin θ`):

```text
(x + iy)(cos θ + i·sin θ) = (x·cos θ − y·sin θ) + i·(x·sin θ + y·cos θ)
```

Compare with the rotation matrix of Entry 5:

```text
[ cos θ  −sin θ ] [ x ]   [ x·cos θ − y·sin θ ]
[ sin θ   cos θ ] [ y ] = [ x·sin θ + y·cos θ ]
```

**The real part is `x'` and the imaginary part is `y'`. The results are
identical.** The integration test `complex_rotation_equals_matrix_rotation`
checks this for 9 angles and 6 vectors.

More generally: multiplication by any complex number `c + di` is the same as
multiplication by the matrix `[[c, −d], [d, c]]`. For `c + di = e^(iθ)`, this
matrix is `R(θ)`.

### 3.6 Properties in complex form

All the rotation properties of Entry 5 have a short complex form:

| Property | Matrix form (Entry 5) | Complex form |
|---|---|---|
| Composition | `R(α)·R(β) = R(α + β)` | `e^(iα)·e^(iβ) = e^(i(α + β))` |
| Inverse | `R(θ)ᵀ = R(−θ)` | `conj(e^(iθ)) = e^(−iθ)` |
| Length | `‖R(θ)·v‖ = ‖v‖` | `|z·e^(iθ)| = |z|` |
| Repetition | `R(θ)ⁿ = R(nθ)` | `(e^(iθ))ⁿ = e^(inθ)` (de Moivre's formula) |

**The conjugate reverses the rotation.**
`conj(cos θ + i·sin θ) = cos θ − i·sin θ = cos(−θ) + i·sin(−θ) = e^(−iθ)`.
(Test: `conjugate_reverses_a_rotation`.)

### 3.7 The relative-position property in complex form

This is the derivation of the RoFormer paper, for one pair of dimensions.

First, a fact: the 2-D dot product equals the real part of a product with a
conjugate. Let `u = a + bi` and `w = c + di`:

```text
u·conj(w) = (a + bi)(c − di) = (ac + bd) + (bc − ad)i
Re[u·conj(w)] = ac + bd = [a, b] · [c, d]
```

Now let `q` be a query at position `m` and `k` a key at position `n`, both
as complex numbers. RoPE rotates them: `q·e^(imθ)` and `k·e^(inθ)`. The
attention score is the dot product:

```text
score = Re[ q·e^(imθ) · conj(k·e^(inθ)) ]
      = Re[ q·e^(imθ) · conj(k)·e^(−inθ) ]       (conjugate of a product)
      = Re[ q·conj(k) · e^(i(m − n)θ) ]           (exponent rule)
```

The score depends on `m` and `n` **only through `m − n`**. (Tests:
`relative_rotation_property` in `complex.rs`, and the two
`relative_position_property_*` tests in `tests/numerical_tests.rs`.)

**Numerical example.** `θ = 0.37`, `q = [0.8, −1.1]`, `k = [−0.3, 1.9]`.
The pairs of positions `(m, n) = (0, 3), (1, 4), (10, 13), (100, 103)` all
have `n − m = 3`. They all give the same score (within `1e-12`). The pair
`(0, 4)` has a different distance and gives a different score.

The matrix form (Entry 5, section 3.10) gave the same result:
`qᵀ·R((n − m)θ)·k`. The complex and matrix derivations are two views of one
fact.

### 3.8 Numerical examples

**Example 1: Euler's identity.**

```text
from_polar(1, π) = cos(π) + i·sin(π) = −1 + 0i        (exact mathematics)
                 = −1 + 1.2246467991473532e-16·i       (f64)
```

The imaginary part is `sin(PI)`, which is not zero (Entry 4, section 3.8).

**Example 2: rotation of `1 + i` by `π/4`.**

```text
1 + i:   magnitude √2 ≈ 1.41421,   phase π/4 ≈ 0.78540
(1 + i)·e^(iπ/4) = (1 + i)(√2/2 + i·√2/2)
                 = (√2/2 − √2/2) + i·(√2/2 + √2/2)
                 = 0 + i·√2
result:  magnitude √2,   phase π/2 ≈ 1.57080
```

The magnitude stays the same. The phase increases by `π/4`.

**Example 3: complex and matrix rotation of `[2, 3]` by `θ = 0.5`.**
`cos(0.5) ≈ 0.877583`, `sin(0.5) ≈ 0.479426`.

```text
x' = 2·0.877583 − 3·0.479426 ≈ 0.316889
y' = 2·0.479426 + 3·0.877583 ≈ 3.591599
```

The program prints both results with 15 decimals. They are bit-for-bit
equal: `(0.316888507968137, 3.591598762879524)`. The two methods do the same
arithmetic operations in the same order.

### 3.9 Precision considerations

- `from_polar(1, θ)` has magnitude 1 only up to rounding. `cos²θ + sin²θ`
  in `f64` differs from 1 by a few `f64::EPSILON`. (Test:
  `unit_complex_numbers_have_magnitude_one`.)
- **De Moivre in `f64`.** Five multiplications `w·w·w·w·w` and one
  `from_polar(1, 5θ)` agree within `1e-12`, but not exactly. Each
  multiplication adds rounding errors. (Test: `de_moivre_formula`.)
- **Accumulated drift.** The integration test
  `many_small_rotations_drift_more_than_one_large_rotation` rotates `[1, 0]`
  10 000 times by `0.001`. We measured on this computer: the length error
  after 10 000 steps is `7.8e-14` (about 350 × `f64::EPSILON`). One rotation
  by `10.0` has a length error of exactly 0. **Conclusion for RoPE:**
  calculate `e^(i·m·θ)` directly from the angle `m·θ`. Do not calculate it as
  `(e^(iθ))ᵐ` with repeated multiplication.
- Large angles `m·θ` have a large absolute error in `f64` (Entry 4,
  section 3.8). This affects `from_polar` in the same way as `sin_cos`.

## 4. Rust concepts learned

### 4.1 Reuse of an operator trait inside a method

```rust
pub fn rotate(self, theta: f64) -> Self {
    self * Self::from_polar(1.0, theta)
}
```

`rotate` uses the `Mul` implementation of Entry 3. The expression
`self * ...` calls `<Complex as Mul>::mul`. We do not duplicate the
multiplication formula. If the formula had a bug, there would be only one
place to correct it.

`Self::from_polar` calls an associated function of the same type. `Self` is
`Complex` here.

### 4.2 Shared helpers across modules

`from_polar` uses `sin_cos` from `trigonometry`, the same as
`Matrix2::rotation`. Thus both rotation methods use the same sine and cosine
values. This is one reason that the complex and the matrix results are
bit-for-bit equal in the example.

### 4.3 Integration tests (`tests/*.rs`)

`tests/numerical_tests.rs` is a **separate crate**. It can use only the
public items of the library:

```rust
use rust_rope::math::complex::Complex;
use rust_rope::math::matrix::{Matrix2, Vector, Vector2, rotate_vector};
```

If a test in this file needs a private item, the test does not compile.
This tests the **public API** from the view of a user. Unit tests (inside
`#[cfg(test)]` modules) test the internal details.

### 4.4 `const` arrays of test data

```rust
const ANGLES: [f64; 9] = [-5.0, -1.0, ...];
const VECTORS: [Vector2; 6] = [[0.0, 0.0], ...];
```

A `const` item is a compile-time value. The compiler puts a copy of it into
each place that uses it. The size is part of the type (`[f64; 9]`), so the
array cannot have a wrong length. We loop over these arrays: `for theta in
ANGLES`. In Rust 2021 and later, `for x in array` iterates over the values
of the array (not references).

### 4.5 `expect` versus `unwrap`

```rust
.dot(&Vector::from_slice(&b)).expect("2-D vectors always have the same dimension")
```

`expect(msg)` is like `unwrap()`: it returns the `Ok` value, or panics if
the result is `Err`. The message documents **why** we know that the error
cannot happen. If the code is changed later and the error happens, the
panic message explains the broken assumption.

### 4.6 Closures that capture variables

```rust
let score = |m: f64, n: f64| dot(rotate_vector(q, m * theta), rotate_vector(k, n * theta));
```

The closure uses `q`, `k` and `theta` from the surrounding function. It
**captures** them. `q`, `k` and `theta` are `Copy`, and the closure only
reads them, so it captures them by reference. The closure cannot outlive
these variables; the compiler checks this.

### 4.7 Examples (`examples/*.rs`)

`examples/math_foundations.rs` is also a separate crate with a `main`
function. `cargo run --example math_foundations` builds and runs it.
`cargo test` compiles all examples, so an example that does not compile
makes the test command fail. This keeps the examples correct.

## 5. Memory and execution

**Execution of `z.rotate(θ)`:**

1. `z` (16 bytes) is copied into `self`.
2. `from_polar(1.0, θ)` calls `sin_cos(θ)` and creates a `Complex` with
   `1.0·cos θ` and `1.0·sin θ`. (The compiler can remove the multiplication
   by `1.0` because `1.0·x = x` exactly in IEEE 754.)
3. `self * w` calculates 4 multiplications and 2 additions.
4. The result is returned in registers.

No heap allocation. No borrowing. All values are `Copy`.

**Comparison of the two rotation methods:**

| | `rotate_vector(v, θ)` (Entry 5) | `z.rotate(θ)` (this entry) |
|---|---|---|
| Input type | `[f64; 2]` | `Complex` |
| Intermediate value | `Matrix2` (32 bytes) | `Complex` (16 bytes) |
| `sin_cos` calls | 1 | 1 |
| Arithmetic | 4 mul + 2 add | 4 mul + 2 add |
| Heap | none | none |

After inlining and optimization, both produce practically the same machine
code.

## 6. Engineering decisions

1. **No general complex exponential `exp(a + bi)`.** The task says: add it
   only if it is useful for learning. RoPE needs only `e^(iθ)` for real `θ`.
   `exp(a + bi) = e^a·e^(ib)` would be one more line, but no later stage uses
   it. `from_polar(1.0, θ)` is `e^(iθ)` exactly. The name `from_polar` is
   the standard name for this operation (the C++ standard library and the
   Rust crate `num-complex` use it too).
2. **`rotate` is a method that returns a new value.** `Complex` is `Copy`, so
   a new value is cheap. An in-place version (`&mut self`) would give no
   benefit.
3. **`rotate` reuses `Mul`.** One multiplication formula in the code.
4. **Integration tests for the cross-module properties.** The fact
   "complex rotation = matrix rotation" belongs to neither module alone. It
   is a property of the public API, so it is in `tests/`.
5. **A new example file.** The task structure lists only
   `examples/basic_rope.rs`, which is for RoPE (Stage 4). We added
   `examples/math_foundations.rs` so that the expected outputs in
   `explain.md` can be checked by running one command. All outputs in
   Entries 2–6 come from this program.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| `Complex::exp(self)` for any complex number | General, mathematically complete | Not needed. Speculative |
| `Complex::cis(θ)` ("cos + i sin") | Short, a known name | Less common in Rust. `from_polar(1.0, θ)` is clearer |
| Rotation via `Matrix2` only | One method | The RoFormer derivation and many implementations use complex form |
| Rotation by repeated multiplication `wᵐ` | Avoids `sin_cos` per position | Accumulates errors (section 3.9). Slower for large `m` unless done by squaring |
| A lookup table of `e^(i·m·θ)` | Fast for many tokens | Needs memory. Planned as the RoPE cache in Stage 3/6 |

## 8. Performance considerations

- `from_polar`: one `sin_cos` call and 2 multiplications. O(1).
- `rotate`: one `sin_cos` call, 4 multiplications, 2 additions. O(1).
- The cost is dominated by `sin_cos` (about 20 to 100 cycles), not by the
  arithmetic (about 6 cycles or less).
- **Implication for RoPE.** For a sequence of `L` tokens and `d/2` pairs,
  RoPE needs `L·d/2` values `e^(i·m·θⱼ)`. They do not depend on the input
  data. Calculate them one time and store them (a table of `L·d/2` complex
  numbers = `L·d/2·16` bytes in `f64`). Then each rotation is only 4
  multiplications and 2 additions. Stage 3 plans this.
- **Memory layout.** An array of `Complex` stores `re, im, re, im, ...`.
  This is the "interleaved" layout. Stage 4 compares it with the
  "split-half" layout that some models use.

## 9. Practical example

```rust
use rust_rope::math::complex::Complex;
use rust_rope::math::matrix::rotate_vector;
use std::f64::consts::{FRAC_PI_4, PI};

let e_i_pi = Complex::from_polar(1.0, PI);
println!("e^(iπ)          = {e_i_pi:?}");

let z = Complex::new(1.0, 1.0);
let rotated = z.rotate(FRAC_PI_4);
println!("(1 + i)·e^(iπ/4) = {rotated:?}");
println!("magnitude {} -> {}, phase {} -> {}",
         z.magnitude(), rotated.magnitude(), z.phase(), rotated.phase());

let by_complex = Complex::new(2.0, 3.0).rotate(0.5);
let by_matrix = rotate_vector([2.0, 3.0], 0.5);
println!("complex: ({:.15}, {:.15})", by_complex.re(), by_complex.im());
println!("matrix:  ({:.15}, {:.15})", by_matrix[0], by_matrix[1]);
```

Expected output (from `cargo run --example math_foundations`):

```text
e^(iπ)          = Complex { re: -1.0, im: 1.2246467991473532e-16 }
(1 + i)·e^(iπ/4) = Complex { re: 0.0, im: 1.4142135623730951 }
magnitude 1.4142135623730951 -> 1.4142135623730951, phase 0.7853981633974483 -> 1.5707963267948966
complex: (0.316888507968137, 3.591598762879524)
matrix:  (0.316888507968137, 3.591598762879524)
```

Note: the real part of `(1 + i)·e^(iπ/4)` is printed as exactly `0.0`.
The calculation is `1·cos(π/4) − 1·sin(π/4)`. In `f64`, `cos(π/4)` and
`sin(π/4)` happen to round to the same value, so the difference is exactly
zero. This is luck, not a guarantee.

## 10. What we learned

**Mathematics:**

- Euler's formula `e^(iθ) = cos θ + i·sin θ`. We derived it from the power
  series of `e^x`, `cos θ` and `sin θ`, with the powers of `i`.
- `e^(iθ)` is the point at angle `θ` on the unit circle. Every complex
  number is `r·e^(iφ)` (the polar form).
- Multiplication by `e^(iθ)` keeps the magnitude and adds `θ` to the phase:
  it is a rotation.
- In components, `(x + iy)·e^(iθ)` is exactly `R(θ)·[x, y]`.
- The conjugate reverses the rotation. `Re[u·conj(w)]` is the 2-D dot
  product.
- The RoPE score `Re[q·e^(imθ)·conj(k·e^(inθ))] = Re[q·conj(k)·e^(i(m−n)θ)]`
  depends only on `m − n`.
- Calculate `e^(i·m·θ)` directly. Repeated multiplication accumulates
  rounding errors.

**Rust:**

- A method can reuse an operator trait implementation (`self * w`).
- Integration tests are separate crates. They test only the public API.
- `const` arrays as test data. `for x in array` iterates by value.
- `expect` documents why an error cannot occur.
- Closures capture surrounding variables.
- `cargo test` also compiles the examples.

---

## Roadmap status after Entry 6

Stage 1 and Stage 2 are complete. The roadmap table at the end of Entry 1
shows the state at that time. This table shows the current state.

| Stage | Status |
|---|---|
| 1 Project foundation | ✅ Done (Entry 1) |
| 2 Mathematical foundations | ✅ Done (Entries 2–6) |
| 3 RoPE fundamentals | ⏳ Planned (next) |
| 4 RoPE implementation | ⏳ Planned |
| 5 Advanced RoPE | ⏳ Planned |
| 6 Performance engineering | ⏳ Planned |
| 7 Validation and experimentation | ⏳ Planned |


---

# Entry 7 — Configuration, validation and inverse frequencies (Stage 3, Component 1)

**Date:** 2026-11-01
**Stage:** 3 (RoPE fundamentals)
**Files:** `src/utils/validation.rs`, `src/rope/config.rs`,
`src/rope/frequencies.rs` (the frequency part), `tests/rope_tests.rs`,
`tests/numerical_tests.rs`, `examples/rope_frequencies.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `RopeError` (one error enum for all invalid RoPE input) | ✅ Implemented |
| `validate_dimension`, `validate_base` | ✅ Implemented |
| `RopeConfig { d, b }`, `DEFAULT_BASE = 10000` | ✅ Implemented |
| `inverse_frequency(config, j)` = `θⱼ` | ✅ Implemented |
| `inverse_frequencies(config)` = `[θ₀, ..., θ_(d/2−1)]` | ✅ Implemented |
| Positions and angles | ✅ Implemented in Entries 8 and 9 |
| Sine, cosine and rotation of embeddings | ⏳ Planned (Stage 4) |
| Frequency scaling (NTK, YaRN) | ⏳ Planned (Stage 5) |

---

## 1. What did we build?

We built the first RoPE-specific code. It calculates the **inverse
frequencies** `θⱼ`. An inverse frequency is the rotation speed of one pair of
dimensions, in radians per token position.

The code has three parts:

1. `utils::validation` checks the input. It has one error type,
   `RopeError`, and one check function for each kind of input.
2. `rope::config::RopeConfig` holds the two parameters of the formula: the
   embedding dimension `d` and the frequency base `b`. It checks both values
   when you create it.
3. `rope::frequencies` calculates `θⱼ = b^(−2j/d)` for one pair or for all
   pairs.

## 2. Why did we build it?

RoPE rotates each pair of embedding dimensions by an angle. The angle is the
token position `m` multiplied by the rotation speed `θⱼ` of the pair. Thus
we must know `θⱼ` before we can calculate an angle, and we must know the angle
before we can rotate. The frequencies are the first step of the data flow:

```text
(d, b) ──► θⱼ ──► φ = m·θⱼ ──► cos φ, sin φ ──► rotation of the pair
 this entry       Entry 9        Stage 4          Stage 4
```

## 3. Mathematical foundation

### 3.1 Notation: `i` in the task, `j` in the project

The task text writes the formula as `θ_i = 1 / base^(2i/d)`. The project
uses `j` for the pair index, because `i` is the imaginary unit `√−1` in
`src/math/complex.rs` (see `docs/mathematics.md`, section 1). The two
formulas are identical. Only the name of the index is different:

```text
θ_i = 1 / base^(2i/d)       (task notation)
θⱼ  = 1 / b^(2j/d)          (project notation)
```

### 3.2 The variables

| Symbol | Name | Meaning | Example |
|---|---|---|---|
| `d` | rotary embedding dimension | The number of dimensions that RoPE rotates. Usually the dimension of one attention head. It must be even. | 8, 64, 128 |
| `P = d/2` | number of pairs | RoPE groups the `d` dimensions into `d/2` pairs. | 4 for `d = 8` |
| `j` | dimension-pair index | Selects one pair. `j = 0, 1, ..., d/2 − 1`. Pair `j` holds the dimensions `2j` and `2j + 1`. | 0, 1, 2, 3 |
| `b` | frequency base | A constant that sets the slowest rotation speed. | 10000 |
| `θⱼ` | inverse frequency of pair `j` | The angle in radians that pair `j` rotates for each step of one token position. | 1, 0.1, 0.01, 0.001 |

**Why "inverse frequency"?** In physics, a frequency counts cycles per unit
of time. Here, `θⱼ` is an angle per position. The name comes from the form
`θⱼ = 1 / b^(2j/d)`: it is "one divided by" the number `b^(2j/d)`. Also,
the wavelength of pair `j` (section 3.7) is proportional to `1/θⱼ`.

### 3.3 Why RoPE uses different frequencies for different pairs

Imagine that all pairs use the same speed, for example `θ = 1` radian per
position. Then the angle at position `m` is `m` radians. A rotation by `2π`
(one full turn) is the same as no rotation. Thus the position information
**wraps around**. Example: `44 mod 2π ≈ 0.0177` (measured with `f64`).
Thus position 44 has almost the same rotation as position 0. The model cannot
separate the two positions.

Now imagine that all pairs use a very slow speed, for example
`θ = 0.0001`. Then the rotation does not wrap around for 62 832 positions.
But neighbouring positions get almost the same angle (difference 0.0001
radian). The model cannot separate position 5 from position 6.

A clock solves the same problem with several hands. The second hand moves
fast: it separates neighbouring seconds, but it repeats every minute. The
hour hand moves slowly: it never repeats in one day, but it cannot show
seconds. **Together**, the hands show the time without ambiguity.

RoPE uses `d/2` "hands" with speeds from fast (`θ₀ = 1`) to slow
(`θ_(d/2−1) ≈ 1/b`). Fast pairs encode small distances. Slow pairs encode
large distances.

### 3.4 Why the exponent is `2j/d`

The exponent `2j/d` has two parts: `2j` and the division by `d`.

**Part 1: why `2j` and not `j`.** Pair `j` holds the dimensions `2j` and
`2j + 1`. Thus `2j` is the **index of the first dimension** of the pair.
The exponent `2j/d` is "the dimension index divided by the dimension". The
original Transformer (Vaswani et al., 2017) uses the same exponent in its
sinusoidal position encoding: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`.
RoPE keeps this formula.

There is a second view. `2j/d = j/(d/2) = j/P`. This is the pair index
divided by the number of pairs: the **fraction of the pairs** before pair
`j`. It goes from 0 (first pair) to almost 1 (last pair).

What happens with `j/d` instead? The exponent then goes only from 0 to about
`0.5`. The slowest speed is `b^(−0.5) = 1/√b = 0.01` for `b = 10000`. The
pairs use only half of the available speed range, and the longest wavelength
is only about 628 positions instead of up to almost 62 832.

**Part 2: why we divide by `d`.** The division **normalizes** the exponent
to the range `[0, 1)`. Thus the speed range is always from `1` down to about
`1/b`, for every dimension `d`. The dimension controls only how many speeds
lie in this range, not the range itself.

What happens without the division? With `θⱼ = b^(−2j)` and `d = 128`, the
last pair has `θ₆₃ = 10000^(−126) = 10^(−504)`. This is smaller than the
smallest positive `f64` (about `4.9e−324`). We measured: `f64` gives exactly
`0`. Even for small `d`, the speeds would fall by a factor of `b² = 10⁸`
from one pair to the next. Most pairs would never rotate.

### 3.5 Why the exponent increases with `j`, and why `θⱼ` decreases

The exponent `2j/d` increases by `2/d` for each step of `j`:

```text
j:       0      1      2      ...    d/2 − 1
2j/d:    0      2/d    4/d    ...    (d − 2)/d
```

For `b > 1`, the power `b^x` is an **increasing** function of `x`. Proof:
`b^x = e^(x·ln b)`, and `ln b > 0` for `b > 1`. A larger `x` gives a larger
exponent `x·ln b`, and `e^y` increases with `y`.

Thus `b^(2j/d)` increases with `j`, and its inverse `θⱼ = 1/b^(2j/d)`
**decreases** with `j`. The higher the pair index, the slower the pair.
(Test: `inverse_frequencies_decrease_strictly`.)

This is also the reason for the rule `b > 1` (section 6). With `b = 1`, all
`θⱼ = 1`. With `0 < b < 1`, `ln b < 0`, and the order is reversed.

### 3.6 A geometric sequence

Divide two neighbouring frequencies:

```text
θ_(j+1) / θⱼ = b^(−2(j+1)/d) / b^(−2j/d) = b^(−2(j+1)/d + 2j/d) = b^(−2/d)
```

The ratio `r = b^(−2/d)` does not depend on `j`. A sequence with a constant
ratio is a **geometric sequence**: `θⱼ = r^j`. The logarithm is a straight
line:

```text
ln θⱼ = −(2j/d)·ln b
```

Thus the speeds are **evenly spaced on a logarithmic scale**. Each pair is
slower than the previous pair by the same factor. (Test:
`inverse_frequencies_form_a_geometric_sequence`.)

**The range.** `0 ≤ 2j/d ≤ (d − 2)/d < 1`. Thus:

```text
1/b < θⱼ ≤ 1
```

`θ₀ = 1` exactly. The last value `θ_(d/2−1) = b^(−(d−2)/d)` is a little
larger than `1/b`. (Test: `inverse_frequencies_stay_between_one_over_base_and_one`.)

### 3.7 The wavelength of a pair

A pair returns to its start direction after a full turn, `2π` radians. At a
speed of `θⱼ` radians per position, this takes:

```text
λⱼ = 2π / θⱼ = 2π · b^(2j/d)      positions
```

`λⱼ` is the **wavelength** of pair `j`. It goes from `2π ≈ 6.28` positions
(pair 0) to almost `2π·b ≈ 62 832` positions (last pair, `b = 10000`).

### 3.8 Numerical derivation: `d = 8`, `b = 10000`

`d = 8` gives `P = 8/2 = 4` pairs: `j = 0, 1, 2, 3`.
We write `b = 10000 = 10⁴`. Then `b^x = (10⁴)^x = 10^(4x)`.

**Pair `j = 0`** (dimensions 0 and 1):

```text
exponent = 2·0 / 8 = 0 / 8 = 0
b^0      = 10000^0 = 1                  (every number to the power 0 is 1)
θ₀       = 1 / 1 = 1
```

**Pair `j = 1`** (dimensions 2 and 3):

```text
exponent = 2·1 / 8 = 2/8 = 0.25
b^0.25   = (10⁴)^0.25 = 10^(4·0.25) = 10^1 = 10      (the fourth root of 10000)
θ₁       = 1 / 10 = 0.1
```

**Pair `j = 2`** (dimensions 4 and 5):

```text
exponent = 2·2 / 8 = 4/8 = 0.5
b^0.5    = (10⁴)^0.5 = 10^2 = 100                    (the square root of 10000)
θ₂       = 1 / 100 = 0.01
```

**Pair `j = 3`** (dimensions 6 and 7):

```text
exponent = 2·3 / 8 = 6/8 = 0.75
b^0.75   = (10⁴)^0.75 = 10^3 = 1000
θ₃       = 1 / 1000 = 0.001
```

**Summary:**

| `j` | dimensions | `2j/d` | `b^(2j/d)` | `θⱼ` | wavelength `2π/θⱼ` |
|---|---|---|---|---|---|
| 0 | 0, 1 | 0 | 1 | 1 | 6.28 |
| 1 | 2, 3 | 0.25 | 10 | 0.1 | 62.83 |
| 2 | 4, 5 | 0.5 | 100 | 0.01 | 628.32 |
| 3 | 6, 7 | 0.75 | 1000 | 0.001 | 6283.19 |

**Interpretation.**

- The ratio is `r = 10000^(−2/8) = 10^(−1) = 0.1`. Each pair is 10 times
  slower than the previous pair.
- Pair 0 makes a full turn every 6.28 tokens. It separates neighbouring
  tokens well, but it repeats often.
- Pair 3 needs 6283 tokens for one turn. In a sequence of 100 tokens it
  turns only `0.1` radian. Its angle changes slowly and never repeats, so it
  carries "coarse" long-distance information.
- `d = 8` is small, so the ratio is large (0.1). With `d = 128`, the ratio is
  `10000^(−2/128) = 10^(−1/16) ≈ 0.866`. The 64 speeds then cover the same
  range `(1/b, 1]`, but much more densely.

### 3.9 Other dimensions and bases

Measured with `cargo run --example rope_frequencies` (`b = 10000`):

| `d` | pairs | ratio `b^(−2/d)` | `θ₁` | last `θ` | longest wavelength |
|---|---|---|---|---|---|
| 8 | 4 | 0.1 | 0.1 | 1.0e−3 | 6 283 |
| 64 | 32 | 0.7499 | 7.498942e−1 | 1.333521e−4 | 47 117 |
| 128 | 64 | 0.8660 | 8.659643e−1 | 1.154782e−4 | 54 410 |

A **larger base** makes every pair (except pair 0) slower, because
`b^(−x)` decreases with `b` for a fixed `x > 0`. Some models use
`b = 500 000` (LLaMA 3) or `b = 1 000 000`, so that the slow pairs do not
wrap around in very long contexts. (Test: `larger_base_gives_slower_pairs`.)
Stage 5 studies this in detail.

### 3.10 Numerical precision

- We use `f64`. An `f64` has 53 significant bits (about 16 decimal digits).
- **`θ₀ = 1` is exact.** IEEE 754 (the floating-point standard) defines
  `pow(x, ±0) = 1` for every `x`. (Test: `first_frequency_is_exactly_one`.)
- **Exact exponents.** For `d = 8`, the exponents `0, 0.25, 0.5, 0.75` are
  exact binary fractions. For `d = 128`, `2j/128 = j/64` is also exact. For
  `d = 6`, `2/6 = 1/3` is not exact, so the exponent itself has a rounding
  error of about `1e−16`.
- **One rounding, not two.** We calculate `b.powf(−x)`, not
  `1.0 / b.powf(x)`. The second form rounds two times (the power and the
  division). For `d = 8`, both forms happen to give the same `f64` values
  (`0.1`, `0.01`, `0.001` are the nearest `f64` values to the exact results).
- **The printed ratio.** `θ₂/θ₁` prints as `0.09999999999999999`. The value
  `0.1` has no exact binary representation, so `0.01 / 0.1` in `f64` is not
  exactly `0.1`. This is a rounding effect of the division, not an error in
  `θ`.
- **`f32` comparison.** We measured: `f32` gives inverse frequencies with a
  relative error of up to `5.4e−8` (compared with `f64`). Many Python
  implementations use `f32`. Entry 9 shows the effect of this error on large
  angles.

### 3.11 Where a Transformer uses this

The inverse frequencies depend only on `d` and `b`, not on the input text.
Thus a model calculates them **one time**. All attention layers and all
heads use the same list. Many implementations store the list as a constant
buffer with the name `inv_freq`. The model has no trainable parameter for
RoPE.

## 4. Rust concepts learned

### 4.1 An `enum` with data for errors

```rust
pub enum RopeError {
    ZeroDimension,
    OddDimension { dimension: usize },
    InvalidBase { base: f64 },
    // ...
}
```

An **enum** (enumeration) is a type whose value is exactly **one** of its
variants. A variant can hold no data (`ZeroDimension`) or named fields
(`OddDimension { dimension }`). The fields let the error message show the
invalid value.

Why one enum and not one struct for each error (as `DimensionMismatch` in
Entry 2)? A RoPE function can fail for several reasons. With one enum, every
RoPE function has the same error type, and `?` works between them without a
conversion. `match` on the enum must handle every variant. If we add a
variant later, the compiler shows every `match` that we must update.

Small independent example:

```rust
enum Shape { Circle { r: f64 }, Square { side: f64 } }

fn area(s: Shape) -> f64 {
    match s {
        Shape::Circle { r } => 3.14159 * r * r,
        Shape::Square { side } => side * side,
    }
}
```

### 4.2 `Display` and `std::error::Error`

`impl fmt::Display for RopeError` gives a human-readable message, so that
`error.to_string()` and `println!("{error}")` work. The empty
`impl std::error::Error for RopeError {}` marks the type as an error. Then
`?` can convert it into `Box<dyn std::error::Error>`. The example program
uses this: `fn main() -> Result<(), Box<dyn std::error::Error>>`.

Example output (from the test `errors_have_readable_messages`):

```text
the embedding dimension 7 is odd; RoPE needs an even dimension, because it rotates the dimensions in pairs
```

### 4.3 `Result<(), E>` and the unit type

```rust
pub fn validate_dimension(dimension: usize) -> Result<(), RopeError>
```

A check has no value to return when the input is valid. The type `()`
("unit") is the empty value. `Ok(())` means "valid". `Err(e)` means "invalid,
for the reason `e`". The caller cannot ignore the result silently: an unused
`Result` gives a compiler warning (`#[must_use]`).

### 4.4 The `?` operator

```rust
pub fn new(dimension: usize, base: f64) -> Result<Self, RopeError> {
    validate_dimension(dimension)?;
    validate_base(base)?;
    Ok(RopeConfig { dimension, base })
}
```

`expr?` means: if `expr` is `Err(e)`, return `Err(e)` from this function
now. If it is `Ok(v)`, continue with the value `v`. The order of the checks
is the order of the lines. The test `dimension_is_checked_before_base`
shows this: `RopeConfig::new(3, −1.0)` reports the dimension error.

### 4.5 "Validate one time, then trust the type"

The fields of `RopeConfig` are private. The only way to create a
`RopeConfig` outside the module is `RopeConfig::new` (or
`with_default_base`), and it checks the values. Thus:

> Every `RopeConfig` that exists has an even `d ≥ 2` and a finite `b > 1`.

This guarantee lets `inverse_frequencies(&RopeConfig) -> Vec<f64>` return a
plain `Vec`, not a `Result`. It cannot fail. The type carries the proof of
validity. This technique is sometimes called "parse, don't validate".

Without private fields, a caller could write
`RopeConfig { dimension: 7, base: 1.0 }`, and every function would have to
check again.

### 4.6 Comparisons with NaN

```rust
if base > 1.0 && base.is_finite() { Ok(()) } else { Err(...) }
```

`NaN` ("not a number") is a special `f64` value, for example the result of
`0.0 / 0.0`. **Every comparison with NaN is `false`.** Thus `NaN > 1.0` is
`false`, and the check rejects NaN without a special case.

Be careful with the negated form. `if base <= 1.0 { error }` does **not**
reject NaN, because `NaN <= 1.0` is also `false`. We write the condition for
the **valid** case, so that NaN falls into the `else` branch.

A test cannot use `assert_eq!` for an error that contains NaN, because
`NaN != NaN`. It uses `matches!` instead:

```rust
assert!(matches!(validate_base(f64::NAN), Err(RopeError::InvalidBase { .. })));
```

`..` means "ignore the remaining fields".

### 4.7 `Vec<f64>`: what it is and how it works

`Vec<f64>` is a **growable array** of `f64` values. It has two parts:

- **On the stack** (24 bytes on a 64-bit computer, measured with
  `std::mem::size_of`): a pointer to the data, the **capacity** (how many
  elements fit in the allocated memory), and the **length** (how many
  elements are in use).
- **On the heap**: one contiguous block of `capacity · 8` bytes.

```text
stack:  Vec { ptr ──┐, cap: 4, len: 4 }
                    ▼
heap:   [ 1.0 | 0.1 | 0.01 | 0.001 ]      4 × 8 = 32 bytes
```

If `push` adds an element and `len == cap`, the `Vec` allocates a larger
block (usually double the size), copies the elements, and frees the old
block. This is called **reallocation**. We avoid it when we know the final
size (section 4.8).

When the `Vec` goes out of scope, Rust calls its destructor (`drop`). The
destructor frees the heap block. There is no garbage collector.

Why `Vec<f64>` and not `[f64; N]`? The number of pairs `d/2` is a run-time
value (from the configuration). An array `[f64; N]` needs `N` at compile
time.

### 4.8 Iterators: `0..n`, `map` and `collect`

```rust
(0..config.num_pairs())
    .map(|j| inverse_frequency_unchecked(j, dimension, base))
    .collect()
```

- `0..n` is a `Range<usize>`. It is an **iterator**: a value with a method
  `next()` that returns `Some(0)`, `Some(1)`, ..., `Some(n − 1)`, then `None`.
- `map(f)` creates a new iterator. Each call of `next()` takes the next
  value `j` and returns `f(j)`. `map` itself does no work. Iterators are
  **lazy**.
- `collect()` calls `next()` until `None` and puts the values in a
  collection. The return type of the function (`Vec<f64>`) tells `collect`
  which collection to build.

**One allocation.** A `Range` knows its exact length, and `map` keeps this
knowledge. (The iterator reports it through its `size_hint` method.) Thus
`collect` allocates the final capacity one time. No reallocation occurs.

**How the compiler sees it.** The compiler inlines `next`, `map` and the
closure. The optimized machine code is the same as a hand-written loop:

```rust
let mut v = Vec::with_capacity(n);
for j in 0..n {
    v.push(inverse_frequency_unchecked(j, dimension, base));
}
```

### 4.9 Closures that capture variables

`|j| inverse_frequency_unchecked(j, dimension, base)` is a **closure**: an
anonymous function. It uses `dimension` and `base` from the surrounding
function. It **captures** them. We copied `config.dimension()` and
`config.base()` into local variables first. Then the closure captures two
small `Copy` values, not the reference `config`.

### 4.10 `as` conversions

```rust
let exponent = (2 * pair_index) as f64 / dimension as f64;
```

`as f64` converts an integer to `f64`. Rust never converts number types
automatically: `2 * pair_index / dimension` would be an **integer** division
(`2 / 8 = 0` in integers). We convert both numbers first, then divide in
floating point. The conversion is exact for integers up to `2^53`.

### 4.11 A private helper with `#[inline]`

```rust
#[inline]
fn inverse_frequency_unchecked(pair_index: usize, dimension: usize, base: f64) -> f64
```

The formula is in exactly one place. Both public functions call it. The
helper is private (no `pub`), because it does not check its input. A public
caller must go through the checked functions. `#[inline]` suggests to the
compiler that it copy the function body into the caller. For a two-line
function this removes the call overhead.

### 4.12 Borrowing the configuration: `&RopeConfig`

```rust
pub fn inverse_frequencies(config: &RopeConfig) -> Vec<f64>
```

The function only **reads** the configuration. A shared reference `&` lends
the value to the function. The caller keeps ownership and can use `config`
again after the call. `RopeConfig` is `Copy` (16 bytes), so we could also
pass it by value. We pass a reference because (1) it states the intent
"read only", and (2) the API stays the same if a later stage adds a field
that is not `Copy` (for example a `Vec` of scaling factors in Stage 5).

### 4.13 Documentation tests with `?`

```rust
/// let config = RopeConfig::new(8, 10_000.0)?;
/// # Ok::<(), rust_rope::utils::validation::RopeError>(())
```

`cargo test` compiles and runs the examples in `///` comments. A doc test is
the body of a hidden function. The hidden last line (`#` hides it in the
documentation) makes this function return `Result`, so `?` works.

## 5. Memory and execution

| Value | Size (measured) | Location | Notes |
|---|---|---|---|
| `RopeConfig` | 16 B | stack | `usize` + `f64`. `Copy`. |
| `RopeError` | 24 B | stack | The compiler stores the variant tag inside the free bits of the `Option<usize>` field. |
| `Vec<f64>` header | 24 B | stack | pointer, capacity, length |
| frequency data | `d/2 · 8` B | heap | 32 B for `d = 8`, 512 B for `d = 128` |

**Execution of `inverse_frequencies(&config)`:**

1. The function reads `d` and `b` from the borrowed configuration.
2. `collect` allocates one heap block of `d/2 · 8` bytes.
3. For each `j`, the code calculates `2j/d` and one `powf`, and writes the
   result into the block.
4. The function **moves** the `Vec` to the caller. A move copies only the
   24-byte header. The heap data stays where it is.
5. When the caller's variable goes out of scope, `drop` frees the block.

**Ownership summary:** the configuration is borrowed (`&`). The frequencies
are owned by the caller. No value is cloned.

## 6. Engineering decisions

1. **The rule `b > 1`, not `b > 0`.** The formula is defined for every
   `b > 0`. But `b = 1` gives equal speeds, and `b < 1` reverses the order of
   the speeds. Neither is RoPE. A strict rule detects a configuration
   mistake early. With this rule, "the frequencies decrease" is true for
   every valid configuration.
2. **`d = 0` is invalid.** Zero is even, but it gives zero pairs. Every later
   function would have to handle the empty case.
3. **Validation in `utils::validation`.** Several modules use the same
   checks (Entries 8 and 9 add more). One place gives one rule and one
   message.
4. **`RopeConfig` validates in its constructor.** The frequency functions
   then need no `Result` (section 4.5).
5. **`inverse_frequency` (single) returns `Result`.** The pair index `j` is
   a separate input from the caller. It can be out of range.
6. **`f64`.** This is an educational implementation and a future reference
   for `f32` code. `f64` keeps the rounding errors far below the effects
   that we want to study. Stage 6 can compare `f32`.
7. **`b.powf(−x)` instead of `1.0 / b.powf(x)`.** One rounding instead of
   two (section 3.10).
8. **No external error crate** (for example `thiserror`). The standard
   library is sufficient: one `Display` implementation and one empty
   `Error` implementation.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| `(-x * b.ln()).exp()` | The same mathematics; one `ln` for all pairs | Two operations per value; `powf` is clearer |
| Repeated multiplication `θ_(j+1) = θⱼ · r` | One `powf` in total | Errors accumulate. Measured: up to `1.6e−15` relative error for `d = 64`, compared with about `1e−16` for `powf` |
| Methods on `RopeConfig` (`config.inverse_frequencies()`) | Short call syntax | Mixes configuration and calculation; the architecture puts frequencies in `frequencies.rs` |
| Free functions with `(d, b)` arguments | No struct | Every function must validate again |
| `[f64; N]` with a const generic `N` | No heap, size checked at compile time | `d` is a run-time configuration value; more complex types |
| `Box<[f64]>` instead of `Vec<f64>` | 16-byte header, cannot grow by mistake | Less familiar; no practical gain here |
| `thiserror` crate | Less boilerplate for `Display` | An external dependency for a few lines of code |

## 8. Performance considerations

- `inverse_frequencies` does `d/2` calls of `powf`. A `powf` call costs
  about 20 to 100 processor cycles. For `d = 128`, this is about 64 calls:
  less than a few microseconds.
- The model calculates the list **one time** and reuses it. Thus this cost
  is negligible compared with attention.
- Memory: `d/2 · 8` bytes. 512 bytes for `d = 128`. This fits easily in the
  L1 cache of the processor.
- Possible optimization (not needed): calculate `ln b` one time and use
  `exp`. Not worth the loss of clarity.

## 9. Practical example

```rust
use rust_rope::rope::config::RopeConfig;
use rust_rope::rope::frequencies::inverse_frequencies;
use std::f64::consts::TAU; // TAU = 2π

let config = RopeConfig::new(8, 10_000.0)?;
let theta = inverse_frequencies(&config);
for (j, &t) in theta.iter().enumerate() {
    let exponent = (2 * j) as f64 / config.dimension() as f64;
    println!("j = {j}: 2j/d = {exponent:<4}  θ = {t:<22?}  wavelength 2π/θ = {:.2}", TAU / t);
}
for w in theta.windows(2) {
    println!("θ(j+1)/θ(j) = {:?}", w[1] / w[0]);
}
```

Expected output (from `cargo run --example rope_frequencies`):

```text
j = 0: 2j/d = 0     θ = 1.0                     wavelength 2π/θ = 6.28
j = 1: 2j/d = 0.25  θ = 0.1                     wavelength 2π/θ = 62.83
j = 2: 2j/d = 0.5   θ = 0.01                    wavelength 2π/θ = 628.32
j = 3: 2j/d = 0.75  θ = 0.001                   wavelength 2π/θ = 6283.19
θ(j+1)/θ(j) = 0.1
θ(j+1)/θ(j) = 0.09999999999999999
θ(j+1)/θ(j) = 0.1
```

An invalid configuration is an error, not a panic:

```rust
assert!(RopeConfig::new(7, 10_000.0).is_err()); // odd dimension
assert!(RopeConfig::new(8, 1.0).is_err());      // base not > 1
```

## 10. What we learned

**Mathematics:**

- `θⱼ = b^(−2j/d)` gives each pair its own rotation speed, from `1` down to
  about `1/b`.
- One speed is not sufficient: fast speeds wrap around, slow speeds cannot
  separate neighbours. Many speeds work together, like the hands of a clock.
- `2j` is the index of the first dimension of pair `j`. `2j/d = j/(d/2)` is
  the fraction of the pairs before pair `j`.
- The division by `d` keeps the exponent in `[0, 1)`, so the speed range
  does not depend on `d`.
- For `b > 1`, `b^x` increases with `x`, so `θⱼ` decreases with `j`.
- The frequencies form a geometric sequence with the ratio `b^(−2/d)`. The
  wavelength `2π/θⱼ` goes from `6.28` to almost `2π·b` positions.
- For `d = 8`, `b = 10000`: `θ = [1, 0.1, 0.01, 0.001]`.

**Rust:**

- An `enum` with data is a good error type. `match` must handle every
  variant.
- `Display` + `std::error::Error` make an error usable with `?` and
  `Box<dyn Error>`.
- `Result<(), E>` for checks. `?` returns early on the first error.
- Private fields + a checking constructor: "validate one time, then trust
  the type".
- Every comparison with NaN is `false`. Write the condition for the valid
  case.
- `Vec<f64>` = a 24-byte header on the stack + a heap block. `collect` from
  an exact-size iterator allocates one time.
- Iterators are lazy. `map` + `collect` compiles to a plain loop.
- `as f64` before a division, to avoid integer division.


---

# Entry 8 — Position indexing (Stage 3, Component 2)

**Date:** 2026-11-01
**Stage:** 3 (RoPE fundamentals)
**Files:** `src/rope/position.rs`, `src/utils/validation.rs`
(`validate_position`, `MAX_POSITION`), `tests/rope_tests.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `Position` (validated newtype around `usize`), `Position::ZERO` | ✅ Implemented |
| `validate_position`, `MAX_POSITION = 2^53` | ✅ Implemented |
| `PositionRange` (configurable start and length, lazy iterator) | ✅ Implemented |
| `PositionRange::index_of` (position → row index) | ✅ Implemented |
| Non-consecutive positions (for example, packed sequences) | ❌ Not implemented. Not needed in Stage 3. |
| Position scaling (interpolation) | ⏳ Planned (Stage 5) |

---

## 1. What did we build?

We built two types for token positions:

- `Position` is one token position `m`. It is a `usize` in a wrapper that
  guarantees `m ≤ 2^53`.
- `PositionRange` is a sequence of consecutive positions
  `start, start + 1, ..., start + L − 1`. The start and the length `L` are
  configurable. The range produces its positions one at a time, without a
  heap allocation.

## 2. Why did we build it?

The RoPE angle is `φ(m, j) = m · θⱼ`. Entry 7 gave us `θⱼ`. This entry
gives us `m`. Entry 9 multiplies them.

A configurable start is necessary in real models. A language model
generates text one token at a time. Example: the prompt has 1000 tokens
(positions 0 to 999). The next generated token has position 1000, not 0.
The model calls `PositionRange::new(1000, 1)` for it. If it used position 0
again, the new token would get the rotation of the first token, and the
relative distances would be wrong.

## 3. Mathematical foundation

### 3.1 Token position versus dimension-pair index

RoPE uses two integer indices. They have completely different roles.

| | Token position `m` | Dimension-pair index `j` |
|---|---|---|
| What it counts | tokens in the sequence | pairs inside one vector |
| Range | `0, 1, 2, ...` (only limited by the context length) | `0, 1, ..., d/2 − 1` (fixed by `d`) |
| What it selects | **how far** a pair rotates (the angle) | **how fast** a pair rotates (the frequency `θⱼ`) |
| Changes when | we move to another token | we move to another pair of the same token |
| Rust type | `Position` | `usize` |

**Example.** The sentence "the cat sat" has three tokens. With `d = 8`, each
token has a vector with 8 numbers, thus 4 pairs:

```text
            j = 0      j = 1      j = 2      j = 3
            (dims 0,1) (dims 2,3) (dims 4,5) (dims 6,7)
m = 0 "the"   0·θ₀       0·θ₁       0·θ₂       0·θ₃
m = 1 "cat"   1·θ₀       1·θ₁       1·θ₂       1·θ₃
m = 2 "sat"   2·θ₀       2·θ₁       2·θ₂       2·θ₃
```

A row is one token (one value of `m`). A column is one pair (one value of
`j`). Entry 9 calculates this table.

### 3.2 Positions are integers

A token position is a count: "the first token", "the second token", and so
on. Thus `m` is a non-negative integer: `m ∈ {0, 1, 2, ...}`. The first
token has `m = 0`, because Rust (and most Transformer code) counts from 0.

The angle formula `m · θⱼ` needs `m` as a real number. Thus we convert `m`
to `f64`. Section 3.3 shows that this conversion has a limit.

### 3.3 Why the limit is `2^53`

An `f64` value is `± 1.f × 2^e`. The fraction `f` has 52 stored bits. With
the leading `1`, the **significand** has 53 bits.

- An integer `n` with `n ≤ 2^53` needs at most 53 significant bits. (`2^53`
  itself is `1 × 2^53`: one significant bit.) Thus `f64` stores it
  **exactly**.
- `2^53 + 1` needs 54 significant bits. `f64` cannot store it. The
  conversion rounds it to `2^53` (round to the nearest even value).

The test `position_limit_is_two_to_the_53` checks both facts:

```rust
assert_eq!(max as f64, (max + 1) as f64);   // 2^53 and 2^53 + 1: the same f64
assert_ne!((max - 1) as f64, max as f64);   // 2^53 − 1 and 2^53: different
```

Above the limit, two different positions would get the **same angle**. The
model could not separate them. Thus `Position::new` rejects `m > 2^53`.

**In practice** the limit is very far away: `2^53 ≈ 9 × 10^15`. Current
models use up to about `10^6` or `10^7` positions. But Entry 9 shows that the
**angle** loses precision much earlier, because the absolute error of
`m · θ` grows with `m`.

### 3.4 A range of consecutive positions

`PositionRange::new(start, L)` describes the positions:

```text
start, start + 1, start + 2, ..., start + L − 1
```

- `L = 0` gives an empty sequence. This is valid.
- The last position `start + L − 1` must be at most `2^53`.
- The **row index** of position `m` in the range is `r = m − start`. It is
  valid when `0 ≤ r < L`. `index_of` calculates it.

Example: `start = 5`, `L = 3` gives positions `5, 6, 7`. Position 6 has row
index `6 − 5 = 1`. Position 8 is not in the range.

### 3.5 Preview: why the start offset does not change attention

Stage 2 proved: `R(mθ)ᵀ R(nθ) = R((n − m)θ)`. The attention score depends
only on `n − m`. If we add the same offset `s` to all positions, then
`(n + s) − (m + s) = n − m`. The scores do not change. Thus the offset is
not an error. It only must be **the same** for all tokens of one sequence.
The test `start_offset_table_equals_the_matching_rows_of_a_table_from_zero`
(Entry 9) checks that a range with an offset gives the same angles as the
matching rows of a range from 0.

## 4. Rust concepts learned

### 4.1 The newtype pattern

```rust
pub struct Position(usize);
```

This is a **tuple struct** with one field. Its field has no name; we access
it as `self.0`. A struct with one field is called a **newtype**.

Why? `Position` and the pair index `j` are both integers. With plain `usize`
values, this mistake compiles:

```rust
fn angle(m: usize, j: usize) -> f64 { /* ... */ }
angle(j, m); // wrong order: compiles, gives a wrong result
```

With the newtype, the compiler rejects it:

```rust
fn angle(m: Position, theta: f64) -> f64 { /* ... */ }
angle(j, theta); // error[E0308]: mismatched types: expected `Position`, found `usize`
```

**Cost:** zero. `Position` has the same size as `usize` (8 bytes, measured)
and the same layout. The wrapper exists only at compile time.

The field is private. Thus code outside `position.rs` cannot write
`Position(10_000_000_000_000_000)`. It must call `Position::new`, which
checks the limit.

### 4.2 `derive` for ordering

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
```

- `Copy`: a `Position` is copied, not moved, when you assign it. A copy of
  8 bytes is as cheap as a move.
- `Eq`: equality is total. This is correct for integers. (`f64` cannot
  implement `Eq`, because `NaN != NaN`.)
- `PartialOrd`, `Ord`: positions have an order, so `a < b` works.

### 4.3 Checked arithmetic

```rust
let last = start
    .checked_add(length - 1)
    .ok_or(RopeError::SequenceTooLong { start, length })?;
```

`a + b` on integers can **overflow** (the result does not fit in the type).
In a debug build, Rust panics on overflow. In a release build, the result
wraps around (for example, `usize::MAX + 1 = 0`). Both are wrong for input
from a caller.

`checked_add` returns `Option<usize>`: `Some(sum)` or `None` on overflow.
`ok_or(error)` converts `Option` into `Result`: `Some(v)` becomes `Ok(v)`,
`None` becomes `Err(error)`. Then `?` returns the error.

Small example:

```rust
assert_eq!(250u8.checked_add(5), Some(255));
assert_eq!(250u8.checked_add(6), None);   // 256 does not fit in a u8
```

`index_of` uses `checked_sub` in the same way. `position − start` is
`None` if `position < start`. The `?` operator also works on `Option` in a
function that returns `Option`: it returns `None` early.

### 4.4 `impl Iterator` in the return type

```rust
pub fn iter(self) -> impl Iterator<Item = Position> {
    (self.start..self.start + self.length).map(Position)
}
```

The exact type of `(a..b).map(Position)` is
`Map<Range<usize>, fn(usize) -> Position>`. This is long, and it is an
implementation detail. `impl Iterator<Item = Position>` means: "this
function returns **some** type that implements `Iterator` with items of
type `Position`".

- The caller can use all iterator methods (`for`, `map`, `count`,
  `collect`, ...).
- The compiler still knows the exact type. It generates direct code. There
  is no heap allocation and no dynamic dispatch (no `Box<dyn Iterator>`).
- We can change the implementation later without a change to the public
  type.

### 4.5 A tuple-struct constructor is a function

`map(Position)` passes the **constructor** `Position` as a function. For a
tuple struct, `Position` can be used as a function `fn(usize) -> Position`.
It is the same as `map(|v| Position(v))`, but shorter. (This works only
inside `position.rs`, because the field is private.)

The test code uses the same idea with a method: `map(Position::value)` is
`map(|m| m.value())`.

### 4.6 `self` by value on a `Copy` type

`PositionRange` is `Copy` (16 bytes). Its methods take `self`, not `&self`.
When you call `range.iter()`, Rust **copies** the range into the method.
The original stays usable:

```rust
let range = PositionRange::new(10, 5)?;
assert_eq!(range.iter().count(), 5);
assert_eq!(range.iter().count(), 5); // still valid: `range` was copied
```

The returned iterator owns its own copy of `start` and `length`. It does
not borrow `range`. Thus it has no lifetime connection to `range`.

### 4.7 Lazy generation and `collect`

`iter()` does not create the positions. It creates a small iterator state
(the current value and the end). Each call of `next()` calculates one
position. If you need a list, you ask for it explicitly:

```rust
let values: Vec<usize> = PositionRange::new(5, 3)?.iter().map(|m| m.value()).collect();
assert_eq!(values, vec![5, 6, 7]);
```

### 4.8 Conditional compilation in tests

```rust
#[cfg(target_pointer_width = "64")]
{ let max = MAX_POSITION as usize; /* ... */ }
```

`2^53` does not fit in a 32-bit `usize`. The block compiles only on 64-bit
targets. On a 32-bit target, every `usize` is far below `2^53`, so the
limit can never be reached there.

`MAX_POSITION` itself is a `u64`, not a `usize`, for the same reason. The
check converts the position: `position as u64 > MAX_POSITION`. The
conversion from `usize` to `u64` never loses information.

## 5. Memory and execution

| Value | Size (measured) | Location |
|---|---|---|
| `Position` | 8 B | stack / register |
| `PositionRange` | 16 B | stack / registers |
| the iterator from `iter()` | about 16 B | stack |

**No heap allocation.** A range of one million positions uses 16 bytes. A
`Vec<Position>` of one million positions would use 8 MB on the heap.

**Execution of `PositionRange::new(start, L)`:**

1. Check `start ≤ 2^53`.
2. If `L > 0`: calculate `last = start + L − 1` with `checked_add`. Check
   `last ≤ 2^53`.
3. Return the 16-byte struct by value.

**Execution of `for m in range.iter()`:** the loop keeps one counter. Each
step increments it and wraps it in `Position`. After inlining, this is the
same machine code as `for v in start..start + L`.

**Ownership:** all types are `Copy`. Nothing is borrowed, nothing is moved
in a way that invalidates the original.

## 6. Engineering decisions

1. **A newtype for `Position`, but not for the pair index.** One newtype is
   sufficient to make the two indices incompatible. A second newtype
   (`PairIndex`) would add conversions everywhere with no extra safety for
   the current API.
2. **`usize` inside `Position`.** Positions are used as counts and as
   offsets for row indices. `usize` is the Rust type for indices.
3. **The limit is `2^53`.** It is the exact boundary for a lossless
   conversion to `f64`. It is not a model limit (a model has its own
   maximum context length; that belongs to a later configuration).
4. **An empty range is valid.** A sequence with zero tokens is a correct
   edge case (for example, an empty batch). Rejecting it would force callers
   to add special cases.
5. **A lazy range, not a `Vec<Position>`.** Positions are consecutive, so
   two numbers describe all of them.
6. **`index_of` returns `Option`.** "The position is not in this range" is a
   normal answer, not an error.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| Plain `usize` for positions | Simplest | The compiler cannot detect swapped indices; no limit check |
| `Position(u64)` | Same size on all targets | Conversions to `usize` for indexing |
| `f64` positions | No conversion | Non-integer positions are possible by mistake (Stage 5 interpolation will need scaled positions deliberately) |
| `Vec<Position>` from a function | Supports arbitrary positions | Allocates `8·L` bytes; consecutive positions do not need it |
| Implement `Iterator` for a custom struct | Full control, nameable type | More code than `impl Iterator` |
| `std::ops::Range<Position>` | Standard type | `Range<T>` iterates only when `T` implements the unstable `Step` trait |

## 8. Performance considerations

- `Position::new`: one comparison. `PositionRange::new`: two comparisons
  and one checked addition. O(1).
- `iter()`: O(1) to create, O(1) per step, no allocation.
- `index_of`: one subtraction and one comparison. O(1).
- The newtype has no run-time cost.

## 9. Practical example

```rust
use rust_rope::rope::position::{Position, PositionRange};

let m = Position::new(3)?;
println!("{} {}", m.value(), m.as_f64());                 // 3 3

let prompt = PositionRange::from_zero(4)?;
let next = PositionRange::new(prompt.len(), 1)?;           // continue after the prompt
let a: Vec<usize> = prompt.iter().map(Position::value).collect();
let b: Vec<usize> = next.iter().map(Position::value).collect();
println!("{a:?} {b:?}");                                   // [0, 1, 2, 3] [4]

println!("{:?}", PositionRange::new(5, 3)?.index_of(Position::new(6)?)); // Some(1)
println!("{}", Position::new(usize::MAX).unwrap_err());
```

Expected output (on a 64-bit computer):

```text
3 3
[0, 1, 2, 3] [4]
Some(1)
the position 18446744073709551615 is larger than the maximum position 9007199254740992 (2^53)
```

## 10. What we learned

**Mathematics:**

- The token position `m` selects **how far** a pair rotates. The pair index
  `j` selects **how fast** it rotates. They are different indices.
- Positions are non-negative integers. The angle needs them as `f64`.
- `f64` stores every integer up to `2^53` exactly. Above `2^53`, neighbouring
  integers can convert to the same `f64`.
- A constant offset on all positions does not change the relative distances,
  so it does not change the attention scores.

**Rust:**

- A newtype (`struct Position(usize)`) makes two integer roles incompatible,
  at zero run-time cost.
- A private field + a checking constructor guarantees an invariant.
- `checked_add` / `checked_sub` return `Option` instead of overflowing.
  `ok_or` converts `Option` to `Result`. `?` works on both.
- `impl Iterator<Item = T>` returns an iterator without naming its type and
  without allocation.
- A tuple-struct constructor can be used as a function (`map(Position)`).
- Methods with `self` on a `Copy` type copy the value; the original stays
  usable.
- `#[cfg(...)]` compiles code only for some targets.


---

# Entry 9 — Positional angles and the angle table (Stage 3, Component 3)

**Date:** 2026-11-01
**Stage:** 3 (RoPE fundamentals)
**Files:** `src/rope/frequencies.rs` (the angle part),
`src/utils/validation.rs` (`validate_inverse_frequency`,
`validate_inverse_frequencies`), `tests/rope_tests.rs`,
`tests/numerical_tests.rs`, `examples/rope_frequencies.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `angle(m, θ)` = `m · θ`, with validation of `θ` | ✅ Implemented |
| `AngleTable` (row-major `L × P` table of `φ(m, j)`) | ✅ Implemented |
| Lookup: `row(m)`, `get(m, j)`, `as_slice()` | ✅ Implemented |
| Test: the angles define rotation matrices with the RoPE properties | ✅ Implemented |
| `cos φ`, `sin φ` cache (the "RoPE cache") | ⏳ Planned (Stage 4) |
| Rotation of embedding vectors | ⏳ Planned (Stage 4) |

---

## 1. What did we build?

We built the calculation of the **positional angles**:

```text
φ(m, j) = m · θⱼ
```

- `angle(m, θ)` calculates one angle.
- `AngleTable::new(positions, &θ)` calculates the angles for a range of
  positions and all pairs. It stores them in one flat `Vec<f64>`.

## 2. Why did we build it?

The angle is the last value before the rotation. Stage 4 will calculate
`cos φ(m, j)` and `sin φ(m, j)` and rotate pair `j` of the token at position
`m`. Each angle completely defines one 2 × 2 rotation matrix (section 3.6).
Thus the angle table **is** the complete position information of RoPE.

## 3. Mathematical foundation

### 3.1 The equation and its variables

```text
φ(m, j) = m · θⱼ
```

| Symbol | Meaning | Unit |
|---|---|---|
| `m` | token position (Entry 8) | positions |
| `j` | dimension-pair index (Entry 7) | — |
| `θⱼ` | inverse frequency of pair `j` (Entry 7) | radians per position |
| `φ(m, j)` | rotation angle of pair `j` of the token at position `m` | radians |

The units explain the formula: (radians per position) × (positions) =
radians. `θⱼ` is a **speed**, `m` is a **distance**, `φ` is the **total
angle**.

### 3.2 Derivation: constant speed

RoPE wants each step of one position to add the same rotation `θⱼ` to pair
`j`. Then:

```text
φ(0, j) = 0                      (start: no rotation)
φ(1, j) = φ(0, j) + θⱼ = θⱼ
φ(2, j) = φ(1, j) + θⱼ = 2θⱼ
...
φ(m, j) = m · θⱼ                 (m steps of θⱼ)
```

A constant step gives a **linear** function of `m`. The test
`angles_increase_linearly_with_position` checks
`φ(m + 1, j) − φ(m, j) = θⱼ` for `m = 0..63` and `d = 8, 64, 128`.

Why must the step be constant? Only then does the rotation between two
positions depend only on their **distance**:
`φ(n, j) − φ(m, j) = (n − m)·θⱼ`. This is the relative-position property
(section 3.7).

### 3.3 The angle table for `d = 8`, `b = 10000`, `m = 0..4`

The frequencies (Entry 7) are `θ = [1, 0.1, 0.01, 0.001]`. Each cell is
`m · θⱼ`:

| `m` | `j = 0` (`θ₀ = 1`) | `j = 1` (`θ₁ = 0.1`) | `j = 2` (`θ₂ = 0.01`) | `j = 3` (`θ₃ = 0.001`) |
|---|---|---|---|---|
| 0 | 0·1 = 0 | 0·0.1 = 0 | 0·0.01 = 0 | 0·0.001 = 0 |
| 1 | 1·1 = 1 | 1·0.1 = 0.1 | 1·0.01 = 0.01 | 1·0.001 = 0.001 |
| 2 | 2·1 = 2 | 2·0.1 = 0.2 | 2·0.01 = 0.02 | 2·0.001 = 0.002 |
| 3 | 3·1 = 3 | 3·0.1 = 0.3 | 3·0.01 = 0.03 | 3·0.001 = 0.003 |
| 4 | 4·1 = 4 | 4·0.1 = 0.4 | 4·0.01 = 0.04 | 4·0.001 = 0.004 |

All values are in radians. For a feeling of size: `1 rad ≈ 57.3°`. Thus
pair 0 at `m = 4` has rotated `4 rad ≈ 229°`, and pair 3 at `m = 4` has
rotated only `0.004 rad ≈ 0.23°`.

**Read the table by column (one pair):** the values grow linearly, with the
step `θⱼ`. **Read the table by row (one token):** the values fall by the
factor 10 from pair to pair (the geometric sequence of Entry 7).

### 3.4 Why position zero gives zero rotation

`φ(0, j) = 0 · θⱼ = 0` for every pair. The rotation matrix for the angle 0 is
the identity:

```text
R(0) = [ cos 0   −sin 0 ] = [ 1  0 ]
       [ sin 0    cos 0 ]   [ 0  1 ]
```

Thus RoPE does **not change** the vectors of the first token. This is
correct: position 0 is the reference point. RoPE measures the other
positions relative to it. In `f64`, `0 · θ` is exactly `0` for every finite
`θ`, so the test `position_zero_gives_zero_angles` uses exact equality.

(Detail: the program prints `R(0, 0) = [[1, -0], [0, 1]]`. The element
`−sin 0` is `−0.0`, the negative zero of IEEE 754. It is equal to `0.0`:
`-0.0 == 0.0` is `true`.)

### 3.5 Why the same position gives different angles for different pairs

At a fixed position `m ≠ 0`, the angles are `m·θ₀, m·θ₁, ..., m·θ_(P−1)`.
Entry 7 proved that `θ₀ > θ₁ > ... > θ_(P−1) > 0` for `b > 1`. Multiply by
`m > 0`: the order does not change. Thus:

```text
φ(m, 0) > φ(m, 1) > ... > φ(m, P − 1) > 0      for every m ≥ 1
```

All angles of one token are **different**. Each pair carries a different
"view" of the same position: pair 0 changes quickly, pair 3 changes slowly.
This is the clock idea of Entry 7, section 3.3. (Test:
`different_pairs_give_different_angles_at_nonzero_positions`.)

### 3.6 The connection to the rotation matrix

Entry 5 defined the rotation matrix for an angle `α`:

```text
R(α) = [ cos α   −sin α ]
       [ sin α    cos α ]
```

RoPE uses the angle `α = φ(m, j) = m · θⱼ`. We write `R(m, j)`:

```text
R(m, j) = [ cos(m·θⱼ)   −sin(m·θⱼ) ]
          [ sin(m·θⱼ)    cos(m·θⱼ) ]
```

Three inputs determine the matrix, through two steps:

```text
(d, b, j) ──Entry 7──► θⱼ ──┐
                            ├──Entry 9──► φ = m·θⱼ ──Entry 5──► R(m, j)
       m ──Entry 8──────────┘
```

- **The frequency step** (`d`, `b`, `j` → `θⱼ`) decides the **speed** of the
  rotation of pair `j`.
- **The position step** (`m` → `m·θⱼ`) decides **how far** the pair has
  rotated at this token.

**Numerical example: `m = 1`, `d = 8`** (from `cargo run --example
rope_frequencies`):

```text
j = 0: φ = 1      R = [[0.540302, -0.841471], [0.841471, 0.540302]]
j = 1: φ = 0.1    R = [[0.995004, -0.099833], [0.099833, 0.995004]]
j = 2: φ = 0.01   R = [[0.999950, -0.010000], [0.010000, 0.999950]]
j = 3: φ = 0.001  R = [[1.000000, -0.001000], [0.001000, 1.000000]]
```

For small angles, `cos φ ≈ 1 − φ²/2` and `sin φ ≈ φ` (the first terms of the
power series, Entry 6). Thus the matrices for the slow pairs are close to
the identity. Check `j = 1`: `1 − 0.01/2 = 0.995` and `sin 0.1 ≈ 0.1`.

**The full vector (planned for Stage 4).** A vector with `d = 8` has 4
pairs. RoPE rotates each pair with its own matrix. As one `8 × 8` matrix,
this is **block-diagonal** (all other elements are 0):

```text
R_m = diag( R(m, 0), R(m, 1), R(m, 2), R(m, 3) )
```

This stage does not implement it. It only calculates the angles.

### 3.7 Properties that follow from the linear angle

These properties follow from Entry 5 (`R(α)R(β) = R(α + β)`,
`R(α)ᵀ = R(−α)`) and `φ(m, j) = m·θⱼ`:

```text
R(0, j)              = I                        (position 0: no rotation)
R(m, j)              = R(1, j)^m                (m steps of the same rotation)
R(m, j)ᵀ · R(n, j)   = R(n − m, j)              (only the distance counts)
```

Proof of the third line: `R(mθ)ᵀ R(nθ) = R(−mθ) R(nθ) = R((n − m)θ)`.

The test `angle_table_defines_rope_rotation_matrices` checks all three
lines with matrices built from the `AngleTable`. The third line is the
reason for the RoPE design: the attention score
`(R(m, j)·q)ᵀ (R(n, j)·k) = qᵀ R(n − m, j) k` depends only on `n − m`.

### 3.8 Numerical precision

**Rounding of the product.** `m · θ` is one multiplication. IEEE 754
rounds it correctly: the relative error is at most `ε/2 ≈ 1.1e−16`. But
the input `θ` is already rounded. Example from the program:

```text
full f64 value of φ(3, 1) = 3 · 0.1: 0.30000000000000004
```

`0.1` is stored as `0.1000000000000000055511151231257827...`. Three times
this value is slightly more than `0.3`, and the nearest `f64` is
`0.30000000000000004`. This is one `f64` step (`5.6e−17`) above the `f64`
value that `0.3` gives. The tests use `approx_eq` with `1e−12`, which
accepts it.

**The absolute error grows with `m`.** The relative error of `m · θ` stays
about `1e−16`. Thus the **absolute** error is about `|m·θ| · 1e−16`. For
`m = 10^6` and `θ = 1`, this is about `1e−10` radian. `sin` and `cos` then
reduce the angle modulo `2π`, but they cannot remove the error that is
already in the input (Entry 4, section 3.8).

**`f32` is much worse.** We measured `θ₁ = 10000^(−2/64)` and `m · θ₁` in
both types:

| `m` | `f64` angle | `f32` angle | absolute difference |
|---|---|---|---|
| 1 000 | 749.894209332 | 749.894226074 | 1.7e−5 |
| 100 000 | 74 989.420933246 | 74 989.421875000 | 9.4e−4 |
| 1 000 000 | 749 894.209332456 | 749 894.187500000 | 2.2e−2 |

At one million positions, the `f32` angle is wrong by `0.022` radian
(about `1.25°`). This is one reason that long-context implementations
calculate the angles in `f32` with care, or in higher precision. Stage 6
studies this.

**We do not reduce the angle modulo `2π`.** `φ mod 2π` would give a value
in `[0, 2π)`. But the reduction itself needs `2π` as an `f64`, which is
rounded. `sin` and `cos` in the standard library do a more accurate
reduction internally. Thus we keep the raw product.

### 3.9 Where a Transformer uses this

The angles depend only on the positions and the frequencies, not on the
input text. Thus implementations calculate the table **one time** for the
maximum context length and reuse it for every layer and every head. Usually
they store `cos φ` and `sin φ` directly (the "RoPE cache"). Stage 4 will
build this cache from the `AngleTable`.

## 4. Rust concepts learned

### 4.1 A flat `Vec` with row-major indexing

```rust
pub struct AngleTable {
    positions: PositionRange,
    num_pairs: usize,
    angles: Vec<f64>,       // angles[r * num_pairs + j]
}
```

A 2-D table can be stored as one 1-D `Vec`. **Row-major** order stores row
0, then row 1, and so on. The element at row `r` and column `j` has the
index `r · P + j`.

```text
P = 2, rows m = 2, 3, 4, θ = [1, 0.5]:

index:   0     1     2     3     4     5
value: [2.0,  1.0,  3.0,  1.5,  4.0,  2.0]
        └ m=2 ┘     └ m=3 ┘     └ m=4 ┘
```

(Test: `table_layout_is_row_major`.)

Why not `Vec<Vec<f64>>`? A `Vec<Vec<f64>>` has one heap allocation for
**each row**, plus one for the outer `Vec`. The rows can be anywhere in
memory. One flat `Vec` is one allocation, and the processor reads it
sequentially. This is faster (better cache use) and simpler to free.

Why row-major and not column-major? Stage 4 rotates **one token** at a time.
It needs all pairs of one position: one row. In row-major order, a row is a
contiguous slice.

### 4.2 `Vec::with_capacity` and `push`

```rust
let mut angles = Vec::with_capacity(num_elements);
for position in positions.iter() {
    for &theta in inverse_frequencies {
        angles.push(angle_unchecked(position, theta));
    }
}
```

`Vec::with_capacity(n)` allocates space for `n` elements but sets the
length to 0. `push` writes one element and increments the length. Because
the capacity is already correct, no `push` reallocates. The table has
**exactly one** heap allocation.

Small example:

```rust
let mut v: Vec<f64> = Vec::with_capacity(3);
assert_eq!((v.len(), v.capacity() >= 3), (0, true));
v.push(1.0); v.push(2.0); v.push(3.0);   // no reallocation
assert_eq!(v.len(), 3);
```

### 4.3 Borrowed input: `&[f64]`

```rust
pub fn new(positions: PositionRange, inverse_frequencies: &[f64]) -> Result<Self, RopeError>
```

`&[f64]` is a **slice**: a borrowed view of a sequence of `f64` values. It
is a pointer and a length (16 bytes). The caller can pass a `&Vec<f64>`, an
array, or a part of a `Vec`. Rust converts `&Vec<f64>` to `&[f64]`
automatically (deref coercion).

The function **borrows** the frequencies. It reads them, and the caller
keeps ownership. The caller can use the same frequencies for many tables.
Without the borrow, the function would take ownership, and the caller would
have to clone the `Vec` for each table.

### 4.4 `for &theta in slice`

`for x in slice` gives references `&f64`. The pattern `&theta` takes the
value out of the reference. `f64` is `Copy`, so this copies 8 bytes. Then
`theta` is a plain `f64`, and `m * theta` needs no `*theta`.

### 4.5 `enumerate()`

`validate_inverse_frequencies` uses `enumerate()` to know the index of an
invalid value:

```rust
for (j, &value) in values.iter().enumerate() {
    // j = 0, 1, 2, ...; value = values[j]
}
```

`enumerate()` wraps an iterator. Each item becomes a pair `(index, item)`.
The index starts at 0 and has the type `usize`. It is cheaper and safer than
a manual counter: you cannot forget to increment it.

Small example:

```rust
let names = ["a", "b", "c"];
let v: Vec<(usize, &str)> = names.iter().copied().enumerate().collect();
assert_eq!(v, [(0, "a"), (1, "b"), (2, "c")]);
```

### 4.6 Returning a slice: `row(&self) -> Option<&[f64]>`

```rust
pub fn row(&self, position: Position) -> Option<&[f64]> {
    let r = self.positions.index_of(position)?;
    let start = r * self.num_pairs;
    Some(&self.angles[start..start + self.num_pairs])
}
```

- `&self.angles[a..b]` is a slice of the `Vec`: elements `a` to `b − 1`. It
  points into the existing heap block. **It copies nothing.**
- The returned slice borrows the table. The full signature is
  `fn row<'a>(&'a self, position: Position) -> Option<&'a [f64]>`. The
  compiler adds the **lifetime** `'a` automatically (lifetime elision): an
  output reference borrows from `self`. While the slice exists, nobody can
  change or drop the table. The borrow checker enforces this at compile
  time.
- `?` on the `Option` from `index_of` returns `None` early when the position
  is not in the table.

### 4.7 `slice.get(j).copied()`

```rust
self.row(position)?.get(pair_index).copied()
```

`slice[j]` panics if `j` is out of range. `slice.get(j)` returns
`Option<&f64>` instead: `None` if `j` is out of range. `.copied()` turns
`Option<&f64>` into `Option<f64>`. Thus `get` never panics.

### 4.8 `checked_mul` and `debug_assert_eq!`

`L · P` can overflow for very large inputs. `checked_mul` detects this
before the allocation and returns `RopeError::TableTooLarge`. The test
`table_too_large_is_an_error` uses `L = 2^53` and `P = 2^11`:
`2^64` does not fit in a 64-bit `usize`.

`debug_assert_eq!(angles.len(), num_elements)` checks an **internal**
invariant. It runs only in debug builds (and tests). If it fails, the
library has a bug. This is the correct use of a panic (CLAUDE.md,
section 6).

## 5. Memory and execution

| Value | Size | Location |
|---|---|---|
| `AngleTable` struct | 48 B (measured): range 16 + `num_pairs` 8 + `Vec` header 24 | stack |
| angle data | `L · P · 8` B | heap, one block |
| the input frequencies | not copied | borrowed from the caller |

Examples of the heap size:

| `L` (positions) | `d` | `P` | heap |
|---|---|---|---|
| 5 | 8 | 4 | 160 B |
| 4096 | 128 | 64 | 2 MiB |
| 131 072 | 128 | 64 | 64 MiB |

**Execution of `AngleTable::new(range, &theta)`:**

1. Check the frequencies (one pass, `P` comparisons).
2. Calculate `L · P` with `checked_mul`.
3. Allocate one block of `L · P · 8` bytes.
4. Two nested loops: for each position `m`, for each `θⱼ`, write `m·θⱼ`.
   The writes are sequential in memory.
5. Move the `Vec` into the struct, and move the struct to the caller (the
   `Ok(...)` result). The data does not move; only the 48-byte struct does.

**How we avoid unnecessary allocations in Stage 3:**

- The positions are never stored as a list (`PositionRange` is lazy).
- The frequencies are borrowed, not cloned.
- The table allocates exactly one block with the final size.
- `row` and `as_slice` return borrowed slices, not copies.

## 6. Engineering decisions

1. **Angles in `frequencies.rs`.** The task allowed `frequencies.rs` or a
   new module. The angle is one multiplication of a frequency by a
   position. A separate module would add a file for about 100 lines, and
   the fixed architecture table in CLAUDE.md has no "angles" module. The
   module comment of `position.rs` from Stage 1 said "convert positions into
   angles". We kept positions in `position.rs` and put the angles next to
   the frequencies, because the angle needs both.
2. **The table takes `&[f64]`, not `&RopeConfig`.** The table works with
   any valid list of frequencies. Thus it does not depend on how the
   frequencies were calculated. (Stage 5 will scale the frequencies; the
   table will accept the scaled list without a change.)
3. **The table validates the frequencies.** A slice is not a validated type
   (unlike `RopeConfig`). Thus `AngleTable::new` and `angle` check it.
4. **The table stores the `PositionRange`.** Then `row(m)` takes a real
   position `m`, not a row index. This avoids the off-by-start mistake when
   `start ≠ 0`.
5. **`Option` for lookups, `Result` for construction.** "Position not in the
   table" is a normal answer. Invalid frequencies are an input error.
6. **No sine or cosine in this stage.** The task restricts Stage 3 to
   angles.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| `Vec<Vec<f64>>` | Natural `table[r][j]` syntax | `L + 1` allocations, scattered memory |
| Column-major layout | One pair over all positions is contiguous | Stage 4 needs one position over all pairs |
| No table: calculate `m·θⱼ` when needed | No memory | Stage 4 would recalculate for each layer and head |
| A cache of `cos φ` and `sin φ` instead of `φ` | Saves the trigonometry in Stage 4 | Hides the angle, which this stage studies; planned for Stage 4 |
| `iter().flat_map(...)` + `collect` | One expression | Harder to read; `flat_map` does not know the exact length, so `collect` can reallocate |
| `f32` angles | Half the memory | Large angle errors at large `m` (section 3.8) |
| The `ndarray` crate | General N-D arrays | External dependency for one 2-D table |

## 8. Performance considerations

- `AngleTable::new`: `L · P` multiplications, `O(L · P)` time, one
  allocation. For `L = 4096`, `d = 128`: 262 144 multiplications, which take
  well below one millisecond.
- Memory: `L · P · 8` bytes. This is the main cost. A model with a context of
  131 072 tokens and `d = 128` needs 64 MiB for the angles in `f64`. A
  `cos`/`sin` cache needs twice that. This is a reason why some
  implementations calculate the values on the fly or in `f32`.
- `row` and `get`: `O(1)`. No copy.
- The inner loop is simple and sequential. The compiler can vectorize it
  (SIMD, Stage 6).

## 9. Practical example

```rust
use rust_rope::math::matrix::Matrix2;
use rust_rope::rope::config::RopeConfig;
use rust_rope::rope::frequencies::{inverse_frequencies, AngleTable};
use rust_rope::rope::position::{Position, PositionRange};

let theta = inverse_frequencies(&RopeConfig::new(8, 10_000.0)?);
let table = AngleTable::new(PositionRange::from_zero(5)?, &theta)?;

for m in table.positions().iter() {
    let row = table.row(m).expect("the table contains its own positions");
    print!("m = {}  ", m.value());
    for phi in row { print!("{phi:<8.3}"); }
    println!();
}
println!("{:?}", table.get(Position::new(3)?, 1));

let phi = table.get(Position::new(1)?, 1).unwrap();
let [[a, b], [c, e]] = Matrix2::rotation(phi).rows();
println!("φ = {phi}  R = [[{a:.6}, {b:.6}], [{c:.6}, {e:.6}]]");

let later = AngleTable::new(PositionRange::new(1000, 3)?, &theta)?;
for m in later.positions().iter() {
    println!("m = {}: φ = {:?}", m.value(), later.get(m, 3).unwrap());
}
```

Expected output (from `cargo run --example rope_frequencies`):

```text
m = 0  0.000   0.000   0.000   0.000
m = 1  1.000   0.100   0.010   0.001
m = 2  2.000   0.200   0.020   0.002
m = 3  3.000   0.300   0.030   0.003
m = 4  4.000   0.400   0.040   0.004
Some(0.30000000000000004)
φ = 0.1  R = [[0.995004, -0.099833], [0.099833, 0.995004]]
m = 1000: φ = 1.0
m = 1001: φ = 1.0010000000000001
m = 1002: φ = 1.002
```

## 10. What we learned

**Mathematics:**

- `φ(m, j) = m · θⱼ`: speed (radians per position) × distance (positions)
  = angle (radians).
- A constant step per position gives a linear angle. Only a linear angle
  makes the rotation between two tokens depend only on their distance.
- Position 0 gives the angle 0 and the identity matrix: the first token is
  not changed.
- At a fixed position `m ≥ 1`, the angles of the pairs are all different
  and decrease with `j`.
- The angle defines the rotation matrix `R(m, j) = R(m·θⱼ)`. The frequency
  step decides the speed; the position step decides how far.
- `R(m, j)ᵀ R(n, j) = R(n − m, j)`: the foundation of relative position.
- The absolute error of `m·θ` grows with `m`. `f32` loses about `0.02`
  radian at one million positions.

**Rust:**

- A 2-D table as one flat row-major `Vec`: one allocation, contiguous rows.
- `Vec::with_capacity` + `push` avoids reallocation.
- `&[f64]` borrows a sequence without copying it.
- `enumerate()` gives `(index, item)` pairs.
- A method can return a slice that borrows `self`. Lifetime elision connects
  the output lifetime to `&self`.
- `get(j).copied()` is a lookup that cannot panic.
- `checked_mul` prevents an overflow in a size calculation.
  `debug_assert!` checks internal invariants.

---

## Roadmap status after Entry 9

Stage 3 is complete (review pending). The roadmap table after Entry 6 shows
the state at that time. This table shows the current state.

| Stage | Status |
|---|---|
| 1 Project foundation | ✅ Done (Entry 1) |
| 2 Mathematical foundations | ✅ Done (Entries 2–6) |
| 3 RoPE fundamentals: frequencies, positions, angles | ✅ Done (Entries 7–9) |
| 4 RoPE implementation (sine/cosine cache, pair rotation, layouts) | ⏳ Planned (next, after review) |
| 5 Advanced RoPE | ⏳ Planned |
| 6 Performance engineering | ⏳ Planned |
| 7 Validation and experimentation | ⏳ Planned |

**Scope note.** The README listed "Applying rotations to embedding
dimensions" under Stage 3. The Stage 3 task excluded it. It moves to
Stage 4.


---

# Entry 10 — The RoPE transformation of an embedding vector (Stage 4)

**Date:** 2026-11-01
**Stage:** 4 (RoPE implementation)
**Files:** `src/rope/rotations.rs`, `src/rope/embeddings.rs`,
`src/rope/config.rs` (review), `src/rope/frequencies.rs`
(`angle_unchecked` is now `pub(crate)`), `src/utils/validation.rs`
(`EmbeddingDimensionMismatch`, `validate_embedding_dimension`),
`examples/basic_rope.rs`, `tests/rope_tests.rs`, `tests/numerical_tests.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| Component A: `rotate_pair(pair, φ)` | ✅ Implemented |
| Component B: `RotaryEmbedding` with `rotate` (new vector) and `rotate_in_place` | ✅ Implemented |
| Component C: configuration review (`d`, `b`; no new options) | ✅ Done |
| Interleaved (adjacent-pair) layout | ✅ Implemented — **the only layout** |
| Split-half layout | ❌ Not implemented. Explained in section 3.8. |
| A precomputed `cos`/`sin` cache | ❌ Not implemented (see the correction note below and section 8) |
| Attention, queries/keys, scaling | ⏳ Later stages |

**Correction note (2026-11-01) to the Stage 3 report and Entry 9.** Entry 9,
section 3.9 says: "Stage 4 will build this cache from the `AngleTable`."
Stage 4 did **not** build a `cos`/`sin` cache. The Stage 4 task asked for
an easy-to-understand implementation without premature optimization. The
cache stays a planned optimization (Stage 6).

---

## 1. What did we build?

We built the complete RoPE transformation of one embedding vector:

```text
RoPE(x, m) = [R(m·θ₀)·(x₀, x₁),  R(m·θ₁)·(x₂, x₃),  ...,  R(m·θ_(d/2−1))·(x_(d−2), x_(d−1))]
```

- `rope::rotations::rotate_pair` rotates one pair by one angle.
- `rope::embeddings::RotaryEmbedding` holds a configuration and its
  inverse frequencies. Its method `rotate` returns a new rotated vector.
  Its method `rotate_in_place` overwrites the input.

## 2. Why did we build it?

This is the core of RoPE. Stages 2 and 3 built the parts: the rotation
matrix, the frequencies, the positions and the angles. This stage connects
them. After this stage, the library can put the position of a token into its
vector.

## 3. Mathematical foundation

### 3.1 The input

An embedding is a vector of `d` real numbers:

```text
x = [x₀, x₁, x₂, x₃, ..., x_(d−1)]
```

`d` is even. The token of this vector has the position `m`.

### 3.2 Step 1: divide the vector into pairs

```text
(x₀, x₁), (x₂, x₃), ..., (x_(d−2), x_(d−1))
```

Pair `j` is `(x₂ⱼ, x₂ⱼ₊₁)`, for `j = 0, 1, ..., d/2 − 1`.

**Why pairs?** A rotation is an operation in a **plane**: a two-dimensional
space. A rotation in two dimensions has exactly one parameter, the angle.
Thus one pair of numbers can carry one angle: one piece of positional
information. With `d/2` independent planes, the vector can carry `d/2`
different angles.

(A rotation in three or more dimensions needs an axis or a plane, and two
rotations in 3-D do not commute. Pairs keep everything simple: each pair is
an independent 2-D rotation, and 2-D rotations commute:
`R(a)R(b) = R(b)R(a) = R(a + b)`.)

### 3.3 Step 2: the angle of each pair

```text
φ(m, j) = m · θⱼ,      θⱼ = b^(−2j/d)
```

**Why a different frequency for each pair?** (Entry 7, section 3.3.) One
speed would wrap around after about 6 positions (fast) or could not
separate neighbours (slow). Different speeds act like the hands of a
clock. Together they identify the position.

**Why does the angle depend on the position?** The position is the
information that RoPE must add. A constant step `θⱼ` per position makes the
angle **linear** in `m` (Entry 9, section 3.2). Only a linear angle makes the
rotation between two positions depend on their distance `n − m`
(section 3.6).

### 3.4 Step 3: rotate each pair

```text
x'₂ⱼ   = x₂ⱼ · cos φ − x₂ⱼ₊₁ · sin φ
x'₂ⱼ₊₁ = x₂ⱼ · sin φ + x₂ⱼ₊₁ · cos φ        with φ = m·θⱼ
```

In matrix form, with the rotation matrix of Entry 5:

```text
[ x'₂ⱼ   ]   [ cos φ  −sin φ ] [ x₂ⱼ   ]
[ x'₂ⱼ₊₁ ] = [ sin φ   cos φ ] [ x₂ⱼ₊₁ ]
```

**How the matrix transforms the pair.** Write the pair in polar form:
`(x₂ⱼ, x₂ⱼ₊₁) = r·(cos α, sin α)`. Entry 5, section 3.6 showed that
`R(φ)` gives `r·(cos(α + φ), sin(α + φ))`. The length `r` stays the same.
The direction angle `α` increases by `φ`.

For the whole vector, the `d/2` matrices form one `d × d`
**block-diagonal** matrix. All elements outside the 2 × 2 blocks are 0:

```text
R_m = [ R(m·θ₀)    0          ...   0                ]
      [ 0          R(m·θ₁)    ...   0                ]
      [ ...                                          ]
      [ 0          0          ...   R(m·θ_(d/2−1))   ]

RoPE(x, m) = R_m · x
```

The implementation never builds this matrix. It would have `d²` elements,
and almost all of them are 0. The loop over pairs does the same work with
`d/2` small rotations.

### 3.5 The complex-number interpretation (Euler's formula)

View pair `j` as one complex number: `zⱼ = x₂ⱼ + i·x₂ⱼ₊₁`. Entry 6 showed
that multiplication by `e^(iφ) = cos φ + i·sin φ` is a rotation by `φ`:

```text
zⱼ · e^(i·m·θⱼ) = (x₂ⱼ + i·x₂ⱼ₊₁)(cos φ + i·sin φ)
                = (x₂ⱼ cos φ − x₂ⱼ₊₁ sin φ) + i·(x₂ⱼ sin φ + x₂ⱼ₊₁ cos φ)
                = x'₂ⱼ + i·x'₂ⱼ₊₁
```

The real part is `x'₂ⱼ`. The imaginary part is `x'₂ⱼ₊₁`. Thus:

> A `d`-dimensional real vector is a `d/2`-dimensional complex vector `z`.
> RoPE multiplies component `j` of `z` by `e^(i·m·θⱼ)`.

The original LLaMA code uses exactly this view. The test
`rope_equals_complex_multiplication_per_pair` checks it bit for bit.

### 3.6 Why RoPE keeps the magnitude of each pair

```text
x'₂ⱼ² + x'₂ⱼ₊₁²
 = (x cos φ − y sin φ)² + (x sin φ + y cos φ)²                     (x = x₂ⱼ, y = x₂ⱼ₊₁)
 = x² cos²φ − 2xy sin φ cos φ + y² sin²φ + x² sin²φ + 2xy sin φ cos φ + y² cos²φ
 = x² (cos²φ + sin²φ) + y² (sin²φ + cos²φ)                          (the cross terms cancel)
 = x² + y²                                                           (sin² + cos² = 1)
```

Each pair keeps its length. The squared length of the whole vector is the sum
of the squared lengths of the pairs, so the whole vector also keeps its
length. In the complex view: `|zⱼ · e^(iφ)| = |zⱼ| · 1 = |zⱼ|`.

### 3.7 Position information without an added vector

The original Transformer **adds** a position vector `p_m` to the embedding:
`x + p_m`. This changes the length and the content of the vector. The model
must learn to separate content from position.

RoPE **multiplies** instead: `R_m · x`. It changes only the **directions**
of the pairs, not their lengths. There is no extra vector and no trainable
parameter. And the dot product of two rotated vectors depends only on the
distance:

```text
(R_m u)ᵀ (R_n v) = uᵀ R_mᵀ R_n v = uᵀ R_(n−m) v
```

because each block satisfies `R(mθ)ᵀ R(nθ) = R((n − m)θ)` (Entry 5). The
test `full_vector_dot_product_depends_only_on_the_distance` checks this for
`d = 16`. (Stage 4 does not implement attention. This test only checks a
property of the rotation.)

### 3.8 Dimension layouts: interleaved and split-half

**Our implementation uses the interleaved (adjacent-pair) layout.**

| Layout | Pair `j` | `d = 4` example | Used by |
|---|---|---|---|
| **Interleaved** (this project) | `(x₂ⱼ, x₂ⱼ₊₁)` | `(x₀, x₁)`, `(x₂, x₃)` | RoFormer paper, original LLaMA (complex view) |
| Split-half | `(xⱼ, x_(j+d/2))` | `(x₀, x₂)`, `(x₁, x₃)` | Hugging Face `rotate_half`, GPT-NeoX |

In split-half form, the rotation is often written as:

```text
x' = x ⊙ [cos φ₀, ..., cos φ_(d/2−1), cos φ₀, ..., cos φ_(d/2−1)]
   + rotate_half(x) ⊙ [sin φ₀, ..., sin φ_(d/2−1), sin φ₀, ..., sin φ_(d/2−1)]
rotate_half([a, b]) = [−b, a]      (a = first half, b = second half)
```

`⊙` is element-by-element multiplication. Both layouts are valid RoPE: each
pair rotates by `m·θⱼ`, so all invariants (length, relative position) hold.
But they give **different numbers** for the same input, because they pair
different components. The two layouts are related by a fixed permutation of
the components. A model trained with one layout must use that layout. When
we compare with an external implementation (Stage 7), we must first convert
the layout.

We did not implement split-half. One layout keeps the code and the tests
simple. Adding it later needs only a different way to select the two
components of pair `j`.

### 3.9 Complete numerical example: `d = 8`, `b = 10000`, `m = 1`

**Input:** `x = [1, 2, 3, 4, 5, 6, 7, 8]`.

**Frequencies** (Entry 7): `θ = [1, 0.1, 0.01, 0.001]`.
**Angles at `m = 1`:** `φ = 1·θ = [1, 0.1, 0.01, 0.001]`.

**Pair 0: `(x₀, x₁) = (1, 2)`, `φ = 1`.**

```text
cos 1 = 0.5403023,   sin 1 = 0.8414710
x'₀ = 1·0.5403023 − 2·0.8414710 = 0.5403023 − 1.6829420 = −1.1426397
x'₁ = 1·0.8414710 + 2·0.5403023 = 0.8414710 + 1.0806046 =  1.9220756
check: (−1.1426397)² + 1.9220756² = 1.3056255 + 3.6943746 = 5.0000001 ≈ 5 = 1² + 2²
```

**Pair 1: `(x₂, x₃) = (3, 4)`, `φ = 0.1`.**

```text
cos 0.1 = 0.9950042,   sin 0.1 = 0.0998334
x'₂ = 3·0.9950042 − 4·0.0998334 = 2.9850125 − 0.3993337 = 2.5856788
x'₃ = 3·0.0998334 + 4·0.9950042 = 0.2995002 + 3.9800167 = 4.2795169
check: 2.5856788² + 4.2795169² = 6.6857349 + 18.3142649 = 24.9999998 ≈ 25 = 3² + 4²
```

**Pair 2: `(x₄, x₅) = (5, 6)`, `φ = 0.01`.**

```text
cos 0.01 = 0.9999500,   sin 0.01 = 0.0099998
x'₄ = 5·0.9999500 − 6·0.0099998 = 4.9997500 − 0.0599990 = 4.9397510
x'₅ = 5·0.0099998 + 6·0.9999500 = 0.0499992 + 5.9997000 = 6.0496992
check: 4.9397510² + 6.0496992² = 24.4011399 + 36.5988604 = 61.0000004 ≈ 61 = 5² + 6²
```

**Pair 3: `(x₆, x₇) = (7, 8)`, `φ = 0.001`.**

```text
cos 0.001 = 0.9999995,   sin 0.001 = 0.0010000 (more exactly 0.00099999983)
x'₆ = 7·0.9999995 − 8·0.0010000 = 6.9999965 − 0.0080000 = 6.9919965
x'₇ = 7·0.0010000 + 8·0.9999995 = 0.0070000 + 7.9999960 = 8.0069960
check: 6.9919965² + 8.0069960² = 48.8880151 + 64.1119849 = 113.0000000 = 7² + 8²
```

The small differences (for example `24.9999998`) come from the rounding of
the 7-decimal values in this hand calculation. The program calculates with
`f64` and gets the lengths within `1e−14`.

**Result:**

```text
x' = [−1.1426397, 1.9220756, 2.5856788, 4.2795169, 4.9397510, 6.0496992, 6.9919965, 8.0069960]
‖x‖² = 5 + 25 + 61 + 113 = 204 = ‖x'‖²
```

**Observations:**

- Pair 0 (fast) changed a lot: its angle increased by 1 rad ≈ 57°.
- Pair 3 (slow) changed very little: its angle increased by 0.001 rad
  ≈ 0.06°.
- Every pair kept its length. The total length is the same.

**Complex check of pair 0:** `(1 + 2i)·e^(i·1) = (cos 1 − 2 sin 1) +
i(sin 1 + 2 cos 1) = −1.1426397 + 1.9220756i`. The same values.

**Position 0:** all angles are 0, so `x' = x` exactly.
**Position 2:** the angles are `[2, 0.2, 0.02, 0.002]`. Pair 0 becomes
`(−2.2347417, 0.0770038)`.

### 3.10 Numerical precision

- **Position 0 is exact.** `cos 0 = 1` and `sin 0 = 0` exactly in IEEE 754.
  Thus `x·1 − y·0 = x` exactly. The test uses `assert_eq!`.
- **The length error** is a few `ε` (`f64::EPSILON ≈ 2.2e−16`) for every
  position, because `cos²φ + sin²φ` is 1 within a few `ε` for every
  `φ`. We measured a squared-norm error of at most `1.1e−14` for
  `‖x‖² ≈ 20` and positions up to `1.3 × 10⁷`.
- **The composition error grows with the position.**
  `RoPE(RoPE(x, m), n)` and `RoPE(x, m + n)` use the angles
  `m·θ + n·θ` and `(m + n)·θ`. Each has an absolute rounding error of about
  `(m + n)·ε`. We measured the largest component difference:

  | `m + n` | 13 | 1 333 | 133 333 | 1 333 333 | 13 333 333 |
  |---|---|---|---|---|---|
  | max difference | 4.4e−16 | 2.2e−14 | 2.6e−12 | 3.9e−11 | 3.3e−11 |

  The test `invariant_e_large_positions_and_bases` uses the bound
  `4·ε·(m + n + 10)·max|x|`. This bound follows from the error model and
  passes for all bases that we tested.
- Thus: the **length** is always safe. The **exact angle** at very large
  positions has an error that grows linearly with `m`. In `f32`, this error
  is about `5 × 10⁸` times larger (Entry 9, section 3.8).

### 3.11 Where a Transformer uses this

A Transformer applies this transformation to the **query** and **key**
vectors of every attention head, at the position of their token, before it
calculates the attention scores. Stage 5 implements this.

## 4. Rust concepts learned

### 4.1 Slices: `&[f64]` and `&mut [f64]`

A **slice** is a view into a sequence of values that somebody else owns. It
is a pointer and a length (16 bytes). `&[f64]` allows reading. `&mut [f64]`
allows reading and writing.

```rust
let v = vec![1.0, 2.0, 3.0, 4.0];
let all: &[f64] = &v;          // the whole Vec
let part: &[f64] = &v[1..3];   // [2.0, 3.0], no copy
```

`rotate` takes `&[f64]`. Thus the caller can pass a `Vec`, an array, or a
part of a larger buffer. The test `rotate_in_place_works_on_a_sub_slice`
rotates 4 values in the middle of a 6-value array.

**The borrow rules.** At one time, a value can have **either** many `&`
borrows **or** one `&mut` borrow. Thus, while `rotate_in_place` holds
`&mut embedding`, no other code can read or change it. This prevents data
races at compile time.

### 4.2 `chunks_exact(2)` and `chunks_exact_mut(2)`

```rust
let v = [1.0, 2.0, 3.0, 4.0, 5.0];
let mut chunks = v.chunks_exact(2);
assert_eq!(chunks.next(), Some(&[1.0, 2.0][..]));
assert_eq!(chunks.next(), Some(&[3.0, 4.0][..]));
assert_eq!(chunks.next(), None);
assert_eq!(chunks.remainder(), &[5.0]);   // the incomplete chunk
```

`chunks_exact(n)` is an iterator over non-overlapping sub-slices of exactly
`n` elements. If the length is not a multiple of `n`, the last incomplete
part goes to `remainder()`. (`chunks(n)` instead would return the short last
chunk as a normal item.)

For RoPE, `n = 2` gives exactly the adjacent pairs. We validate the length
first (`d` is even), so the remainder is always empty.

`chunks_exact_mut(2)` gives `&mut [f64]` sub-slices. The borrow checker
accepts many mutable chunks at the same time, because the chunks do not
overlap.

We use `chunks_exact` and not `chunks`, because "exactly 2" lets the
compiler know the length of every chunk. Then it can remove the bounds
checks of `pair[0]` and `pair[1]`.

### 4.3 How the iterators process the pairs: `zip`

```rust
for (pair, &theta) in embedding.chunks_exact_mut(2).zip(&self.inverse_frequencies) {
    let phi = angle_unchecked(position, theta);
    let rotated = rotate_pair([pair[0], pair[1]], phi);
    pair.copy_from_slice(&rotated);
}
```

`a.zip(b)` creates one iterator from two. Each item is a tuple `(itemA,
itemB)`. It stops when the **shorter** iterator ends. Here, both have
`d/2` items:

```text
chunks_exact_mut(2): [x0,x1]  [x2,x3]  [x4,x5]  [x6,x7]
&inverse_frequencies:  &θ₀      &θ₁      &θ₂      &θ₃
zip:                 ([x0,x1],&θ₀) ([x2,x3],&θ₁) ...
```

The pattern `(pair, &theta)` takes the tuple apart and copies the `f64`
out of the reference `&θⱼ`.

`&self.inverse_frequencies` is `&Vec<f64>`. `zip` accepts anything that
can become an iterator (`IntoIterator`). `&Vec<f64>` becomes an iterator over
`&f64`.

### 4.4 `enumerate()` as an alternative to `zip`

The same loop with an explicit pair index `j`:

```rust
for (j, pair) in embedding.chunks_exact_mut(2).enumerate() {
    let theta = self.inverse_frequencies[j];   // indexed access, bounds check
    // ...
}
```

`enumerate()` gives `(0, item0), (1, item1), ...`. This is useful when we
need `j` itself (the tests use it to print the pair index). In the main
loop we do not need `j`, only `θⱼ`. `zip` gives `θⱼ` directly, without an
index and without a bounds check. Both versions produce the same machine
code in practice.

### 4.5 `copy_from_slice`

`pair.copy_from_slice(&rotated)` copies the two values of the array
`rotated` into the two slots of `pair`. It panics if the lengths differ; here
both are 2. It is the same as `pair[0] = rotated[0]; pair[1] = rotated[1];`.

### 4.6 Ownership of the returned vector

```rust
pub fn rotate(&self, embedding: &[f64], position: Position) -> Result<Vec<f64>, RopeError> {
    validate_embedding_dimension(embedding.len(), self.dimension())?;
    let mut output = embedding.to_vec();
    self.rotate_pairs(&mut output, position);
    Ok(output)
}
```

1. `embedding` is borrowed. The caller still owns it.
2. `to_vec()` allocates a new `Vec` and copies the `d` values. The function
   owns `output`.
3. `rotate_pairs(&mut output, ...)` borrows `output` mutably for the loop.
   The borrow ends when the call returns.
4. `Ok(output)` **moves** `output` to the caller. Only the 24-byte header
   moves. The heap data stays where it is. The caller now owns it.

### 4.7 When cloning is necessary, and when it is not

- **Necessary:** in `rotate`. The caller keeps the input and gets a new
  output. Two vectors exist, so one copy (`to_vec`) is unavoidable.
- **Not necessary:** in `rotate_in_place`. The caller gives up the old
  values. The function writes into the existing memory.
- **Not necessary:** the frequencies. `RotaryEmbedding` owns them. Each call
  borrows them (`&self`). They are never copied.
- **Not necessary:** a pair. `[f64; 2]` is `Copy`. `rotate_pair` receives a
  copy on the stack, not a heap clone.
- **In tests:** `x.clone()` is necessary only when a test needs the
  original and the in-place result at the same time.

### 4.8 `pub(crate)`

`angle_unchecked` in `frequencies.rs` is `pub(crate)`: visible in the whole
crate, invisible to users of the library. The embedding code can use the
one angle formula without the `Result` of the public `angle` function. Its
frequencies come from a validated `RopeConfig`, so a check would be useless
work.

### 4.9 A struct that caches derived data

`RotaryEmbedding` keeps `inverse_frequencies: Vec<f64>`. They are
**derived** from `config`. The struct calculates them one time in `new`.
The fields are private, so nobody can change the frequencies without the
configuration. This is the reason for the struct: "do not repeat frequency
calculations unnecessarily".

## 5. Memory and execution

| Value | Size | Location |
|---|---|---|
| `RotaryEmbedding` | 40 B (16 B config + 24 B `Vec` header) + `d/2·8` B | stack + heap |
| input of `rotate` | not copied until `to_vec` | borrowed |
| output of `rotate` | 24 B header + `d·8` B | stack + heap (one allocation) |
| a pair inside the loop | 16 B | registers |

**Complete execution flow of `rope.rotate(&x, m)`:**

```text
1. validate_embedding_dimension(x.len(), d)  → Err(...) returns early, no allocation
2. output = x.to_vec()                        → one heap allocation of d·8 bytes, copy
3. rotate_pairs(&mut output, m):
     for each (pair j, θⱼ):                   → d/2 iterations
        φ = m as f64 · θⱼ                     → 1 multiplication
        (sin φ, cos φ) = sin_cos(φ)           → 1 trigonometric call (inside rotate_vector)
        [a, b] = [x·cos − y·sin, x·sin + y·cos] → 4 multiplications, 2 additions
        write a, b into the same two slots
4. Ok(output)                                  → move to the caller
```

`rotate_in_place` is the same without step 2 and with a move of nothing.

**When values go out of scope:** the caller's output `Vec` is dropped when
its variable goes out of scope; `drop` frees the `d·8` bytes. The pairs and
angles in the loop are stack or register values; they need no cleanup.

## 6. Engineering decisions

1. **`rotate_pair` calls `math::matrix::rotate_vector`.** The rotation
   formula exists once. The RoPE module only adds the RoPE name and
   documentation.
2. **A struct `RotaryEmbedding`.** It computes the frequencies once. A free
   function `rope(config, x, m)` would recompute `d/2` powers per call.
3. **Two methods: new vector and in place.** `rotate` is convenient and
   safe (the input stays). `rotate_in_place` avoids the allocation. Both use
   one private loop (`rotate_pairs`), so the logic is not duplicated.
4. **Validation before allocation.** An invalid input costs nothing.
5. **No check of the component values.** NaN or infinity stays in its own
   pair (test `non_finite_values_stay_in_their_pair`). A check would cost
   `d` comparisons per call, and RoPE cannot repair bad values.
6. **One layout (interleaved).** It matches the mathematics most directly.
   Split-half is documented, not implemented.
7. **No new configuration option.** See `config.rs`: `d` and `b` are
   sufficient.
8. **A new error variant `EmbeddingDimensionMismatch`.** It contains the
   expected and the actual length.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| Build the `d × d` block-diagonal matrix | Direct matrix view | `d²` memory and `d²` multiplications; almost all zeros |
| Use `Complex` for each pair | Matches the LLaMA code | Needs conversion from and to `f64` pairs; the same arithmetic |
| An iterator chain with `flat_map` + `collect` for `rotate` | No `to_vec` + overwrite | `flat_map` does not report an exact length, so `collect` may reallocate; harder to read |
| Only `rotate_in_place` | No allocation at all | The caller must clone manually if it needs the original |
| Return `[f64; D]` with const generics | No heap | `d` is a run-time value |
| A `cos`/`sin` cache per position | No trigonometry per call | Memory `L·d/2·16` bytes; premature now (Stage 6) |
| Split-half layout | Matches Hugging Face weights | Not needed for learning now |

## 8. Performance considerations

- **Time:** `O(d)` per vector: `d/2` iterations.
- **Trigonometry:** `d/2` `sin_cos` calls per vector. This is the main cost
  (about 20 to 100 cycles each). The arithmetic is 5 multiplications and 2
  additions per pair.
- **Space:** `rotate`: `d·8` bytes for the output (one allocation).
  `rotate_in_place`: no allocation. The struct: `d/2·8` bytes of
  frequencies, allocated once.
- **Can sine and cosine be reused?** Yes. `cos(m·θⱼ)` and `sin(m·θⱼ)` depend
  only on `m` and `j`, not on the vector. All vectors at the same position
  (all heads, all layers, queries and keys) use the same values. A table of
  `L·d/2` pairs `(cos, sin)` removes all trigonometry from the inner loop.
  This is the "RoPE cache" of many implementations. We do not build it now.
- **Does in place reduce allocations?** Yes: from one per call to zero.
  For many vectors, the allocation (about 20 to 100 ns) can cost as much as
  the arithmetic for small `d`.
- **Later optimizations:** the `cos`/`sin` cache; SIMD (Single Instruction,
  Multiple Data) over several pairs; batches of vectors in one contiguous
  buffer (Stage 5 prepares this); `f32` with care for large positions.

## 9. Practical example

```rust
use rust_rope::rope::config::RopeConfig;
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::Position;

let rope = RotaryEmbedding::new(RopeConfig::new(8, 10_000.0)?);
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
println!("{:?}", rope.rotate(&x, Position::new(1)?)?);

let mut v = x;                                   // `[f64; 8]` is Copy
rope.rotate_in_place(&mut v, Position::new(1)?)?;
println!("{v:?}");

println!("{}", rope.rotate(&[1.0, 2.0, 3.0], Position::ZERO).unwrap_err());
```

Expected output (from `cargo run --example basic_rope`):

```text
[-1.1426396637476532, 1.922075596544176, 2.585678829246765, 4.279516911052588, 4.939751002078326, 6.049699169170825, 6.991996501333625, 8.006995998833666]
[-1.1426396637476532, 1.922075596544176, 2.585678829246765, 4.279516911052588, 4.939751002078326, 6.049699169170825, 6.991996501333625, 8.006995998833666]
the embedding has 3 components, but the RoPE configuration has the dimension 8
```

## 10. Test results and what we learned

**Test results** (`cargo test`, 2026-11-01): all tests pass. New Stage 4
tests:

- Unit tests: 7 in `rotations.rs`, 7 in `embeddings.rs`, 1 in
  `validation.rs`.
- `tests/rope_tests.rs`: 13 new API tests (`d = 2, 4, 8`, the worked
  example, positions 0 and 1, positions 0..9, bases, invalid lengths, zero
  vector, negative values, new vector versus in place, sub-slices).
- `tests/numerical_tests.rs`: 10 new invariant tests: A (position 0), B
  (pair lengths), C (composition, pair and vector form), D (each pair gets
  `m·θⱼ`), E (large positions and bases), complex form, linearity, relative
  dot product, inverse rotation.

**Mathematics:**

- RoPE divides `x` into `d/2` adjacent pairs and rotates pair `j` by
  `m·θⱼ`. As one matrix, this is block-diagonal: `R_m = diag(R(m·θⱼ))`.
- In complex form: `zⱼ ↦ zⱼ·e^(i·m·θⱼ)`.
- Each pair keeps its length: the cross terms cancel, and
  `sin² + cos² = 1`.
- RoPE multiplies instead of adds: it changes directions, not lengths, and
  needs no parameters.
- `(R_m u)ᵀ(R_n v) = uᵀ R_(n−m) v`: only the distance counts.
- Interleaved and split-half layouts are both valid but not interchangeable.
- The length is always exact within a few `ε`; the angle error grows with
  `m`.

**Rust:**

- Slices `&[f64]` / `&mut [f64]` borrow a part of memory without copying.
- `chunks_exact_mut(2)` gives non-overlapping mutable pairs; `zip` walks
  pairs and frequencies together; `enumerate` adds an index when needed.
- `to_vec` is the one necessary copy for a new output; an in-place version
  needs no allocation.
- Returning a `Vec` moves only its header.
- `pub(crate)` shares an internal helper inside the crate.
- A struct can own derived data (the frequencies) so that it is computed
  once.


---

# Entry 11 — Attention tensor shapes and the flat `Tensor4` (Stage 5, Component A)

**Date:** 2026-11-01
**Stage:** 5 (RoPE with query and key tensors)
**Files:** `src/math/tensor.rs` (new), `src/math/mod.rs`,
`tests/rope_tests.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `TensorShape { batch, sequence, heads, features }` | ✅ Implemented |
| `Tensor4`: `new`, `zeros`, `from_fn`, `index`, `get`, `get_mut`, `vector`, `vector_mut`, `as_slice`, `as_mut_slice`, `into_vec` | ✅ Implemented |
| `TensorError` (zero extent, overflow, length mismatch) | ✅ Implemented |
| Learned projections `W_Q`, `W_K`, `W_V`, softmax, masking | ❌ Not implemented (outside the scope) |
| Strided views, transposes, general N-D tensors | ❌ Not implemented (not needed) |

**Note on the stage numbering (2026-11-01).** The roadmap in Entry 1 and
the README had "Stage 5 = Advanced RoPE (scaling)". The project owner
redefined the stages: Stage 5 = query/key integration, Stage 6 = PyTorch
reference validation, Stage 7 = layouts and partial RoPE, Stage 8 =
scaling, Stage 9 = CPU performance. Older entries keep their original
stage numbers.

---

## 1. What did we build?

We built a four-dimensional array type, `Tensor4`. It stores the values of
the logical shape `[B, S, H, D]` in **one** flat `Vec<f64>`. It calculates
the position of each element with an index formula.

## 2. Why did we build it?

A Transformer does not rotate one vector at a time. It rotates all query
and key vectors of all heads, of all tokens, of all sequences in a batch.
These vectors form a four-dimensional tensor. We need a container for this
data before we can apply RoPE to it (Entry 12).

## 3. Mathematical foundation

### 3.1 Tensors

A **tensor** here is a multi-dimensional array of numbers. A vector is a
1-D tensor (one index). A matrix is a 2-D tensor (two indices: row,
column). A 4-D tensor has four indices. The **shape** lists the number of
values along each index (each **axis**). Example: shape `[2, 4, 2, 4]` has
`2·4·2·4 = 64` values.

### 3.2 Input hidden states: `X ∈ R^(B × S × D_model)`

- **Batch size `B`.** A model processes several independent sequences
  together (a **batch**), because this uses the hardware better. Sequence
  `b` never influences sequence `b' ≠ b` in attention.
- **Sequence length `S`.** The number of tokens in each sequence. Token
  `s = 0, 1, ..., S − 1`.
- **Hidden dimension `D_model`.** The length of the vector for one token.
- **Hidden state.** The vector `X[b, s, :]` (all `D_model` values of token
  `s` in sequence `b`). It represents the token **in its context**: the
  first layer receives the token embedding; each later layer receives the
  output of the previous layer.

**Example.** `B = 2` sentences, `S = 4` tokens each, `D_model = 8`:

```text
X[0] = "the cat sat down":   4 tokens × 8 numbers
X[1] = "a dog ran fast":     4 tokens × 8 numbers
X[0, 1, :] = hidden state of "cat", for example
           = [0.1, −0.3, 0.7, 0.2, −0.5, 0.4, 0.0, 0.9]
```

`X` has `2 · 4 · 8 = 64` numbers.

### 3.3 Query, key and value projections

```text
Q = X·W_Q,      K = X·W_K,      V = X·W_V,      W_Q, W_K, W_V ∈ R^(D_model × D_model)
```

The product is applied to every token vector: `Q[b, s, :] = X[b, s, :]·W_Q`.

- **Query `q`** ("what does this token look for?"). It is compared with all
  keys.
- **Key `k`** ("what does this token offer for the comparison?"). The
  attention score of query token `m` and key token `n` is the dot product
  `q_mᵀ k_n` (scaled by `1/√D` and normalized by softmax in a full model).
- **Value `v`** ("what information does this token give?"). The output of
  attention for token `m` is a weighted sum of the values `v_n`, with the
  weights from the scores.
- **The projection matrices** `W_Q`, `W_K`, `W_V` are learned parameters.
  Each one is a linear map: it selects and mixes the features of the hidden
  state for its role. We do not implement or train them. RoPE starts
  **after** the projections.

### 3.4 Multi-head shapes

**Multi-head attention** runs `H` attention operations ("heads") in
parallel. Each head uses its own part of the projected vector:

```text
D_model = H · D        (D = head dimension)
```

Example: `D_model = 8`, `H = 2` → `D = 4`. The projected query of one token,
`[q₀, q₁, ..., q₇]`, is divided into:

```text
head 0: [q₀, q₁, q₂, q₃]        head 1: [q₄, q₅, q₆, q₇]
```

The **reshape** `[B, S, D_model] → [B, S, H, D]` only **reinterprets** the
indices. The values stay in the same memory order. Hidden index `c` becomes
`(h, d)` with `c = h·D + d`. Thus the reshape costs nothing.

For the task example, `B = 2, S = 4, H = 2, D = 4`:

```text
Q, K, V ∈ R^(2 × 4 × 2 × 4)
axis 0 (b): which sequence      0..2
axis 1 (s): which token         0..4  → the token position m = start + s
axis 2 (h): which head          0..2
axis 3 (d): which feature       0..4  → RoPE pairs (d₀, d₁), (d₂, d₃)
```

### 3.5 Flat storage: derivation of the index formula

We store the values in **row-major** order: the last index changes fastest.
Count how many values come before element `(b, s, h, d)`:

1. Before batch element `b`: `b` complete batch elements, each with
   `S·H·D` values → `b·S·H·D`.
2. Inside batch element `b`, before token `s`: `s` complete tokens, each
   with `H·D` values → `s·H·D`.
3. Inside token `s`, before head `h`: `h` complete heads, each with `D`
   values → `h·D`.
4. Inside head `h`, before feature `d`: `d` values.

```text
index = b·S·H·D + s·H·D + h·D + d
      = ((b·S + s)·H + h)·D + d            (factor out D, then H, then S)
```

The second form is the **Horner form**: it needs only 3 multiplications and
3 additions.

**Example** (`S = 4, H = 2, D = 4`): element `(b, s, h, d) = (1, 2, 1, 3)`:

```text
((1·4 + 2)·2 + 1)·4 + 3 = (6·2 + 1)·4 + 3 = 13·4 + 3 = 55
check: 1·32 + 2·8 + 1·4 + 3 = 32 + 16 + 4 + 3 = 55
```

The last element `(1, 3, 1, 3)` has index `63 = 64 − 1`.

**Strides.** The **stride** of an axis is the index step for a step of 1
on that axis: `d` → 1, `h` → `D`, `s` → `H·D`, `b` → `S·H·D`. For the
example: strides `[32, 8, 4, 1]`.

### 3.6 Memory layout and cache locality

```text
memory: [ b0 s0 h0 (4 values) | b0 s0 h1 (4) | b0 s1 h0 (4) | ... | b1 s3 h1 (4) ]
          └── one head vector: contiguous ──┘
          └──────── one token: H·D = 8 contiguous values ────────┘
```

The processor loads memory in **cache lines** (usually 64 bytes = 8 `f64`).
When code reads memory in increasing order, each cache line is used
completely, and the hardware **prefetcher** loads the next lines early. This
is **spatial locality**. RoPE reads one head vector (`D` contiguous values)
at a time, and our loop visits the head vectors in memory order. Thus the
traversal is a single sequential pass.

## 4. Rust concepts learned

### 4.1 Structs with a private `Vec` and a public shape

`Tensor4 { shape: TensorShape, data: Vec<f64> }` has private fields. Thus
the invariant `data.len() == B·S·H·D` can only be created by the checked
constructors. `TensorShape` has public fields, because a shape alone has no
invariant.

### 4.2 Ownership in `new`

```rust
pub fn new(shape: TensorShape, data: Vec<f64>) -> Result<Self, TensorError>
```

`data: Vec<f64>` (not `&[f64]`) means the function **takes ownership**. The
`Vec` header (24 bytes) moves into the struct. The heap buffer does not
move or copy. After the call, the caller cannot use its variable:

```rust
let v = vec![0.0; 4];
let t = Tensor4::new(TensorShape::new(1, 1, 1, 4), v)?;
// println!("{v:?}");   // error[E0382]: borrow of moved value: `v`
```

`into_vec(self)` does the reverse: it consumes the tensor and returns the
buffer, again without a copy.

### 4.3 Borrowing and mutable references

- `get(&self, ...) -> Option<f64>` returns a copy of one value.
- `get_mut(&mut self, ...) -> Option<&mut f64>` returns a **mutable
  reference**: a pointer through which the caller can write
  (`*x = 7.5`). While it exists, the whole tensor is mutably borrowed. The
  compiler rejects any other access to the tensor during that time.
- `vector(&self, b, s, h) -> Option<&[f64]>` returns a slice that borrows
  the tensor. `vector_mut` returns `&mut [f64]`.

A test showed this rule in practice. This test did not compile:

```rust
assert_eq!((q2, k2), (q3, k3));   // moves q3 and k3 into a tuple
r.rotate_tensor_in_place(&mut single, ...);
assert_eq!(single, q3);            // error[E0382]: borrow of moved value: `q3`
```

The fix compares references: `assert_eq!((&q2, &k2), (&q3, &k3))`.

### 4.4 `Option` for out-of-range indices

`index(b, s, h, d)` checks every index against its extent. A check of only
the final flat index is **not** sufficient: with `D = 4`, `(0, 0, 0, 5)`
gives the flat index 5, which is valid, but it is the element
`(0, 0, 1, 1)`. The unit test `out_of_range_indices_return_none` checks this.

### 4.5 Errors in `math` without `utils`

The architecture rule says that `math` does not depend on `rope` or
`utils`. Thus `tensor.rs` has its own error enum, `TensorError`. The axis
name in `ZeroExtent { axis: &'static str }` is a **string literal**: a
`&'static str` lives for the whole program, so the error can store it
without an allocation.

### 4.6 `from_fn` with `impl FnMut`

```rust
pub fn from_fn(shape: TensorShape, mut f: impl FnMut(usize, usize, usize, usize) -> f64)
```

`impl FnMut(...)` accepts any closure with this signature. The compiler
creates one copy of `from_fn` for each closure type (**monomorphization**),
so the call `f(b, s, h, d)` is direct and can be inlined. `FnMut` (not `Fn`)
allows the closure to change captured variables. The unit test
`index_formula_matches_memory_order` uses a counter inside the closure; this
also proves that `from_fn` visits the elements in memory order.

### 4.7 Checked multiplication of the shape

`B·S·H·D` can overflow. `num_elements` uses `checked_mul` three times with
`?` on `Option`. Every extent must also be at least 1: a zero extent would
make `chunks_exact_mut(0)` panic in the RoPE loop.

## 5. Memory and execution

| Value | Size | Location |
|---|---|---|
| `TensorShape` | 32 B | stack (`Copy`) |
| `Tensor4` | 56 B (32 shape + 24 `Vec` header) | stack |
| data | `B·S·H·D·8` B | heap, one block |

Example sizes: `[2, 4, 2, 4]` → 512 B. `[2, 512, 16, 64]` → 8 MiB.

## 6. Engineering decisions

1. **A new module `math/tensor.rs`.** A 4-D array is a general data
   structure, not RoPE logic, and no existing `math` module fits it.
2. **No tensor framework** (for example `ndarray`). The standard library is
   sufficient for one fixed rank and row-major order.
3. **Named axes in `TensorShape`** instead of `[usize; 4]`: `shape.heads` is
   clearer than `shape[2]` and prevents axis mix-ups.
4. **Fixed rank 4.** Rank-generic code (const generics) adds complexity
   without a current need.
5. **Zero extents are errors** (see 4.7).

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| `Vec<Vec<Vec<Vec<f64>>>>` | Natural `t[b][s][h][d]` syntax; ragged shapes possible | `B·S·H + B·S + B + 1` heap allocations; pointer chasing; the rows can be anywhere in memory; no single slice for SIMD or for a GPU copy; the shape can become inconsistent |
| Flat `Vec<f64>` (our choice) | One allocation; contiguous; any sub-block is a slice; matches the layout of PyTorch, NumPy and GPU buffers; one `memcpy` to move to another device | Index arithmetic in code; the shape must be validated |
| `ndarray` crate | General N-D, strides, views | External dependency |
| `[[[[f64; D]; H]; S]; B]` | No heap; compile-time shape | Sizes must be constants; large arrays overflow the stack |

**Why the flat representation helps later CPU and GPU optimization.** SIMD
instructions load several contiguous values at once. A thread pool can give
each thread a contiguous sub-slice (`chunks_mut`) without any overlap. A GPU
copy is one contiguous transfer. None of this is possible with nested `Vec`s
without first copying into a flat buffer.

## 8. Performance considerations

- `index`: 4 comparisons + 3 multiplications + 3 additions.
- `from_fn`: one allocation, `B·S·H·D` sequential `push` operations.
- `vector(b, s, h)`: no copy; one slice of `D` values.

## 9. Practical example

```rust
use rust_rope::math::tensor::{Tensor4, TensorShape};

let t = Tensor4::from_fn(TensorShape::new(2, 4, 2, 4), |b, s, h, d| {
    (1000 * b + 100 * s + 10 * h + d) as f64
})?;
println!("{:?}", t.index(1, 2, 1, 3));      // Some(55)
println!("{:?}", t.get(1, 2, 1, 3));        // Some(1213.0)
println!("{:?}", t.vector(0, 1, 1));        // Some([110.0, 111.0, 112.0, 113.0])
println!("{:?}", t.index(0, 0, 0, 4));      // None
```

## 10. What we learned

**Mathematics:** hidden states `[B, S, D_model]`; projections `Q = XW_Q`,
`K = XW_K`, `V = XW_V`; `D_model = H·D`; the reshape into heads is a
reinterpretation of indices; the row-major index
`((b·S + s)·H + h)·D + d` and the strides `[S·H·D, H·D, D, 1]`.

**Rust:** a struct with a private `Vec` keeps a shape invariant; `new`
takes ownership of the buffer without copying; `&mut f64` and `&mut [f64]`
borrow mutably; `Option` for safe indexing; `impl FnMut` and
monomorphization; `&'static str` in an error.

---

# Entry 12 — RoPE on query and key tensors; relative position in attention scores (Stage 5, Components B and C)

**Date:** 2026-11-01
**Stage:** 5
**Files:** `src/rope/embeddings.rs` (new methods),
`src/utils/validation.rs` (`QueryKeyShapeMismatch`),
`tests/rope_tests.rs`, `tests/numerical_tests.rs`,
`benches/rope_benchmark.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `RotaryEmbedding::rotate_tensor_in_place(&mut Tensor4, start)` | ✅ Implemented |
| `RotaryEmbedding::rotate_query_key_in_place(&mut q, &mut k, start)` | ✅ Implemented |
| `RotaryEmbedding::rotate_query_key(&q, &k, start) -> (q', k')` | ✅ Implemented |
| Configurable start position | ✅ Implemented |
| Value tensor unchanged (no value parameter) | ✅ Implemented and tested |
| Simple `std::time` benchmark | ✅ Implemented |
| Softmax, masking, KV cache, projections | ❌ Not implemented (outside the scope) |

---

## 1. What did we build?

We apply RoPE to every query and key head vector of `[B, S, H, D]`
tensors. The token at sequence index `s` has the position `m = start + s`.
The value tensor is not an input, so it cannot change.

## 2. Why did we build it?

This is where RoPE does its work in a Transformer: on queries and keys,
just before the attention scores. Section 3.4 proves that this makes every
score depend only on the relative distance of the two tokens.

## 3. Mathematical foundation

### 3.1 The operation

For every `b`, every `s` (position `m = start + s`) and every `h`:

```text
Q'[b, s, h, :] = R_m · Q[b, s, h, :]
K'[b, s, h, :] = R_m · K[b, s, h, :]
V'             = V                           (unchanged)
```

`R_m = diag(R(m·θ₀), ..., R(m·θ_(D/2−1)))` is the block-diagonal matrix of
Entry 10. `D` is the head dimension.

### 3.2 The same frequencies for all heads

`θⱼ = b^(−2j/D)` depends only on `D` and `b`. All heads have the same `D`
and use the same configuration. Thus all heads, all batch elements, the
queries and the keys use **the same** `θⱼ`, and at one position `m`, the
same matrix `R_m`. `RotaryEmbedding` stores `θⱼ` one time.

What differs between heads is the **content** of the vectors (each head has
its own part of `W_Q` and `W_K`), not the rotation.

### 3.3 Why queries and keys, but not values

Attention uses the vectors in two different ways:

1. **Scores** compare a query with a key: `q_mᵀ k_n`. This is where the
   relation between two positions matters.
2. **Outputs** mix the values: `out_m = Σₙ wₘₙ · v_n`, with weights `wₘₙ`
   from the scores.

Standard RoPE is **designed** to put position information into the scores,
and only there. If both `q_m` and `k_n` are rotated, the score depends on
`n − m` (section 3.4). The values carry content; the weights already
contain the position information.

Rotating the values is mathematically possible. It would be a different
design. It would make the output of token `m` a mix of vectors rotated by
the absolute positions `n` of the other tokens, so the output would carry
absolute position information, which is what RoPE wants to avoid. Standard
RoPE (RoFormer, LLaMA and most models) does not do this. Our API makes the
standard design explicit: there is no value parameter.

### 3.4 Derivation of the relative-position property

Let `Q'_m = R(m)·Q_m` and `K'_n = R(n)·K_n`, where `R(m)` is the
block-diagonal `R_m`. The score is:

```text
(Q'_m)ᵀ K'_n
 = (R(m)·Q_m)ᵀ (R(n)·K_n)                   substitute
 = Q_mᵀ R(m)ᵀ R(n) K_n                      (A·x)ᵀ = xᵀ·Aᵀ
 = Q_mᵀ R(−m) R(n) K_n                      orthogonality: R(m)ᵀ = R(m)⁻¹ = R(−m)
 = Q_mᵀ R(n − m) K_n                        composition: R(a)·R(b) = R(a + b)
```

**Every step explained:**

1. **Substitute** the definitions of `Q'_m` and `K'_n`.
2. **Transpose of a product.** `(A·x)ᵀ = xᵀAᵀ` (the order reverses).
3. **Orthogonality.** For each 2 × 2 block, `R(φ)ᵀ = R(−φ)` (Entry 5:
   transposing swaps the signs of `sin φ`, and `sin(−φ) = −sin φ`). A
   block-diagonal matrix is transposed block by block. Thus
   `R(m)ᵀ = diag(R(−m·θⱼ)) = R(−m)`.
4. **Composition.** Block `j`: `R(−m·θⱼ)·R(n·θⱼ) = R((n − m)·θⱼ)`.
   Block-diagonal matrices multiply block by block, so the pairs compose
   **independently**: `R(−m)·R(n) = diag(R((n − m)·θⱼ)) = R(n − m)`.

**Why the result contains only `n − m`.** The middle matrix `R(n − m)`
depends only on the difference. `Q_m` and `K_n` carry the content. Thus the
absolute positions vanish from the score. Shifting every position by the same
amount changes nothing (test
`tensor_scores_do_not_depend_on_the_start_position`).

**Per pair:** the score is a sum over the pairs,
`Σⱼ (q-pair j)ᵀ R((n − m)·θⱼ) (k-pair j)`. Fast pairs (large `θⱼ`) react to
small distances; slow pairs react to large distances.

### 3.5 Numerical demonstration in two dimensions

Choose: `q = [1, 2]`, `k = [3, −1]`, frequency `θ = 0.5`, query position
`m = 2`, key position `n = 5`.

**1. Original query:** `q = [1, 2]`. **2. Original key:** `k = [3, −1]`.
(Without rotation: `qᵀk = 1·3 + 2·(−1) = 1`.)

**3. Rotated query:** angle `m·θ = 2·0.5 = 1`.

```text
cos 1 = 0.5403023,  sin 1 = 0.8414710
q'₀ = 1·0.5403023 − 2·0.8414710 = 0.5403023 − 1.6829420 = −1.1426397
q'₁ = 1·0.8414710 + 2·0.5403023 = 0.8414710 + 1.0806046 =  1.9220756
```

**4. Rotated key:** angle `n·θ = 5·0.5 = 2.5`.

```text
cos 2.5 = −0.8011436,  sin 2.5 = 0.5984721
k'₀ = 3·(−0.8011436) − (−1)·0.5984721 = −2.4034308 + 0.5984721 = −1.8049587
k'₁ = 3·0.5984721 + (−1)·(−0.8011436) =  1.7954164 + 0.8011436 =  2.5965600
```

**5. Rotated dot product:**

```text
q'ᵀk' = (−1.1426397)·(−1.8049587) + 1.9220756·2.5965600
      = 2.0624174 + 4.9907847
      = 7.0532021
```

**6. The equivalent relative rotation:** angle `(n − m)·θ = 3·0.5 = 1.5`.

```text
cos 1.5 = 0.0707372,  sin 1.5 = 0.9974950
(R(1.5)k)₀ = 3·0.0707372 − (−1)·0.9974950 = 0.2122116 + 0.9974950 = 1.2097066
(R(1.5)k)₁ = 3·0.9974950 + (−1)·0.0707372 = 2.9924850 − 0.0707372 = 2.9217478
```

**7. Final dot product:**

```text
qᵀ R(1.5) k = 1·1.2097066 + 2·2.9217478 = 1.2097066 + 5.8434955 = 7.0532021
```

Both calculations give `7.0532021` (in `f64`: `7.053202107896084` for both;
the test `two_dimensional_relative_position_demonstration` checks this
within `1e−12`). The positions 2 and 5 appear only through their difference
3.

## 4. Rust concepts learned

### 4.1 Nested `chunks_exact_mut` that follow the layout

```rust
for batch_element in tensor.as_mut_slice().chunks_exact_mut(batch_len) {        // S·H·D
    for (token, position) in batch_element.chunks_exact_mut(token_len)           // H·D
                                          .zip(positions.iter()) {
        for head in token.chunks_exact_mut(shape.features) {                     // D
            self.rotate_pairs(head, position);
        }
    }
}
```

Each level divides the previous slice into non-overlapping sub-slices. The
three levels match the strides of the layout. Thus:

- **Batch boundaries** are preserved: a `batch_element` slice cannot reach
  into the next one.
- **Sequence order** is preserved: `zip(positions.iter())` gives token `s`
  the position `start + s`.
- **Head boundaries** are preserved: `rotate_pairs` receives exactly `D`
  values.

There is no index arithmetic in the loop at all, and thus no off-by-one
risk. The compiler checks that the slices do not overlap.

### 4.2 Reuse of the Stage 4 loop

The tensor code calls `rotate_pairs`, the same private loop as
`rotate_in_place`. The rotation logic exists once.

### 4.3 Two mutable borrows at the same time

`rotate_query_key_in_place(&self, query: &mut Tensor4, key: &mut Tensor4, ...)`
holds two `&mut` borrows. This is allowed because they are two different
values. If a caller tries `rotate_query_key_in_place(&mut t, &mut t, ...)`,
the compiler rejects it: two mutable borrows of the same value.

### 4.4 Validate everything, then mutate

`check_query_key` checks both tensors before the first write. This gives
"all or nothing" behavior: an error leaves both tensors unchanged (test
`invalid_head_dimensions`).

### 4.5 Clone only where necessary

`rotate_query_key(&q, &k, ...)` borrows the inputs, so it must clone them
(two allocations). `rotate_query_key_in_place` needs no clone.

## 5. Memory and execution

**Execution of `rotate_query_key_in_place(&mut q, &mut k, start)`:**

1. Check `(B, S)` equal, `D = d` for both, and `start + S − 1 ≤ 2^53`.
2. For `q`, then `k`: one sequential pass over the data. For each head
   vector: `D/2` angle multiplications, `D/2` `sin_cos` calls, `D/2` pair
   rotations.
3. No allocation.

**Ownership:** `self` (the frequencies) is shared-borrowed. `q` and `k` are
mutably borrowed during the call and are usable again after it.

## 6. Engineering decisions

1. **Methods on `RotaryEmbedding`** in `embeddings.rs`. Its planned
   responsibility (Stage 1) was "query and key vectors, and batches".
2. **Positions `start + s`**, the same for every batch element. This is the
   usual convention when all sequences in the batch are aligned.
   (Per-sequence position offsets, for example for padded batches, are not
   needed now.)
3. **`B` and `S` must match between `q` and `k`; `H` may differ.** Each
   tensor is rotated independently, so different head counts (grouped-query
   attention) need no extra code.
4. **No value parameter** (section 3.3).
5. **The `D ≠ d` error reuses `EmbeddingDimensionMismatch`**: a head vector is
   an embedding of length `D`.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| Index-based loops with the formula | Mirrors the mathematics | Off-by-one risk; bounds checks; the chunk version is clearer |
| `chunks_exact_mut(D)` over the whole slice + a computed `s = (v / H) % S` | One loop | Division per vector; hides the structure |
| Take `q`, `k` by value and return them | No clone, functional style | On an error, the tensors are dropped and lost |
| A `cos`/`sin` table per position, reused for all `B·H` vectors | `S·D/2` instead of `B·S·H·D/2` trig calls | Memory and setup; planned for Stage 9 |

## 8. Performance considerations

- **Time:** `O(B·S·H·D)`. **Extra space:** 0 (in place) or `2·B·S·H·D·8`
  bytes (copying version).
- **Trigonometry:** `B·S·H·D/2` `sin_cos` calls per tensor. The values
  depend only on `(s, j)`, so a table would need only `S·D/2` calls: a
  reduction by the factor `B·H` (for example 32 for `B = 2, H = 16`). Not
  implemented now (no premature optimization).
- **Memory access:** one sequential pass; every cache line is used fully.

**Benchmark** (`cargo bench`, release build, this computer, `std::time`,
best of 15 runs after 3 warm-up runs; `D = 64`, `b = 10000`):

| Shape `[B, S, H, D]` | `rotate_tensor_in_place` | per pair | `rotate_query_key` (q + k, 2 clones) |
|---|---|---|---|
| `[1, 128, 8, 64]` | 0.255 ms | 7.8 ns | 0.699 ms |
| `[1, 512, 8, 64]` | 1.114 ms | 8.5 ns | 3.153 ms |
| `[2, 512, 16, 64]` | 4.409 ms | 8.4 ns | 11.665 ms |

The per-pair time is almost constant: the cost is linear in the size.
`rotate_query_key` rotates **two** tensors and clones both: about 2 × the
in-place time plus about 3 ns per pair for the copies. These numbers are a
first measurement, not a rigorous benchmark.

## 9. Practical example

```rust
use rust_rope::math::tensor::{Tensor4, TensorShape};
use rust_rope::rope::{config::RopeConfig, embeddings::RotaryEmbedding, position::Position};

let rope = RotaryEmbedding::new(RopeConfig::new(4, 10_000.0)?);
let shape = TensorShape::new(2, 4, 2, 4);
let mut q = Tensor4::from_fn(shape, |b, s, h, d| (b + s + h + d) as f64)?;
let mut k = q.clone();
let v = q.clone();
rope.rotate_query_key_in_place(&mut q, &mut k, Position::ZERO)?;
assert_eq!(q.vector(0, 0, 0), v.vector(0, 0, 0)); // s = 0: position 0
assert_ne!(q.vector(0, 1, 0), v.vector(0, 1, 0)); // s = 1: rotated
// `v` was never passed to RoPE: unchanged.
```

## 10. Test results and what we learned

**Test results:** all tests pass (`cargo test`, 2026-11-01). New Stage 5
tests: 6 unit tests in `tensor.rs`; 11 API tests in `rope_tests.rs`
(batch 1 and 2, sequence 1 and more, several heads, `D = 2, 4, 8`, start
positions, invalid shapes, invalid head dimensions, value preservation,
batch and head isolation, sequence order); 5 numerical tests in
`numerical_tests.rs` (A: position 0, B: pair lengths, C: relative-position
property for all `(b, h, m, n)`, score independence of `start`, the 2-D
demonstration).

**Mathematics:** RoPE rotates `Q` and `K` at their positions, never `V`;
all heads share `θⱼ`; `(R(m)q)ᵀ(R(n)k) = qᵀR(n − m)k` follows from
`(Ax)ᵀ = xᵀAᵀ`, `R(m)ᵀ = R(−m)` and `R(a)R(b) = R(a + b)`, block by block.

**Rust:** nested `chunks_exact_mut` mirror a tensor layout without index
arithmetic; two simultaneous `&mut` borrows of different values are
allowed; validate first, then mutate; clone only when the caller keeps the
original.


---

# Entry 13 — Independent PyTorch reference validation and floating-point precision (Stage 6)

**Date:** 2026-11-01
**Stage:** 6 (reference validation)
**Files:** `examples/export_reference_cases.rs` (new), `reference/pytorch_rope.py`,
`reference/compare_rope.py`, `reference/test_pytorch_rope.py`,
`reference/README.md`, `reference/requirements.txt` (all new; Python
validation tooling), `.gitignore`

### Implementation status of this entry

| Item | Status |
|---|---|
| Rust export of deterministic cases (51 vectors, 15 tensors) | ✅ Implemented |
| Independent PyTorch reference (frequencies, angles, `cos`/`sin`, interleaved, split-half, partial) | ✅ Implemented |
| Comparison in float64 (strict) and float32 (precision study) | ✅ Implemented — **all 66 cases pass** |
| Reference self-tests and negative controls | ✅ 12 tests pass |
| Split-half and partial RoPE **in Rust** | ⏳ Stage 7 (Entry 14). In Stage 6, the comparison covers only the interleaved, full-rotation Rust code. |

**The Python code is validation tooling, not the production RoPE
implementation.** The library in `src/` does not depend on it. PyTorch
2.15.0 (CPU) and NumPy 2.5.4 run in a project-local virtual environment
`reference/.venv` (git-ignored).

---

## 1. What did we build?

A validation pipeline with three parts:

```text
cargo run --example export_reference_cases
        │  deterministic inputs + Rust outputs (text, exact f64)
        ▼
reference/data/rust_cases.txt
        │
        ▼
reference/compare_rope.py ──► pytorch_rope.apply_rope (float64 and float32)
        │
        ▼
max abs error, mean abs error, max relative error, tolerance → PASS / FAIL
```

## 2. Why did we build it?

Our tests so far check **properties** (length preservation, composition,
the relative-position identity) and values that **we** calculated. A
property test cannot detect a wrong convention that keeps the property. For
example, a wrong frequency formula still keeps every length. An
**independent reference** detects such errors: two implementations that
were written separately are unlikely to contain the same mistake.

## 3. Mathematical foundation

### 3.1 Why two independent implementations should agree

Both implementations calculate the same mathematical function:

```text
f(x, m)ₖ = x₂ⱼ cos(m·θⱼ) − x₂ⱼ₊₁ sin(m·θⱼ)    (k = 2j)
f(x, m)ₖ = x₂ⱼ sin(m·θⱼ) + x₂ⱼ₊₁ cos(m·θⱼ)    (k = 2j + 1),   θⱼ = b^(−2j/d)
```

In exact arithmetic, the results are identical. In floating-point
arithmetic, each operation rounds. The two implementations use different
operations (section 3.6), so their roundings differ. Thus we expect:

```text
Rust_RoPE(x, m) ≈ PyTorch_RoPE(x, m)      (equal within a tolerance, not bit for bit)
```

### 3.2 Error measures

Let `r` be the Rust output and `p` the PyTorch output, both with `N` values.

```text
max_error  = maxₖ |rₖ − pₖ|                    (the worst single value)
mean_error = (1/N) Σₖ |rₖ − pₖ|                (the typical value)
max_rel    = maxₖ |rₖ − pₖ| / |pₖ|             (only where |pₖ| ≥ 10⁻³ · max|p|)
```

**Absolute error** measures the difference in the units of the data.
**Relative error** measures the difference as a fraction of the value.
Example: an error of `0.001` is large for the value `0.002` (50 %), but small
for the value `1000` (0.0001 %).

**Why relative error can mislead near zero.** If `pₖ = 10⁻¹⁵` (a
mathematical zero with a rounding residue) and `rₖ = 2·10⁻¹⁵`, the relative
error is 100 %, although both values are correct within rounding. Thus we
calculate the relative error only for values that are not close to zero
compared with the largest value.

### 3.3 IEEE 754 floating-point numbers

IEEE 754 is the standard for binary floating-point numbers. A number is:

```text
value = (−1)^sign × 1.fraction × 2^(exponent − bias)
```

| Type | Bits | Sign | Exponent | Fraction | Significant bits | Machine epsilon `ε` |
|---|---|---|---|---|---|---|
| `f32` / `float32` | 32 | 1 | 8 | 23 | 24 | `2^−23 ≈ 1.19 × 10⁻⁷` |
| `f64` / `float64` | 64 | 1 | 11 | 52 | 53 | `2^−52 ≈ 2.22 × 10⁻¹⁶` |

**Machine epsilon `ε`** is the distance from 1.0 to the next larger
representable number. Every basic operation (`+ − × ÷ √`) gives the exact
result **rounded to the nearest representable number**. The relative error
of one rounding is at most `ε/2`. This is a "unit in the last place" (ulp)
level error.

### 3.4 How errors accumulate in RoPE

| Step | Operation | Error |
|---|---|---|
| `θⱼ` | `powf` (or `**` and `1/…`) | 1 to 2 roundings: relative `≈ ε` |
| `φ = m·θⱼ` | 1 multiplication | relative `≈ 2ε` → **absolute** `≈ 2ε·φ` |
| `cos φ`, `sin φ` | library functions | about 1 ulp each, **plus** the input error of `φ` |
| `x cos φ − y sin φ` | 2 multiplications, 1 subtraction | `≈ 3ε·|x|` |

The important row is `φ`. Its **absolute** error grows with the angle,
and the angle grows with the position. `cos` and `sin` change by about the
same absolute amount as their input. Thus the output error is about:

```text
|error| ≈ c · ε · (1 + φ_max) · max|x|          (c = a small constant)
```

This is our **tolerance model** with `c = 8`:

```text
tol = 8 · ε · (1 + max_angle) · max(1, max|x|)
```

- `ε` is the epsilon of the **reference** dtype.
- `max_angle = (start + S − 1) · θ₀` with `θ₀ = 1`.
- `8` covers the roughly 4 to 6 roundings of the table above.

The tolerance follows from the error analysis. It was not tuned to make
the tests pass.

### 3.5 float32 versus float64

In float32 mode, PyTorch calculates everything in float32. The Rust result
is float64. Thus the float32 error measures **the error of float32 RoPE**.
`ε₃₂/ε₆₄ = 2^29 ≈ 5.4 × 10⁸`. At large positions, the float32 angle error
becomes visible: at `m = 100 000`, `d = 128`, we measured a maximum error of
`1.15 × 10⁻²` (section 3.7). This is the reason why long-context
implementations calculate the angles carefully.

### 3.6 Why mathematically equal code gives slightly different numbers

1. **Different formula forms.** Rust: `θⱼ = b.powf(−2j/d)` (one rounding).
   PyTorch reference: `θⱼ = 1 / b^(2j/d)` (two roundings). The two results
   can differ by 1 ulp.
2. **Different library functions.** Rust calls the C runtime `sin`/`cos`.
   PyTorch can use its own vectorized implementations. Both are accurate to
   about 1 ulp, but not always identical.
3. **Different operation order.** For example `a·c − b·s` versus
   `a·c + (−b)·s` are identical, but `x·cos + rotate_half(x)·sin` with a
   different order of the products would round differently.

### 3.7 Validation results (2026-11-01)

**All 66 cases pass in both modes** (`compare_rope.py`, exit code 0).

| Mode | Worst max abs error | Largest mean abs error | Worst max rel error |
|---|---|---|---|
| float64 | `1.94 × 10⁻¹¹` | `6.84 × 10⁻¹³` | `3.62 × 10⁻¹¹` |
| float32 | `1.15 × 10⁻²` | `4.34 × 10⁻⁴` | `6.22 × 10⁻²` |

Representative cases (max abs error / tolerance):

| Case | float64 | float32 |
|---|---|---|
| `d = 2`, `m = 1` | `0` / `7.3e−15` | `6.5e−8` / `3.9e−6` |
| `d = 8`, `m = 100 000` | `0` / `3.8e−10` | `1.6e−5` / `2.0e−1` |
| `d = 128`, `m = 1000` | `1.2e−13` / `3.9e−12` | `7.3e−5` / `2.1e−3` |
| `d = 128`, `m = 100 000` | `1.9e−11` / `4.0e−10` | `1.2e−2` / `2.1e−1` |
| tensor `[3, 7, 2, 128]`, `b = 500 000`, start 4096 | `3.2e−13` / `1.6e−11` | `7.8e−4` / `8.8e−3` |
| tensor `[2, 4, 2, 4]`, start 0 | `0` / `1.6e−14` | `1.7e−7` / `8.3e−6` |

**Analysis of the float64 differences.** For `d = 2` and `d = 8`, the
float64 error is exactly 0, even at `m = 100 000`. For `d = 64` and
`d = 128`, it grows with `m`. We investigated before we drew a conclusion:

| `d` | `θⱼ` values that differ between `b^(−x)` and `1/b^x` | max relative difference | angle difference at `m = 10⁵` |
|---|---|---|---|
| 8 | 0 of 4 | 0 | 0 |
| 64 | 11 of 32 | `1.8 × 10⁻¹⁶` | `5.6 × 10⁻¹²` |
| 128 | 24 of 64 | `2.0 × 10⁻¹⁶` | `5.6 × 10⁻¹²` |

**Root cause:** the two frequency conventions (section 3.6, item 1) differ
by 1 ulp for some `θⱼ`. The position multiplies this difference. The output
difference `1.9 × 10⁻¹¹` equals about this angle difference times
`max|x| ≈ 2.25`, plus rounding. **There is no bug.** Both implementations
are correct within the error model.

### 3.8 The failure-analysis checklist

The task required a checklist for mismatches. We apply it to every new
comparison:

1. Same dimension layout? (case field `layout`)
2. Same frequency formula? (both `b^(−2j/r)`, different rounding form)
3. Same position indices? (`start + s` in both)
4. Same tensor interpretation? (`[B, S, H, D]`, row-major in both)
5. Same rotary dimension? (case field `rotary_dim`)
6. Same dtype? (Rust f64; reference f64 or f32 by mode)
7. Same base? (case field `base`, exact text)
8. A precision difference? (compare with the error model)

**Negative controls** prove that the comparison can fail: with a wrong
layout, a wrong base or a wrong start position, the error becomes larger
than `10⁻³` (tests in `test_pytorch_rope.py`).

## 4. Rust concepts learned

### 4.1 Exact text exchange with `{:?}`

`format!("{v:?}")` for an `f64` prints the **shortest** decimal string that
parses back to exactly the same bits (for example `0.1`, not
`0.1000000000000000055511151231257827`). Python's `float()` uses the same
round-trip rule. Thus the text file does not add any rounding error, and we
need no external serialization crate.

### 4.2 `std::fmt::Write` for building a `String`

```rust
use std::fmt::Write as _;
let _ = write!(out, " {v:?}");
```

`write!` normally writes to a stream. With the trait `std::fmt::Write` in
scope, it also writes into a `String`. `as _` imports the trait only for its
methods, without a name. Writing into a `String` cannot fail, so we ignore
the `Result` with `let _ =`.

### 4.3 `Box<dyn std::error::Error>` in a program

`main() -> Result<(), Box<dyn std::error::Error>>` lets `?` propagate
`RopeError`, `TensorError` and `std::io::Error` with one return type.
`dyn Error` is a **trait object**: a pointer to some value whose type
implements `Error`; the concrete type is decided at run time.

### 4.4 Python in brief (for readers who do not know PyTorch)

- `torch.tensor(list, dtype=torch.float64)`: an N-D array (like
  `Tensor4`).
- `x.reshape(B, S, H, D)`: reinterpret the same values with another shape.
- `x[..., :r]`: all leading axes, the first `r` features (like a Rust
  slice `&v[..r]` on each head vector).
- **Broadcasting:** `cos` with the shape `[1, S, 1, r/2]` multiplies a
  tensor `[B, S, H, r/2]`; PyTorch repeats `cos` along the axes of size 1.
  This is how one angle table serves every batch element and head.
- `torch.outer(p, θ)`: the table `p[s]·θ[j]` (Entry 9).

## 5. Memory and execution

The export program holds one case at a time in memory and appends text to
one `String` (about 200 KB for 66 cases). The comparison runs in less than
one second.

## 6. Engineering decisions

1. **The reference reads Rust's inputs**, so the inputs are identical by
   construction. The input generator exists only in Rust.
2. **A different frequency form in the reference** (`1/b^x`) to make the
   implementations more independent. Section 3.7 shows its measurable
   effect.
3. **Vectorized PyTorch code** (broadcasting, `stack`, `cat`) instead of a
   loop over pairs: a different algorithm structure from the Rust loop.
4. **A text format, not JSON or binary.** No dependency, human-readable,
   exact.
5. **A virtual environment** for PyTorch, so the global Python stays
   unchanged.
6. **The tolerance comes from an error model**, scaled with dtype, angle
   and magnitude. A fixed `1e−12` would be too strict for `m = 10⁵` and too
   loose for `m = 0`.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| Call Rust from Python (PyO3) | One process | Adds a heavy dependency; the reference would no longer be independent of the build |
| JSON with `serde` | Standard format | An external Rust dependency for one file |
| Hugging Face `transformers` RoPE | A widely used implementation | A large dependency; its functions change between versions; split-half only |
| Bitwise comparison | Simple | Fails for correct code (section 3.6) |
| A fixed tolerance | Simple | Ignores the growth of the error with the position |

## 8. Performance considerations

The validation is not performance-critical. The export and the comparison
each take less than one second for 66 cases.

## 9. Practical example

```text
$ cargo run --example export_reference_cases
wrote 66 cases to reference/data/rust_cases.txt
$ reference/.venv/Scripts/python reference/compare_rope.py
...
worst over all cases, float64: max abs 1.936e-11, largest mean abs 6.836e-13, max rel 3.619e-11
worst over all cases, float32: max abs 1.146e-02, largest mean abs 4.338e-04, max rel 6.217e-02

all 66 cases pass in float64 and float32
$ reference/.venv/Scripts/python -m unittest discover -s reference
Ran 12 tests ... OK
```

## 10. What we learned

**Mathematics and numerics:** an independent reference catches convention
errors that property tests cannot catch; absolute versus relative error;
IEEE 754 formats and `ε`; the absolute angle error grows with the position;
a tolerance derived from an error model; a 1-ulp difference in `θ` becomes a
`10⁻¹¹` output difference at `m = 10⁵`; float32 RoPE has errors near
`10⁻²` at `m = 10⁵`.

**Rust:** `{:?}` gives exact round-trip text for `f64`; `fmt::Write` into
a `String`; `Box<dyn Error>` for programs with several error types.


---

# Entry 14 — Interleaved and split-half layouts (Stage 7, part 1)

**Date:** 2026-11-01
**Stage:** 7 (layouts and partial RoPE)
**Files:** `src/rope/config.rs` (`RopeLayout`, `with_layout`),
`src/rope/embeddings.rs` (`rotate_pairs` with `match`),
`tests/rope_tests.rs`, `tests/numerical_tests.rs`,
`examples/export_reference_cases.rs`, `benches/rope_benchmark.rs`

### Implementation status of this entry

| Item | Status |
|---|---|
| `enum RopeLayout { Interleaved, SplitHalf }` (default `Interleaved`) | ✅ Implemented |
| `RopeConfig::with_layout` | ✅ Implemented |
| Split-half rotation for vectors and `[B, S, H, D]` tensors | ✅ Implemented |
| PyTorch comparison of both layouts | ✅ All cases pass (Entry 15, section 3.6) |
| Automatic conversion of weights between layouts | ❌ Not implemented (a test shows the permutation) |

---

## 1. What did we build?

A configuration choice between the two common pair layouts, and the
split-half rotation. The interleaved code of Stage 4 did not change.

## 2. Why did we build it?

Different model families store the pairs differently. To compare with (or
load weights from) a model, we must use its layout. Entry 10, section 3.8,
explained the layouts; now we implement both.

## 3. Mathematical foundation

### 3.1 The interleaved layout (unchanged)

For `x = [x0, x1, x2, x3, x4, x5]` (`d = 6`), the pairs are `(x0, x1)`,
`(x2, x3)`, `(x4, x5)`. Pair `j` = `(x₂ⱼ, x₂ⱼ₊₁)`, angle `φⱼ = m·θⱼ`.
The existing code uses `chunks_exact_mut(2)` for this.

### 3.2 The split-half layout

Divide the rotary part into two halves of `d/2` values:

```text
x = [ x0  x1  x2 | x3  x4  x5 ]
      first half   second half
pairs: (x0, x3), (x1, x4), (x2, x5)      → pair j = (xⱼ, x_(j + d/2))
```

For pair `j`, the rotation is the same 2-D rotation as always, applied to
the two components `xⱼ` and `x_(j + d/2)`:

```text
x'ⱼ         = xⱼ · cos φⱼ − x_(j + d/2) · sin φⱼ
x'_(j + d/2) = xⱼ · sin φⱼ + x_(j + d/2) · cos φⱼ
```

**Step by step:**

1. Select the "first" coordinate `xⱼ` from the first half and the
   "second" coordinate `x_(j + d/2)` from the second half.
2. Treat `(xⱼ, x_(j + d/2))` as a point in a plane.
3. Apply `R(φⱼ)`: the first coordinate gets `cos·first − sin·second`, the
   second coordinate gets `sin·first + cos·second`.

**Why it is still a collection of 2-D rotations.** Every component belongs
to exactly one pair, and no pair shares a component with another pair. Thus
the full transformation is a block-diagonal rotation, only in a different
**ordering** of the coordinates. Formally, with a permutation matrix `P`
that moves `xⱼ` to slot `2j` and `x_(j + d/2)` to slot `2j + 1`:

```text
RoPE_split(x) = Pᵀ · RoPE_interleaved(P·x)
```

`P` is orthogonal (`PᵀP = I`), so all properties (length preservation, the
relative-position identity) are the same. The test
`layouts_are_equivalent_after_a_permutation` checks this bit for bit with
`P = [0, 4, 1, 5, 2, 6, 3, 7]` for `d = 8`.

### 3.3 The "rotate_half" form

Many PyTorch implementations write the split-half rotation without pairs:

```text
x' = x ⊙ [cos φ, cos φ] + rotate_half(x) ⊙ [sin φ, sin φ]
rotate_half([first, second]) = [−second, first]
```

Check component `j < d/2`: `xⱼ cos φⱼ + (−x_(j+d/2)) sin φⱼ` — the first
equation. Component `j + d/2`: `x_(j+d/2) cos φⱼ + xⱼ sin φⱼ` — the second
equation. Our PyTorch reference uses this form; the Rust code uses explicit
pairs. Two different algorithms, the same mathematics.

### 3.4 The data layouts differ, the mathematics does not

| | Interleaved | Split-half |
|---|---|---|
| Pair `j` | `(x₂ⱼ, x₂ⱼ₊₁)` | `(xⱼ, x_(j+d/2))` |
| Distance between partners | 1 | `d/2` |
| Frequency of pair `j` | `θⱼ = b^(−2j/d)` | `θⱼ = b^(−2j/d)` (the same) |
| Frequency along the vector | `θ₀ θ₀ θ₁ θ₁ θ₂ θ₂ …` | `θ₀ θ₁ θ₂ … θ₀ θ₁ θ₂ …` |
| Used by | RoFormer, original LLaMA code (complex view) | Hugging Face Llama/GPT-NeoX "rotate_half" |

**Why the conventions must match.** The weights `W_Q`, `W_K` of a trained
model produce features in a fixed order. If the model was trained with
split-half, its feature `j` and feature `j + d/2` were trained as partners.
Interleaved RoPE would rotate feature `j` with feature `j + 1` instead: a
different function. The scores would be wrong, although every RoPE property
still holds. (The Hugging Face conversion scripts for LLaMA permute the
rows of `W_Q` and `W_K` for exactly this reason.)

### 3.5 Numerical example: split-half, `d = 6`, `m = 1`, `b = 10000`

`x = [1, 2, 3, 4, 5, 6]`. Frequencies: `θⱼ = 10000^(−2j/6)`:

```text
θ₀ = 10000^0      = 1
θ₁ = 10000^(−1/3) = 1 / 21.5443469 = 0.0464159
θ₂ = 10000^(−2/3) = 1 / 464.158883 = 0.0021544
```

Pairs `(x0, x3) = (1, 4)`, `(x1, x4) = (2, 5)`, `(x2, x5) = (3, 6)`.

```text
pair 0: φ = 1          cos = 0.5403023  sin = 0.8414710
        x'0 = 1·0.5403023 − 4·0.8414710 = 0.5403023 − 3.3658839 = −2.8255816
        x'3 = 1·0.8414710 + 4·0.5403023 = 0.8414710 + 2.1612092 =  3.0026802
pair 1: φ = 0.0464159  cos = 0.9989230  sin = 0.0463992
        x'1 = 2·0.9989230 − 5·0.0463992 = 1.9978460 − 0.2319961 =  1.7658498
        x'4 = 2·0.0463992 + 5·0.9989230 = 0.0927984 + 4.9946149 =  5.0874133
pair 2: φ = 0.0021544  cos = 0.9999977  sin = 0.0021544
        x'2 = 3·0.9999977 − 6·0.0021544 = 2.9999930 − 0.0129266 =  2.9870664
        x'5 = 3·0.0021544 + 6·0.9999977 = 0.0064633 + 5.9999861 =  6.0064494
```

```text
x' = [−2.8255816, 1.7658498, 2.9870664, 3.0026802, 5.0874133, 6.0064494]
```

Check pair 0: `2.8255816² + 3.0026802² = 7.9839 + 9.0161 = 17.0000 = 1² + 4²`.

(The test `split_half_numerical_example` checks these values with 9
decimals. Correction note: the first version of that test contained
expected constants that we had not calculated; the test failed, and we
replaced them with values from an independent Python `math` calculation.
The test caught our own mistake.)

## 4. Rust concepts learned

### 4.1 Enums and how Rust represents them

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeLayout {
    #[default]
    Interleaved,
    SplitHalf,
}
```

An `enum` type has a fixed set of variants. A value is exactly one variant.
These variants carry no data, so the compiler stores a `RopeLayout` as a
small integer **discriminant** (1 byte): 0 for `Interleaved`, 1 for
`SplitHalf`. `#[default]` selects the value of `RopeLayout::default()`.

**Why not a string?** `layout: &str = "split_half"` accepts
`"split-half"`, `"SplitHalf"` and `"spilt_half"`; every typing mistake
becomes a run-time error (or a silent wrong branch). With an enum, a wrong
name does not compile, and the compiler checks that every `match` handles
every variant.

### 4.2 Pattern matching

```rust
match self.config.layout() {
    RopeLayout::Interleaved => { /* chunks_exact_mut(2) */ }
    RopeLayout::SplitHalf => { /* split_at_mut(d / 2) */ }
}
```

`match` compares the value with each pattern from top to bottom and runs
the first arm that matches. The match must be **exhaustive**. The compiler
translates this simple match into one comparison (or a jump table). The
layout is the same for every vector, so the processor's branch predictor
predicts it correctly nearly every time.

### 4.3 `split_at_mut`: two mutable halves

```rust
let (first, second) = rotary.split_at_mut(rotary_dimension / 2);
for ((a, b), &theta) in first.iter_mut().zip(second.iter_mut()).zip(&self.inverse_frequencies) {
    let [ra, rb] = rotate_pair([*a, *b], phi);
    *a = ra;
    *b = rb;
}
```

The borrow rules forbid two `&mut` borrows of the same slice. But
`split_at_mut(k)` returns **two** mutable slices, `[0, k)` and `[k, len)`.
The standard library guarantees that they do not overlap, so the compiler
accepts both at the same time. Then `iter_mut()` gives `&mut f64` items;
`*a` reads the value and `*a = ra` writes through the reference.

The equivalent code `rotary[j]` and `rotary[j + d/2]` with indices would
also work, but it has two bounds checks per pair and hides the structure.

### 4.4 Struct update syntax and builder methods

```rust
pub fn with_layout(self, layout: RopeLayout) -> Self {
    RopeConfig { layout, ..self }
}
```

`..self` copies all other fields from `self`. The method takes `self` **by
value** and returns a new value. `RopeConfig` is `Copy`, so the caller's
value stays usable. Calls chain: `RopeConfig::new(64, b)?.with_layout(..)`.

## 5. Memory and execution

- Interleaved: each pair is 16 contiguous bytes; the loop reads and writes
  sequentially.
- Split-half: each pair is two values `8·d/2` bytes apart; the loop walks
  two sequential streams (the first half and the second half) at the same
  time. Both streams are sequential, so both are cache-friendly.
- Both: `d/2` `sin_cos` calls, `d/2` angle multiplications, `2d`
  multiplications and `d` additions per vector; `d` reads and `d` writes.

## 6. Engineering decisions

1. **The layout is part of `RopeConfig`**, not a parameter of each call: a
   model uses one layout everywhere.
2. **The default stays `Interleaved`**, so no earlier behaviour changed
   (all earlier tests pass unchanged).
3. **One loop function (`rotate_pairs`) with a `match`.** The vector and
   tensor APIs, and every later feature, get both layouts automatically.
4. **No layout conversion function.** The test shows the permutation; a
   public function is not needed now.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| A trait `Layout` with two types and generics | No `match` at run time | More types and generic parameters; the branch is predictable anyway |
| Permute to interleaved, rotate, permute back | One rotation loop | Two extra copies per vector |
| The `rotate_half` formula in Rust | Matches PyTorch code | Needs a temporary vector or careful aliasing |
| A string or `bool` parameter | Short | Typing mistakes; `true`/`false` do not say what they mean |

## 8. Performance considerations

Measured (`cargo bench`, release, `[2, 512, 16, 128]`, full rotation, best
of 15): interleaved **8.56 ms**, split-half **9.01 ms** (8.2 vs 8.6 ns per
pair). The difference (about 5 %) is close to the run-to-run variation of
this simple benchmark (median 9.0 vs 9.6 ms). We do not conclude that one
layout is faster. Stage 9 will measure this with a better methodology.

## 9. Practical example

```rust
use rust_rope::rope::config::{RopeConfig, RopeLayout};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::Position;

let rope = RotaryEmbedding::new(RopeConfig::new(6, 10_000.0)?.with_layout(RopeLayout::SplitHalf));
let y = rope.rotate(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Position::new(1)?)?;
// y ≈ [−2.8255816, 1.7658498, 2.9870664, 3.0026802, 5.0874133, 6.0064494]
```

## 10. What we learned

**Mathematics:** both layouts are block rotations of disjoint pairs; they
differ by a fixed permutation `P`; `RoPE_split = Pᵀ·RoPE_inter·P`; the
convention must match the trained weights; the `rotate_half` formula is the
split-half rotation without explicit pairs.

**Rust:** enums with a discriminant; exhaustive `match`; `split_at_mut`
for two non-overlapping mutable slices; `iter_mut` and writes through
`&mut`; struct update syntax `..self`; builder methods on a `Copy` type.

---

# Entry 15 — Partial RoPE, tensor integration and Stage 7 validation (Stage 7, part 2)

**Date:** 2026-11-01
**Stage:** 7
**Files:** `src/rope/config.rs` (`with_head_dimension`, `head_dimension`,
`rotary_dimension`), `src/utils/validation.rs`
(`RotaryDimensionTooLarge`, `validate_head_dimension`),
`src/rope/embeddings.rs`, tests, export, benchmark

### Implementation status of this entry

| Item | Status |
|---|---|
| Partial RoPE: head dimension `D` ≥ rotary dimension `d` | ✅ Implemented |
| Both layouts with partial RoPE | ✅ Implemented |
| Vectors and `[B, S, H, D]` tensors | ✅ Implemented |
| PyTorch comparison of 160 cases (all layouts and partial) | ✅ All pass |
| Rotation of a part other than the first `d` components | ❌ Not supported (not needed) |

---

## 1. What did we build?

Partial RoPE. A configuration has a rotary dimension `d` and a head
dimension `D ≥ d`. RoPE rotates components `0..d` with the selected layout
and copies components `d..D` unchanged:

```text
x' = [ RoPE(x[0..d]),  x[d..D] ]
```

## 2. Why did we build it?

Some models (for example GPT-NeoX, GPT-J and Phi models) rotate only a
fraction of each head (a "rotary percentage", for example 25 % or 50 %).
The unrotated components carry **position-independent** content: their
dot-product contribution `qᵀk` is the same at every distance.

## 3. Mathematical foundation

### 3.1 The equation

For a head vector `x ∈ R^D` and rotary dimension `d`:

```text
x'ₖ = (RoPE_layout(x[0..d], m))ₖ     for 0 ≤ k < d
x'ₖ = xₖ                             for d ≤ k < D
θⱼ  = b^(−2j/d),  j = 0 … d/2 − 1    (d, not D, in the exponent)
```

As a matrix: `R_m^partial = diag(R_m^(d), I_(D−d))`, a block-diagonal
matrix whose last block is the identity.

### 3.2 Why the rules

- **`d` positive and even.** RoPE rotates pairs. `d = 0` rotates nothing
  (that is not RoPE); an odd `d` leaves one rotary component without a
  partner.
- **`d ≤ D`.** RoPE can only rotate components that exist.
- **`D` may be odd** when `d < D`: the pass-through part needs no pairs.

### 3.3 Frequencies use the rotary dimension

With `D = 128`, `d = 64`: `θ₁ = 10000^(−2/64) = 0.7499`, not
`10000^(−2/128) = 0.8660`. The 32 rotary pairs still cover the full speed
range `(1/b, 1]`. This is the convention of the common implementations (the
frequency formula uses the rotary dimension). The test `partial_rope_128_with_64`
checks `θ₁`.

### 3.4 The score with partial RoPE

```text
q'ᵀk' = (q_rotᵀ R((n − m)) k_rot) + (q_passᵀ k_pass)
         └ depends on n − m ┘        └ does not depend on position ┘
```

The relative-position property still holds (the second term depends on no
position at all). Test: `stage_4_invariants_hold_for_every_layout_and_partial_rope`.

### 3.5 Example: `D = 128`, `d = 64`

```text
x = [x0 … x63 | x64 … x127]
     rotary     pass-through (bit-for-bit unchanged)
interleaved: pairs (x0,x1) … (x62,x63)       split-half: pairs (x0,x32) … (x31,x63)
```

### 3.6 Stage 7 PyTorch validation results

`cargo run --example export_reference_cases` now writes **160 cases** (66
from Stage 6, 94 new). `compare_rope.py`: **all 160 pass** in float64 and
float32 (exit code 0).

| Group | Cases (per mode) | float64 max abs | float32 max abs |
|---|---|---|---|
| Split-half, full | 36 | `1.6 × 10⁻¹³` | `5.7 × 10⁻⁵` |
| Split-half, partial | 25 | `7.6 × 10⁻¹⁴` | `6.0 × 10⁻⁴` |
| Interleaved, partial | 25 | `1.6 × 10⁻¹³` | `6.9 × 10⁻⁴` |
| All 160 cases | 160 | `1.9 × 10⁻¹¹` (Stage 6 case `d = 128`, `m = 10⁵`) | `1.1 × 10⁻²` |

The PyTorch reference implements split-half with `rotate_half` (a different
algorithm), and partial RoPE with `torch.cat`. Each case states its layout
and rotary dimension, so the comparison never mixes layouts. The negative
control "wrong layout" (Entry 13) shows that a mix would fail.

## 4. Rust concepts learned

### 4.1 Range indexing and `split_at_mut` for the pass-through part

```rust
let (rotary, _pass_through) = embedding.split_at_mut(rotary_dimension);
```

`split_at_mut(d)` is the mutable form of `(&v[..d], &v[d..])`. We use only
`rotary`. The name `_pass_through` (with a leading `_`) documents the
second part and tells the compiler that we do not use it on purpose. The
code **never touches** `x[d..D]`, so these values stay bit-for-bit unchanged
without a copy.

### 4.2 `dimension()` now means two things — how we kept compatibility

Before Stage 7, `RopeConfig::dimension()` was both the input length and
the `d` of the frequency formula. Now they can differ:

- `RopeConfig::dimension()` and `rotary_dimension()`: `d` (frequencies).
- `RopeConfig::head_dimension()` and `RotaryEmbedding::dimension()`: `D`
  (input length).

`RopeConfig::new(d, b)` sets `D = d`. Thus every earlier call gives the same
result, and all earlier tests pass unchanged.

## 5. Memory and execution

Per head vector, partial RoPE does `d/2` instead of `D/2` `sin_cos` calls,
and reads and writes `d` instead of `D` values. The pass-through values are
not read. But the cache lines that contain them may still be loaded,
because the hardware prefetcher fetches sequential memory, and the next
head's rotary part follows directly after.

## 6. Engineering decisions

1. **`with_head_dimension(D)` on the config**, not a separate "rotary
   percentage". An integer is exact; a percentage can produce an odd or
   rounded `d`.
2. **The rotary part is the first `d` components**, as in the common
   implementations.
3. **Validation in `utils::validation`** with a new error variant that
   contains both values.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| `rotary_fraction: f64` | Matches some configuration files | Rounding; odd results possible |
| Caller passes a sub-slice `&mut v[..d]` | No new configuration | Tensors cannot do this per head; error-prone |
| Copy `x[d..]` into the output explicitly | Explicit | Useless work in the in-place path |

## 8. Performance considerations

Measured (`[2, 512, 16, 128]`, in place, best of 15, release):

| Layout | `d = 128` (full) | `d = 64` (partial) | ratio |
|---|---|---|---|
| Interleaved | 8.56 ms | 7.26 ms | 0.85 |
| Split-half | 9.01 ms | 8.21 ms | 0.91 |

Partial RoPE does half of the trigonometry and half of the writes, but it
was only 9 % to 15 % faster here, not 2× faster. **We did not measure the
reason.** A possible reason is memory traffic: the tensor is 16 MiB, larger
than the processor caches, and the prefetcher may still load the
pass-through cache lines. Stage 9 (CPU performance, planned) must
investigate this before any optimization.

## 9. Practical example

```rust
use rust_rope::rope::config::{RopeConfig, RopeLayout};
use rust_rope::rope::embeddings::RotaryEmbedding;
use rust_rope::rope::position::Position;

let config = RopeConfig::new(4, 10_000.0)?
    .with_layout(RopeLayout::Interleaved)
    .with_head_dimension(8)?;
let rope = RotaryEmbedding::new(config);
let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let y = rope.rotate(&x, Position::new(1)?)?;
assert_eq!(&y[4..], &x[4..]); // components 4..8 unchanged
assert_ne!(&y[..4], &x[..4]); // components 0..4 rotated
```

## 10. Test results and what we learned

**Rust tests:** all pass. Stage 7 added 3 unit tests (`config.rs`,
`validation.rs`), 11 API tests in `rope_tests.rs` (adjacent pairing,
split-half pairing, split-half example, pair lengths, permutation
equivalence, partial rotation, `128/64`, `d = D`, invalid rotary dimensions,
tensors with layouts and partial, tensor head-dimension check), and 3
numerical tests (all Stage 4 invariants for every layout and partial
configuration; split-half angles; equal frequencies in both layouts).
**Python:** 12 self-tests pass; 160 comparison cases pass.

**Mathematics:** partial RoPE is `diag(R_m^(d), I)`; the frequencies use
`d`; the pass-through components add a position-independent term to the
score; the relative property still holds.

**Rust:** `split_at_mut` to protect a pass-through region; compatible API
evolution (new fields with defaults that preserve old behaviour).


---

# Stage 8 — RoPE Scaling and Long Context

# Entry 16 — Position interpolation, linear scaling, NTK-aware scaling and YaRN (Stage 8)

**Date:** 2026-11-01
**Stage:** 8 (RoPE scaling methods)
**Files:** `src/rope/scaling.rs` (new content), `src/rope/embeddings.rs`
(`with_scaling`, `scaling`, `attention_factor`), `src/rope/frequencies.rs`
(`inverse_frequency_unchecked` is now `pub(crate)`),
`src/utils/validation.rs` (4 new errors, 3 new checks),
`reference/pytorch_rope.py` (`scaling_parameters`),
`reference/compare_rope.py`, `reference/test_pytorch_rope.py`,
`examples/export_reference_cases.rs`, tests, `docs/references.md`

### Implementation status of this entry

| Item | Status |
|---|---|
| `RopeScaling::None` | ✅ Implemented (bit-identical to standard RoPE) |
| `RopeScaling::PositionInterpolation { original_context, target_context }` | ✅ Implemented |
| `RopeScaling::Linear { factor }` | ✅ Implemented |
| `RopeScaling::NtkAware { factor }` (static NTK-aware, `b' = b·s^(d/(d−2))`) | ✅ Implemented |
| `RopeScaling::Yarn { factor, original_context, alpha, beta }` (paper formulation) | ✅ Implemented |
| PyTorch comparison: 128 new scaling cases (288 total) | ✅ All pass |
| Dynamic NTK (factor depends on the current length) | ❌ Not implemented |
| The Hugging Face discretization of YaRN (`floor`/`ceil` ramp over dimension indices, `truncate`) | ❌ Not implemented (section 3.7) |
| LongRoPE, per-pair searched factors | ❌ Not implemented |
| Fine-tuning (all methods are normally followed by a short fine-tuning) | ❌ Outside the scope: this project has no model |

---

## 1. What did we build?

A scaling module. Each method changes the inverse frequencies `θⱼ` into
`θ'ⱼ` (and YaRN adds one constant factor). `RotaryEmbedding::with_scaling`
calculates them one time. The rotation loop did not change.

## 2. Why did we build it?

### 2.1 Why long-context extension is needed

A model is trained with sequences of at most `L` tokens (the **original
context length**, for example 2048 or 4096). We often want to use it with
`L' > L` tokens (the **target context length**). Using positions `m ≥ L`
without any change is called **extrapolation**.

### 2.2 The limitation of standard RoPE

Standard RoPE accepts any position. The mathematics does not break at
`m = L`. The **model** breaks: during training, each pair `j` only saw the
angles `φ(m, j) = m·θⱼ` for `0 ≤ m < L`.

Example, `d = 8`, `b = 10000`, `θ = [1, 0.1, 0.01, 0.001]`, `L = 2048`:

| Pair `j` | `θⱼ` | wavelength `2π/θⱼ` | largest trained angle `2047·θⱼ` | angle at `m = 8191` | new? |
|---|---|---|---|---|---|
| 0 | 1 | 6.28 | 2047 rad (326 turns) | 8191 rad | no — every direction was seen |
| 1 | 0.1 | 62.8 | 204.7 rad (32.6 turns) | 819.1 rad | no — every direction was seen |
| 2 | 0.01 | 628 | 20.47 rad (3.26 turns) | 81.91 rad | partly |
| 3 | 0.001 | 6283 | **2.047 rad (0.33 turn)** | **8.191 rad** | **yes** — never seen |

**Geometric meaning.** A **high-frequency** pair (large `θⱼ`, short
wavelength) turns many times inside `L`. All its directions on the circle
were seen in training; it encodes **local** distance. A **low-frequency**
pair (small `θⱼ`, long wavelength) turns less than once inside `L`. Its
direction acts like a "coarse clock" for long distances. Beyond `L`, it
points into directions that the model **never saw**. Attention scores
become unreliable. This is why extending the context is **not** only
"increase a configuration value".

### 2.3 Angles at small and large positions (`d = 8`, `b = 10000`)

| `m` | `j = 0` | `j = 1` | `j = 2` | `j = 3` |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0.1 | 0.01 | 0.001 |
| 2 | 2 | 0.2 | 0.02 | 0.002 |
| 3 | 3 | 0.3 | 0.03 | 0.003 |
| 2047 | 2047 | 204.7 | 20.47 | 2.047 |
| 100 000 | 100 000 | 10 000 | 1000 | 100 |

## 3. Mathematical foundation

### 3.1 The baseline (unchanged)

```text
θⱼ = 1 / b^(2j/d)          (j = pair index, d = rotary dimension, b = base)
φ(m, j) = m · θⱼ           (m = token position)
```

All scaling methods below use the symbols:

| Symbol | Meaning |
|---|---|
| `L` | original (training) context length |
| `L'` | target context length |
| `s = L'/L` | scaling factor, `s ≥ 1` |
| `θ'ⱼ` | scaled inverse frequency |
| `φ'(m, j) = m·θ'ⱼ` | scaled angle |

### 3.2 Position interpolation (Chen et al., 2023)

**Idea:** do not extrapolate. **Compress** the new positions into the
trained range.

```text
scale     = L / L'                  (= 1/s ≤ 1)
m_scaled  = m · scale
φ'(m, j)  = m_scaled · θⱼ
```

**Derivation of the property.** The largest new position `L' − 1` maps to
`(L' − 1)·L/L' = L − L/L' < L`. Thus every scaled position is inside
`[0, L)`, the trained range, for every pair.

**Example** (`L = 2048`, `L' = 8192`, `m = 4096`):

```text
scale    = 2048 / 8192 = 0.25
m_scaled = 4096 · 0.25 = 1024
angles without scaling: 4096·θ = [4096, 409.6, 40.96, 4.096]
angles with scaling:    1024·θ = [1024, 102.4, 10.24, 1.024]
```

The token at position 4096 is rotated exactly like the token at position
1024 without scaling. The last position 8191 maps to `2047.75`, inside
`[0, 2048)`. (Test: `interpolated_angles_stay_in_the_trained_range`.)

**Trade-off.** Interpolation divides **every** frequency by `s`. Two
neighbouring tokens now differ by `θⱼ/s` instead of `θⱼ`. For the fast
pairs, which encode local word order, the resolution drops by the factor
`s` (here 4). The scaled positions are also non-integers (`1024.25`,
`1024.5`, …), which the model never saw. Thus position interpolation needs
fine-tuning, and it can lose precision for short-range relations.

### 3.3 Linear scaling: positions versus frequencies

```text
scaling positions:    φ' = (m · (1/s)) · θⱼ
scaling frequencies:  φ' = m · (θⱼ / s)
```

Multiplication is associative and commutative: `(m·(1/s))·θ = m·(θ/s)`.
Thus both give **the same angle**. Many implementations call the frequency
form "linear" scaling with a `factor = s`. In floating point, the two forms
can differ by one rounding; for `s = 4` both are exact, because division by
a power of two is exact (test `position_interpolation_equals_linear_scaling`
in `scaling.rs` checks bit equality).

**Our implementation.** Both variants exist in the API, so the method is
explicit:

- `PositionInterpolation { original_context, target_context }` — the
  context-length form; it checks `1 ≤ L ≤ L'`.
- `Linear { factor }` — the factor form; it checks `s ≥ 1`.

Both are implemented by the **same** frequency operation (`θ'ⱼ = θⱼ·L/L'`
or `θⱼ/s`), because positions are integers in this library (`Position`).
This is not duplicated logic: it is one transformation with two
parameterizations. The PyTorch reference implements position
interpolation by scaling the **positions** (`position_ids * L/L'`), so the
comparison checks the equivalence independently.

### 3.4 NTK-aware scaling (bloc97, 2023)

**The Neural Tangent Kernel at a high level.** The Neural Tangent Kernel
(NTK, Jacot et al., 2018) describes how a very wide neural network changes
during training. One result (with Tancik et al., 2020, "Fourier features"):
networks learn high-frequency details poorly from low-dimensional inputs
unless the input already contains high-frequency features. For RoPE this
suggests: **keep the high frequencies**; they carry fine detail that the
network cannot easily re-learn. Position interpolation destroys exactly
those. The name "NTK-aware" refers to this argument. It is a heuristic, not
a theorem.

**Idea:** instead of dividing every `θⱼ` by `s`, change the **base**. This
changes the frequencies unevenly: high frequencies almost stay, low
frequencies are interpolated.

**Derivation.** We want:

1. the fastest pair unchanged: `θ'₀ = θ₀ = 1` (automatic: `b'^0 = 1`);
2. the slowest pair fully interpolated: `θ'_(d/2−1) = θ_(d/2−1) / s`.

With `θ'ⱼ = b'^(−2j/d)` and `j = d/2 − 1` (exponent `(d − 2)/d`):

```text
b'^(−(d−2)/d) = b^(−(d−2)/d) / s
b'^((d−2)/d)  = b^((d−2)/d) · s                    (take the reciprocal)
b'            = b · s^(d/(d−2))                     (raise to the power d/(d−2))
```

This is the formula of the original post ("NTK-Aware Scaled RoPE", bloc97,
2023) and of the YaRN paper (section 3.1). It needs `d ≥ 4`: for `d = 2`,
`d − 2 = 0`.

**How the frequencies change.** `θ'ⱼ/θⱼ = (b'/b)^(−2j/d) = s^(−2j/(d−2))`.
This goes from `1` (`j = 0`) to `1/s` (`j = d/2 − 1`) geometrically.

**Example** (`d = 8`, `b = 10000`, `s = 4`):

```text
b' = 10000 · 4^(8/6) = 10000 · 6.3496 = 63 496.04
θ' = [1, 0.0630, 0.003969, 0.00025]
θ'/θ = [1, 0.630, 0.397, 0.25]          (= 4^(−2j/6))
```

**Difference to position interpolation:**

| | Position interpolation | NTK-aware |
|---|---|---|
| Fast pair (`j = 0`) | `÷ s` (resolution lost) | unchanged |
| Slow pair (`j = d/2 − 1`) | `÷ s` | `÷ s` |
| Middle pairs | `÷ s` | `÷ s^(2j/(d−2))` |
| What changes | the positions (or all θ) | the base |

**Variants we did not implement.** "Dynamic NTK" recomputes `s` from the
current sequence length during inference. Other variants use different
exponents. Our `NtkAware` is the static formula above only.

### 3.5 YaRN (Peng et al., 2023)

**Why simple interpolation reduces positional resolution:** section 3.2.
**Why NTK-aware is not ideal either:** its ratio `θ'ⱼ/θⱼ` is a smooth
geometric curve that ignores how many turns a pair makes in the training
context. Some middle pairs are interpolated although they turned many times
in `L` (no extrapolation problem), and the slowest pairs can still be
slightly extrapolated.

YaRN ("Yet another RoPE extensioN method") uses the **number of turns in
the original context** to decide, per pair:

```text
λⱼ   = 2π / θⱼ                      wavelength (positions per turn)
r(j) = L / λⱼ                       turns of pair j inside the training context
```

- `r(j) > β`: the pair turned many times in training. **Do not
  interpolate** (keep `θⱼ`): high-frequency band.
- `r(j) < α`: the pair turned less than `α` times. **Interpolate fully**
  (`θⱼ / s`): low-frequency band.
- in between: blend linearly.

**The correction range** is `[α, β]` in units of "turns in `L`". The paper
uses `α = 1`, `β = 32` for LLaMA models.

**The blending (ramp) function:**

```text
γ(r) = 0                    if r < α
     = 1                    if r > β
     = (r − α) / (β − α)    otherwise

θ'ⱼ = (1 − γ(r(j))) · θⱼ / s  +  γ(r(j)) · θⱼ
```

`γ = 1` keeps the frequency; `γ = 0` interpolates it; values in between
mix the two.

**The attention scaling factor.** The paper observes that interpolation
makes the attention softmax distribution flatter (lower "temperature"
effect on the logits). YaRN multiplies the logits by `1/t` with:

```text
√(1/t) = 0.1 · ln(s) + 1
```

Implementations multiply **both** the rotated query and the rotated key by
`√(1/t)` (equivalently: multiply `cos φ` and `sin φ` by it), so the score
`qᵀk` is multiplied by `1/t`. Our `RotaryEmbedding` does exactly this: it
multiplies the rotated components by `attention_factor()`. Thus a YaRN
rotation is **no longer length-preserving**: lengths grow by the factor
`0.1·ln(s) + 1`. The pass-through components of partial RoPE are not
multiplied (the same as scaling `cos` and `sin`).

**Example** (`d = 8`, `b = 10000`, `L = 2048`, `s = 4`, `α = 1`, `β = 32`):

| `j` | `θⱼ` | `λⱼ` | `r(j) = 2048/λⱼ` | `γ` | `θ'ⱼ` | `θ'ⱼ/θⱼ` |
|---|---|---|---|---|---|---|
| 0 | 1 | 6.28 | 325.9 | 1 | 1 | 1 |
| 1 | 0.1 | 62.8 | 32.59 | 1 | 0.1 | 1 |
| 2 | 0.01 | 628.3 | 3.259 | `(3.259 − 1)/31 = 0.0729` | `0.9271·0.0025 + 0.0729·0.01 = 0.003047` | 0.305 |
| 3 | 0.001 | 6283 | 0.326 | 0 | 0.00025 | 0.25 |

Attention factor: `0.1·ln 4 + 1 = 1.1386`; the score factor `1/t` is
`1.1386² = 1.2965`.

Compare with NTK-aware for the same case: pair 1 is `÷1.59` under NTK, but
unchanged under YaRN, because it turns 32.6 times in `L`.

### 3.6 Which formulation we implement, and why

We implement the formulas of the **YaRN paper** (Peng et al., 2023,
sections 3.2 "NTK-by-parts" and 3.4 "YaRN"): a continuous ramp `γ(r)` over
the number of turns `r`, and `√(1/t) = 0.1·ln(s) + 1`. These formulas are
clearly defined and need only `s`, `L`, `α`, `β`.

### 3.7 A known difference: the Hugging Face / reference-code discretization

The published code (and Hugging Face `transformers`) computes the ramp in a
different way. It converts the boundaries `β` ("beta_fast") and `α`
("beta_slow") into **dimension indices**, rounds them with `floor` and
`ceil`, and ramps linearly over the dimension index. This gives similar but
**not identical** frequencies for the pairs inside the ramp. It also has
options (`truncate`, `mscale`, `mscale_all_dim`) that some models use.
**Our implementation does not reproduce these details**, and we do not
claim equality with the Hugging Face output. Our PyTorch reference
implements the paper's formulation independently.

### 3.8 Numerical precision

- All methods change `θⱼ` once; the angle error model of Entry 13 is
  unchanged: about `ε·(1 + m·θ'₀)`. Scaling makes the angles **smaller**
  (`θ'ⱼ ≤ θⱼ`), so it does not increase the error.
- NTK-aware: `s^(d/(d−2))` and `b'^(−2j/d)` add one or two roundings to
  `θ'ⱼ` (relative `≈ 2ε`).
- YaRN: the ramp introduces `ln`, divisions and a blend; all relative
  `≈ few ε`.
- Measured agreement with PyTorch (float64): at most `2.2 × 10⁻¹²`
  (section 9).

## 4. Rust concepts learned

### 4.1 An enum with data for the method

```rust
pub enum RopeScaling {
    None,
    PositionInterpolation { original_context: usize, target_context: usize },
    Linear { factor: f64 },
    NtkAware { factor: f64 },
    Yarn { factor: f64, original_context: usize, alpha: f64, beta: f64 },
}
```

Each variant carries exactly the parameters that its method needs. A
`Linear` value cannot have an `alpha`; a `Yarn` value cannot be created
without `original_context`. A struct with optional fields
(`factor: Option<f64>, alpha: Option<f64>, …`) would allow meaningless
combinations. In memory, the enum is as large as its largest variant
(`Yarn`: 32 bytes) plus a tag.

### 4.2 `match *self` and binding fields

```rust
match *scaling {
    RopeScaling::Linear { factor } => theta.iter_mut().for_each(|t| *t /= factor),
    RopeScaling::Yarn { factor, original_context, alpha, beta } => { /* ... */ }
    // ...
}
```

The pattern `Linear { factor }` **binds** the field to a local variable
`factor`. `*scaling` dereferences the reference; this is allowed because
the enum is `Copy`. `RopeScaling::Yarn { factor, .. }` in
`attention_factor` ignores the other fields with `..`; `_ => 1.0` matches
all remaining variants.

### 4.3 Validation per variant

`RopeScaling::validate(&self, &RopeConfig)` checks each variant with the
functions in `utils::validation`. `with_scaling` calls it before it
calculates anything (`?`), so an invalid method never produces a
`RotaryEmbedding`.

### 4.4 `iter_mut().for_each` and in-place frequency changes

`theta.iter_mut().for_each(|t| *t *= scale)` changes every element of the
`Vec` in place. `t` is `&mut f64`; `*t *= scale` writes through it. No new
`Vec` is allocated: `scaled_inverse_frequencies` allocates exactly one
`Vec` (the same as the unscaled path).

### 4.5 Keeping scaling out of the rotation loop

The only change to the hot loop is `pair[0] = ra * factor`. For every
method except YaRN, `factor = 1.0`, and IEEE 754 guarantees `x · 1.0 = x`
exactly. Thus `RopeScaling::None` gives **bit-identical** output to Stage 7
(test `scaling_none_gives_bit_identical_standard_rope`). The cost is two
multiplications per pair (not measured separately; Stage 9).

## 5. Memory and execution

- `RotaryEmbedding` grows by the `RopeScaling` value (40 B with the tag)
  and one `f64`.
- `with_scaling`: one allocation of `d/2` values; YaRN does `d/2`
  divisions, one `ln`, and `d/2` ramp evaluations; NTK does one `powf` for
  `b'` and `d/2` `powf` for `θ'ⱼ`. All of it happens one time.
- Rotation: unchanged, plus 2 multiplications per pair.

## 6. Engineering decisions

1. **Scaling is a property of `RotaryEmbedding`, not of `RopeConfig`.**
   `RopeConfig` defines the geometry (dimensions, base, layout); scaling is
   a modification for deployment. `RotaryEmbedding::new(config)` keeps the
   old behaviour.
2. **Everything becomes frequencies.** One code path for all methods; the
   rotation loop stays simple.
3. **Integer positions stay integers**: position interpolation is applied
   to the frequencies, which is mathematically identical.
4. **Explicit variants instead of one "factor" field**, and two variants
   for the same linear transformation, because the task asked for an
   explicit API and because the context-length form validates `L ≤ L'`.
5. **The YaRN paper formulation**, documented as different from the
   Hugging Face discretization.
6. **`s ≥ 1` everywhere.** `s < 1` is not a context extension.

## 7. Alternative approaches

| Alternative | Advantage | Disadvantage |
|---|---|---|
| Scaling inside `RopeConfig` | One object | Mixes model geometry with deployment changes |
| A trait `ScalingMethod` with one type per method | Extensible by users | Generics or `dyn` in `RotaryEmbedding`; more code; a closed enum is sufficient |
| `f64` positions for position interpolation | Literal formula | Loses the integer guarantees of `Position` |
| The Hugging Face YaRN discretization | Numerically equal to HF models | More parameters and rounding rules; not the paper's formula |
| Dynamic NTK | Better without fine-tuning for short inputs | The frequencies depend on the input length; a cache per length |

## 8. Performance considerations

- The scaling cost is paid one time in `with_scaling` (`O(d)`).
- The rotation cost does not depend on the method (2 extra multiplications
  per pair for all methods).
- No allocation in the rotation path.

## 9. Test results

**Rust** (`cargo test`, 2026-11-01): **all 298 tests pass** (171 unit,
43 numerical, 58 API, 26 doc tests). New Stage 8 tests: 6 unit tests in
`scaling.rs` and 1 in `validation.rs`; 8 API tests (none = bit-identical,
factor 1 = standard, position-interpolation angles, longer targets give
slower frequencies, methods are distinguishable, invalid configurations,
both layouts + partial + tensors, YaRN factor only on the rotary part); 5
numerical tests (PI = standard RoPE at `m/4`, angles inside the trained
range, relative property with scaling, NTK for `d = 4…128`, stability up
to `m = 10⁷` with `s = 64`).

**Python** (`unittest`): **16 tests pass**, including the negative control
"the comparison detects a missing scaling method" for every scaled case
with `m ≥ 4096`.

**PyTorch comparison:** **all 288 cases pass** in float64 and float32
(128 new scaling cases: 3 dimensions × 4 positions + 4 tensor cases with
both layouts and partial RoPE, for 8 method settings).

| Method | float64 max abs | float32 max abs |
|---|---|---|
| PositionInterpolation 2048→8192 | `2.5 × 10⁻¹³` | `2.5 × 10⁻⁴` |
| Linear `s = 4` | `2.5 × 10⁻¹³` | `2.5 × 10⁻⁴` |
| Linear `s = 2.5` | `4.9 × 10⁻¹³` | `3.8 × 10⁻⁴` |
| Linear `s = 1` | `9.9 × 10⁻¹³` | `1.3 × 10⁻³` |
| NtkAware `s = 4` | `2.2 × 10⁻¹²` | `1.2 × 10⁻³` |
| NtkAware `s = 8` | `1.2 × 10⁻¹²` | `8.6 × 10⁻⁴` |
| YaRN `s = 4`, `L = 2048` | `1.1 × 10⁻¹²` | `1.5 × 10⁻³` |
| YaRN `s = 16`, `L = 4096` | `1.3 × 10⁻¹²` | `1.7 × 10⁻³` |

Position interpolation and `Linear s = 4` give identical error statistics:
the PyTorch reference scales the positions, Rust scales the frequencies,
and both agree with the Rust output to the same digits.

**Why relative error near zero can mislead:** see Entry 13, section 3.2.
The comparison computes it only for values above `10⁻³ · max|p|`.

## 10. Known limitations and what we learned

**Limitations:** no fine-tuning (scaling methods are designed to be
followed by a short fine-tuning; this project only implements the
transformation); no dynamic NTK; YaRN is the paper formulation, not the
Hugging Face discretization; no LongRoPE.

**Mathematics:** extrapolation fails for the low-frequency pairs, whose
angles leave the trained range; position interpolation compresses every
position into `[0, L)` at the cost of local resolution;
`(m/s)·θ = m·(θ/s)`; NTK-aware scaling changes the base to
`b·s^(d/(d−2))`, which keeps `θ₀` and divides the slowest pair by `s`;
YaRN decides per pair from `r = L/λ` with a ramp `γ` and adds the
attention factor `0.1·ln s + 1`.

**Rust:** an enum whose variants carry exactly their own parameters;
`match` with field binding and `..`; validation per variant; in-place
changes with `iter_mut`; keeping a feature out of the hot loop.

---

## Roadmap status after Entry 16

The roadmap tables after Entries 6 and 9 show the state at those times.
This table shows the current state, with the redefined stage numbers.

| Stage | Status |
|---|---|
| 1 Project foundation | ✅ Done (Entry 1) |
| 2 Mathematical foundations | ✅ Done (Entries 2–6) |
| 3 Frequencies, positions, angles | ✅ Done (Entries 7–9) |
| 4 RoPE transformation of a vector | ✅ Done (Entry 10) |
| 5 Query and key tensors | ✅ Done (Entries 11–12) |
| 6 PyTorch reference validation | ✅ Done (Entry 13) |
| 7 Layouts and partial RoPE | ✅ Done (Entries 14–15) |
| 8 RoPE scaling | ✅ Done (Entry 16) — review pending |
| 9 CPU performance engineering | ⏳ Planned — waits for approval |
