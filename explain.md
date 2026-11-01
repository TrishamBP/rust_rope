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
