# CLAUDE.md — Permanent Project Instructions for `rust_rope`

Read this file before you start every task in this project.
Follow these instructions in every task.
These instructions are permanent. They apply to all stages of the project.

---

## 0. Project purpose

`rust_rope` implements Rotary Position Embeddings (RoPE) from scratch in Rust.

This is an educational engineering project. It has two equal objectives:

1. **Learn Rust.** Learn ownership, borrowing, lifetimes, memory allocation,
   data structures, traits, generics, iterators, error handling, performance
   and low-level systems programming.
2. **Learn RoPE mathematics.** Learn the equations, the derivations, the
   geometric interpretation, the trigonometry, the linear algebra and the
   use of RoPE inside Transformer architectures.

The implementation is the vehicle for learning. It is not the only goal.

- Do not optimize for the smallest amount of code.
- Optimize for learning, mathematical understanding, clean engineering and
  performance awareness.

---

## 1. Language and explanation standard: ASD-STE100

All explanations must follow **ASD-STE100 (Simplified Technical English)**.
This rule is mandatory. It applies to chat answers, `explain.md`, `docs/`,
the README and code comments.

Apply these rules:

- Use short, clear sentences. Try to keep a sentence below 20 words.
- Explain one concept at a time. Use one paragraph for one topic.
- Do not use complex vocabulary when a simple word is sufficient.
- Define a technical term before you use it.
- Do not use an abbreviation before you write its full form.
  Example: "Rotary Position Embedding (RoPE)".
- Use the active voice. Write "The function allocates a vector."
  Do not write "A vector is allocated by the function."
- Use simple sentence structures.
- Explain why something happens, not only what happens.
- Give practical examples.
- Explain mathematical notation before you use it.
- Do not assume that the learner already knows a Rust concept or a
  mathematics concept.

The objective: make complex systems programming and mathematics easy to
understand. Do not reduce technical accuracy to make text simpler.

---

## 2. Dual learning approach

Every implementation task must teach Rust **and** mathematics.

### A. Rust programming explanations

When you implement a Rust feature, answer these questions:

1. What Rust feature do we use?
2. Why do we use it?
3. What problem does it solve?
4. How does the Rust compiler interpret it?
5. How does ownership work in this code?
6. Does the code allocate memory on the heap, or does it use the stack?
7. Do we borrow values, or do we move values?
8. What happens when values go out of scope?
9. Could we implement this in a different way?
10. What are the performance implications?

Explain each relevant Rust concept with a small, independent example.

### B. Mathematical explanations

When you implement a RoPE feature, answer these questions:

1. What mathematical concept do we implement?
2. Why is this mathematical operation necessary?
3. What is the equation?
4. What does each variable represent?
5. How do we derive the equation?
6. What is the geometric interpretation?
7. How does the mathematics translate into Rust code?
8. What happens in a small numerical example?
9. What are the numerical precision considerations?
10. Where does a Transformer use this operation?

Do not skip mathematical derivations.

**Example of the expected depth.** When you implement a rotation, explain the
two-dimensional rotation matrix:

```text
R(θ) = [[cos(θ), -sin(θ)],
        [sin(θ),  cos(θ)]]
```

Explain how `R(θ)` rotates a vector `(x, y)` by the angle `θ`.
Explain how RoPE applies this rotation to each pair of embedding dimensions,
with a different angle for each pair.

---

## 3. Mandatory explanation format for `explain.md`

After you complete **every** implementation milestone, update `explain.md`.
Use this structure for each new entry:

1. **What did we build?** — Describe the feature.
2. **Why did we build it?** — Explain its role in the RoPE implementation.
3. **Mathematical foundation** — Equations, derivations, numerical examples.
4. **Rust concepts learned** — Rust features, and why we selected them.
5. **Memory and execution** — Ownership, borrowing, allocation, data layout,
   execution flow.
6. **Engineering decisions** — Why we selected this implementation.
7. **Alternative approaches** — Other implementations and their trade-offs.
8. **Performance considerations** — Complexity, memory use, possible
   optimizations.
9. **Practical example** — A small example with the expected output.
10. **What we learned** — A summary of the Rust and mathematics concepts.

Rules for `explain.md`:

- Do not write superficial explanations. `explain.md` is a learning resource
  that we read again later.
- **Never delete or overwrite previous explanations.** Append new entries.
  If an old entry becomes incorrect, add a dated correction note. Do not
  remove the old text.
- Number each entry (Entry 1, Entry 2, ...) and give it a date.
- Clearly mark what is **implemented** and what is **planned**.

---

## 4. Architecture rules

The module responsibilities are fixed:

| Module | Responsibility |
|---|---|
| `src/rope/config.rs` | RoPE configuration |
| `src/rope/frequencies.rs` | Positional rotation frequencies |
| `src/rope/rotations.rs` | Vector rotation operations |
| `src/rope/embeddings.rs` | Applying RoPE to embeddings |
| `src/rope/position.rs` | Position indexing |
| `src/rope/scaling.rs` | RoPE scaling methods |
| `src/math/complex.rs` | Complex number operations |
| `src/math/trigonometry.rs` | Sine and cosine operations |
| `src/math/matrix.rs` | Matrix and vector operations |
| `src/utils/validation.rs` | Input and configuration validation |

Rules:

- Keep general mathematical primitives in `src/math/`.
  `src/math/` must **not** depend on `src/rope/`.
- Keep RoPE-specific logic in `src/rope/`. It can use `src/math/`.
- `src/main.rs` is a thin demonstration binary. Put all logic in the library
  (`src/lib.rs` and its modules), so that tests, examples and benchmarks can
  use it.
- Do not add unnecessary abstractions. Add a trait or a generic only when a
  real need exists. Explain the need in `explain.md`.
- Implement only the current stage. Do not add speculative features.

---

## 5. Dependency rules

- Prefer the Rust standard library.
- Do not add an external dependency without a written explanation in
  `explain.md`. The explanation must state: the purpose, why the standard
  library is not sufficient, the alternatives, and the cost (compile time,
  binary size, trust).
- Do not add GPU dependencies.
- Development-only tools (for example, a benchmark framework) go in
  `[dev-dependencies]`, not in `[dependencies]`.

---

## 6. Code conventions

- Rust edition: 2024 (see `Cargo.toml`).
- Format all code with `cargo fmt`.
- Code must pass `cargo clippy --all-targets` without warnings, when possible.
- Every public item has a documentation comment (`///`).
  Every module has a module comment (`//!`).
- Document the mathematics next to the code that implements it.
  Use the same variable names in the code and in the equations.
- Return `Result` for errors that a caller can cause (for example, an odd
  embedding dimension). Do not use `panic!` for those errors.
  Use `panic!` only for internal logic errors (bugs).
- Use `f32` and `f64` deliberately. Explain the precision choice.
- Do not use `unsafe` without a separate explanation of the safety
  argument in `explain.md`.

---

## 7. Testing and validation

- Every implemented function has unit tests or integration tests.
- Put API behaviour tests in `tests/rope_tests.rs`.
- Put numerical and mathematical-invariant tests in
  `tests/numerical_tests.rs`. Examples of invariants: a rotation keeps the
  vector norm; the dot product of two rotated vectors depends only on the
  relative position.
- Compare floating-point numbers with a tolerance. Explain the tolerance.
- Before you finish a task, run:

```bash
cargo fmt
cargo check --all-targets
cargo test
```

- Report the results honestly. If a test fails, show the output.

---

## 8. Task workflow

For every task:

1. Read `CLAUDE.md`.
2. Read the related sections of `explain.md`, `docs/mathematics.md` and
   `docs/architecture.md`.
3. Explain the plan: the mathematics first, then the Rust design.
4. Implement the smallest complete step.
5. Add tests.
6. Run the checks in section 7.
7. Append a new entry to `explain.md` (section 3 format).
8. Update `docs/` and the README roadmap status, if necessary.
9. Report what we built and what we learned, in ASD-STE100.
