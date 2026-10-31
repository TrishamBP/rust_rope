# Architecture

This document describes the structure of `rust_rope`.
It explains what each part does, and why the parts are separate.

> **Status:** Stage 2 is complete. The `math` module is implemented. The
> `rope` and `utils` modules are empty (planned).

---

## 1. Crates in this package

A **crate** is the unit that the Rust compiler compiles. One Cargo
**package** can contain several crates.

| Crate | Root file | Purpose | Status |
|---|---|---|---|
| Library | `src/lib.rs` | All logic. Other crates use it. | `math` implemented |
| Binary | `src/main.rs` | A thin demonstration program. | Prints the status |
| Integration tests | `tests/rope_tests.rs` | RoPE API tests. | Placeholder |
| Integration tests | `tests/numerical_tests.rs` | Cross-module mathematical invariants. | Implemented (Stage 2) |
| Example | `examples/math_foundations.rs` | Prints the Stage 2 numerical examples. | Implemented |
| Example | `examples/basic_rope.rs` | A RoPE usage example. | Placeholder (Stage 4) |
| Benchmark | `benches/rope_benchmark.rs` | Performance measurements. | Placeholder (Stage 6) |

Why do we put all logic in the library? Tests, examples and benchmarks can
only use code from the library crate. They cannot use code from `main.rs`.

## 2. Module tree

```text
rust_rope (library)
├── math            General mathematical primitives (no RoPE knowledge) ✅
│   ├── (mod.rs)      approx_eq, DEFAULT_TOLERANCE                      ✅
│   ├── trigonometry  Angle conversion, sine, cosine, unit circle       ✅
│   ├── matrix        Vector, Vector2, Matrix2, rotations               ✅
│   └── complex       Complex numbers, polar form, rotation             ✅
├── rope            RoPE-specific logic                                 ⏳
│   ├── config        RoPE configuration
│   ├── frequencies   Positional rotation frequencies
│   ├── position      Position indexing
│   ├── rotations     Vector rotation operations
│   ├── embeddings    Applying RoPE to embeddings
│   └── scaling       RoPE scaling methods
└── utils           Shared helpers                                      ⏳
    └── validation    Input and configuration validation
```

## 3. Implemented module responsibilities (Stage 2)

### `math` (`src/math/mod.rs`)

| Item | Kind | Responsibility |
|---|---|---|
| `approx_eq(a, b, tol)` | function | Compares floats: `|a − b| ≤ tol·max(1, |a|, |b|)`. Handles infinities and NaN. |
| `DEFAULT_TOLERANCE` | constant | `1e-12`, the project-wide test tolerance. |

### `math::trigonometry`

| Item | Responsibility |
|---|---|
| `degrees_to_radians`, `radians_to_degrees` | Angle conversion. Bit-for-bit equal to `f64::to_radians`/`to_degrees`. |
| `sine`, `cosine`, `sin_cos` | Thin, `#[inline]` wrappers around the standard library. |
| `unit_circle_point(θ)` | Returns `[cos θ, sin θ]`. |

Depends on: nothing.

### `math::matrix`

| Item | Kind | Responsibility |
|---|---|---|
| `Vector` | struct (heap, `Vec<f64>`) | `n`-dimensional vector: `new`, `from_slice`, `dimension`, `is_empty`, `components`, `into_components`, `checked_add`, `checked_sub`, `dot`, `magnitude`. |
| `DimensionMismatch` | error struct | Returned by vector operations on different dimensions. Implements `Display` and `std::error::Error`. |
| `Vector2` | type alias `[f64; 2]` | 2-D vector, stored inline. |
| `Matrix2` | struct (inline, `Copy`) | 2 × 2 matrix: `new`, `from_rows`, `rows`, `get`, `IDENTITY`, `transpose`, `determinant`, `mul_vector`, `mul_matrix`, `rotation(θ)`. |
| `rotate_vector(v, θ)` | function | `R(θ)·v`. |

Depends on: `math::trigonometry`.

### `math::complex`

| Item | Kind | Responsibility |
|---|---|---|
| `Complex` | struct (inline, `Copy`) | `new`, `re`, `im`, `ZERO`, `ONE`, `I`, `conjugate`, `magnitude` (`hypot`), `phase` (`atan2`), `from_polar(r, φ)`, `rotate(θ)`. |
| `Add`, `Sub`, `Mul` | trait impls | `z + w`, `z − w`, `z * w`. |

Depends on: `math::trigonometry`.

## 4. Dependency direction

```text
        rope ───────► math ◄──── (tests, examples, benches)
          │
          └─────────► utils

  inside math:   matrix ──► trigonometry ◄── complex
```

- `rope` can use `math` and `utils`.
- `math` must not use `rope` or `utils`.
- Inside `math`, only `trigonometry` is a shared dependency. `matrix` and
  `complex` do not depend on each other. The tests that compare them are in
  `tests/numerical_tests.rs`.

Why? The mathematical primitives are general. A rotation matrix or a complex
number does not need to know about RoPE. If `math` stays independent, we can
test it alone, and we can reuse it. A one-way dependency also prevents
circular dependencies between modules.

## 5. Data representation decisions (Stage 2)

| Concept | Rust type | Size | Location | Reason |
|---|---|---|---|---|
| `n`-dimensional vector | `Vector` (`Vec<f64>`) | 24 B + 8·n B | stack + heap | `n` known only at run time |
| 2-D vector | `[f64; 2]` | 16 B | inline | size fixed, compiler-checked |
| 2 × 2 matrix | `[[f64; 2]; 2]` | 32 B | inline | size fixed, row-major |
| Complex number | `{ re: f64, im: f64 }` | 16 B | inline | two real numbers |

Error policy: caller data errors (dimension mismatch) return `Result`.
Caller bugs with fixed bounds (`Matrix2::get(2, 0)`) panic and are
documented in a `# Panics` section.

## 6. Planned data flow (Stages 3–4)

This flow is planned. It is not implemented.

```text
config (d, base)                 position m
      │                              │
      ▼                              │
frequencies: θⱼ = base^(-2j/d)       │
      │                              │
      └──────────────┬───────────────┘
                     ▼
          angles: φ = m · θⱼ
                     │
                     ▼
     trigonometry: cos(φ), sin(φ)          (math::trigonometry ✅)
                     │
                     ▼
 rotations: rotate each pair (x, y) by φ   (same formula as math::matrix::rotate_vector ✅)
                     │
                     ▼
 embeddings: apply to query and key vectors
```

`validation` checks the inputs at the start (for example, `d` must be even).
`scaling` modifies the frequencies or the positions for long contexts.

## 7. Design rules

- One module has one responsibility.
- No abstraction without a real need.
- Only the standard library, unless `explain.md` justifies a dependency.
- Errors that a caller can cause return `Result`.
