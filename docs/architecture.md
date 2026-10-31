# Architecture

This document describes the structure of `rust_rope`.
It explains what each part does, and why the parts are separate.

> **Status:** Stage 1. The structure exists. The modules are empty.

---

## 1. Crates in this package

A **crate** is the unit that the Rust compiler compiles. One Cargo
**package** can contain several crates.

| Crate | Root file | Purpose |
|---|---|---|
| Library | `src/lib.rs` | All logic. Other crates use it. |
| Binary | `src/main.rs` | A thin demonstration program. |
| Integration tests | `tests/*.rs` | Each file is a separate crate that uses the library. |
| Example | `examples/basic_rope.rs` | A runnable usage example. |
| Benchmark | `benches/rope_benchmark.rs` | Performance measurements. |

Why do we put all logic in the library? Tests, examples and benchmarks can
only use code from the library crate. They cannot use code from `main.rs`.

## 2. Module tree

```text
rust_rope (library)
├── math            General mathematical primitives (no RoPE knowledge)
│   ├── complex       Complex numbers
│   ├── trigonometry  Sine and cosine
│   └── matrix        Vectors and matrices
├── rope            RoPE-specific logic
│   ├── config        RoPE configuration
│   ├── frequencies   Positional rotation frequencies
│   ├── position      Position indexing
│   ├── rotations     Vector rotation operations
│   ├── embeddings    Applying RoPE to embeddings
│   └── scaling       RoPE scaling methods
└── utils           Shared helpers
    └── validation    Input and configuration validation
```

## 3. Dependency direction

```text
        rope ───────► math
          │
          └─────────► utils
```

- `rope` can use `math` and `utils`.
- `math` must not use `rope`.

Why? The mathematical primitives are general. A rotation matrix or a complex
number does not need to know about RoPE. If `math` stays independent, we can
test it alone, and we can reuse it. A one-way dependency also prevents
circular dependencies between modules.

## 4. Planned data flow (Stages 3–4)

This flow is planned. It is not implemented.

```text
config (d, base)                 position m
      │                              │
      ▼                              │
frequencies: θ_i = base^(-2i/d)      │
      │                              │
      └──────────────┬───────────────┘
                     ▼
          angles: φ = m · θ_i
                     │
                     ▼
     trigonometry: cos(φ), sin(φ)
                     │
                     ▼
 rotations: rotate each pair (x, y) by φ
                     │
                     ▼
 embeddings: apply to query and key vectors
```

`validation` checks the inputs at the start (for example, `d` must be even).
`scaling` modifies the frequencies or the positions for long contexts.

## 5. Design rules

- One module has one responsibility.
- No abstraction without a real need.
- Only the standard library, unless `explain.md` justifies a dependency.
- Errors that a caller can cause return `Result`.
