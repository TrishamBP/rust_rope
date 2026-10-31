# Mathematics Notes

This document collects the mathematics of the project.
We develop it stage by stage. Each section states its status.

> **Status:** Stage 1. This document contains only the notation and the
> outline. The derivations come in Stages 2–5.

---

## 1. Notation

We define each symbol before we use it.

| Symbol | Meaning |
|---|---|
| `d` | The embedding dimension of one attention head. It must be even. |
| `i` | The index of a dimension pair. `i = 0, 1, ..., d/2 − 1`. |
| `m`, `n` | Token positions in the sequence. `m = 0, 1, 2, ...` |
| `b` | The frequency base. The RoFormer paper uses `b = 10000`. |
| `θ_i` | The rotation frequency of pair `i` (radians per position). |
| `φ` | A rotation angle in radians. |
| `q`, `k` | A query vector and a key vector. |
| `R(φ)` | The 2 × 2 rotation matrix for the angle `φ`. |
| `⟨a, b⟩` | The dot product of the vectors `a` and `b`. |

## 2. Outline (planned)

### 2.1 Vectors and dot products — Stage 2, planned
The dot product measures similarity. Attention uses it to compare a query
with a key.

### 2.2 The two-dimensional rotation matrix — Stage 2, planned

```text
R(φ) = [[cos(φ), −sin(φ)],
        [sin(φ),  cos(φ)]]
```

Planned topics: the derivation from the angle-addition formulas, the proof
that `R(φ)` keeps the vector length, and the identity
`R(α)ᵀ R(β) = R(β − α)`.

### 2.3 Complex numbers — Stage 2, planned
Planned topics: the point `(x, y)` as the complex number `x + i·y`, Euler's
formula `e^(iφ) = cos(φ) + i·sin(φ)`, and rotation as multiplication by
`e^(iφ)`.

### 2.4 RoPE frequencies — Stage 3, planned

```text
θ_i = b^(−2i/d)
```

Planned topics: why the frequencies form a geometric sequence, and the
wavelength of each pair, `λ_i = 2π / θ_i`.

### 2.5 The relative-position property — Stage 3, planned

```text
⟨R(mθ) q, R(nθ) k⟩ = ⟨q, R((n − m)θ) k⟩
```

Planned topic: the full derivation. This property is the reason that RoPE
gives relative position information to attention.

### 2.6 Dimension layouts — Stage 4, planned
Planned topics: the "interleaved" layout (pairs `(0,1), (2,3), ...`) and the
"split-half" layout (pairs `(j, j + d/2)`).

### 2.7 Context extension — Stage 5, planned
Planned topics: position interpolation, NTK-aware scaling and YaRN.

### 2.8 Floating-point precision — all stages
Planned topics: `f32` versus `f64`, the error of `sin` and `cos` for large
angles, and tolerances in tests.
