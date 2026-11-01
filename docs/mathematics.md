# Mathematics Notes

This document collects the mathematics of the project in one place. It is a
compact reference. The full explanations, with Rust details and numerical
examples, are in `explain.md` (the entry numbers are given below).

> **Status:** Stage 3 is complete. Sections 2–7 are **implemented** in
> `src/math/`. Section 8 (frequencies, positions and angles) is
> **implemented** in `src/rope/`. Sections 9–11 are **planned**.

---

## 1. Notation

We define each symbol before we use it.

| Symbol | Meaning |
|---|---|
| `v = [v₁, ..., vₙ]` | A vector with `n` components. `n` is the dimension. |
| `a · b` | The dot product of the vectors `a` and `b`. |
| `‖v‖` | The magnitude (Euclidean norm, length) of `v`. |
| `M`, `mᵣ꜀` | A matrix, and its element in row `r`, column `c`. |
| `Mᵀ` | The transpose of `M`. |
| `det(M)` | The determinant of `M`. |
| `I` | The identity matrix. |
| `θ`, `φ`, `α`, `β` | Angles, in radians. |
| `π` | The ratio of circumference to diameter, `≈ 3.14159`. |
| `R(θ)` | The 2 × 2 rotation matrix for the angle `θ`. |
| `i` | The imaginary unit, `i² = −1`. |
| `z = a + bi` | A complex number. `a = Re(z)`, `b = Im(z)`. |
| `z̄`, `conj(z)` | The conjugate of `z`. |
| `|z|`, `arg(z)` | The magnitude and the phase of `z`. |
| `e` | Euler's number, `≈ 2.71828`. |
| `n!` | The factorial `1·2·…·n`. |

**RoPE notation (used from Stage 3):**

| Symbol | Meaning |
|---|---|
| `d` | The embedding dimension of one attention head. It must be even. |
| `j` | The index of a dimension pair, `j = 0, 1, ..., d/2 − 1`. |
| `m`, `n` | Token positions in the sequence (0, 1, 2, ...). |
| `b` | The frequency base. The RoFormer paper uses `b = 10000`. |
| `θⱼ` | The rotation frequency of pair `j` (radians per position). |
| `P` | The number of pairs, `d/2`. (Added in Stage 3.) |
| `φ(m, j)` | The rotation angle of pair `j` at position `m`: `m·θⱼ`. (Added in Stage 3.) |
| `λⱼ` | The wavelength of pair `j`: `2π/θⱼ` positions. (Added in Stage 3.) |
| `q`, `k` | A query vector and a key vector. |

**Notation change.** In Stage 1, this document used `i` for the pair
index. From Stage 2, the pair index is `j`, because `i` is the imaginary
unit. This avoids expressions like `e^(i·m·θᵢ)` where `i` has two meanings.

---

## 2. Vectors — implemented (`src/math/matrix.rs`, Entry 2)

**Addition and subtraction** (only for equal dimensions):

```text
a ± b = [a₁ ± b₁, a₂ ± b₂, ..., aₙ ± bₙ]
```

**Dot product:**

```text
a · b = Σᵢ aᵢ·bᵢ = ‖a‖·‖b‖·cos(α)
```

`α` is the angle between `a` and `b`. Derivation in 2-D: write
`a = r·[cos φ, sin φ]` and `b = s·[cos ψ, sin ψ]`. Then
`a · b = r·s·(cos φ cos ψ + sin φ sin ψ) = r·s·cos(ψ − φ)`.

- `a · b > 0`: similar direction. `a · b = 0`: perpendicular.
  `a · b < 0`: opposite directions.
- Attention uses `q · k` as the similarity score.

**Magnitude** (the theorem of Pythagoras, applied `n − 1` times):

```text
‖v‖ = √(Σᵢ vᵢ²) = √(v · v)
```

Example: `‖[3, 4]‖ = √(9 + 16) = 5`.

---

## 3. Complex numbers — implemented (`src/math/complex.rs`, Entry 3)

`z = a + bi` is the point `(a, b)` in the complex plane. Two real numbers
describe it completely, so it is stored as two `f64` values.

| Operation | Formula |
|---|---|
| Addition | `(a + bi) + (c + di) = (a + c) + (b + d)i` |
| Subtraction | `(a + bi) − (c + di) = (a − c) + (b − d)i` |
| Multiplication | `(a + bi)(c + di) = (ac − bd) + (ad + bc)i` |
| Conjugate | `conj(a + bi) = a − bi` |
| Magnitude | `|z| = √(a² + b²)` |
| Phase | `arg(z) = atan2(b, a) ∈ [−π, π]` |

**Derivation of the multiplication:** expand and use `i² = −1`:
`ac + adi + bci + bd·i² = (ac − bd) + (ad + bc)i`.

**Key properties:**

- `z·z̄ = a² + b² = |z|²`.
- `|z·w| = |z|·|w|` and `arg(z·w) = arg(z) + arg(w)` (modulo `2π`).
  **Multiplication multiplies the lengths and adds the angles.**
- Multiplication by `i` is a rotation by 90°.
- `Re[u·conj(w)]` equals the 2-D dot product of the points `u` and `w`.

**Why `atan2` and not `atan(b/a)`:** `atan(b/a)` cannot separate `1 + i`
from `−1 − i` (both give `b/a = 1`), and it is undefined for `a = 0`.

---

## 4. Trigonometry — implemented (`src/math/trigonometry.rs`, Entry 4)

**Radians.** The angle in radians is the arc length divided by the radius:
`θ = s / r`. A full turn is `2π`.

```text
θ_radians = θ_degrees × π / 180,      θ_degrees = θ_radians × 180 / π
```

Radians are the natural unit: `d/dθ sin θ = cos θ`, the power series, and
`sin θ ≈ θ` for small `θ` are only correct in radians.

**Unit circle.** The point at angle `θ` on the circle with radius 1 is:

```text
x = cos θ,     y = sin θ
```

| θ | 0 | π/2 | π | 3π/2 |
|---|---|---|---|---|
| `(cos θ, sin θ)` | (1, 0) | (0, 1) | (−1, 0) | (0, −1) |

**Identities:**

```text
sin²θ + cos²θ = 1                                 (Pythagoras)
cos(−θ) = cos θ,   sin(−θ) = −sin θ               (symmetry)
sin(θ + 2πk) = sin θ,   cos(θ + 2πk) = cos θ      (period 2π)
cos(α + β) = cos α cos β − sin α sin β            (angle addition)
sin(α + β) = sin α cos β + cos α sin β
```

The angle-addition formulas follow from complex multiplication of
`cos α + i sin α` and `cos β + i sin β` (section 3: the angles add).

---

## 5. 2 × 2 matrices and rotations — implemented (`src/math/matrix.rs`, Entry 5)

```text
[ a  b ] [ x ]   [ a·x + b·y ]
[ c  d ] [ y ] = [ c·x + d·y ]

[ a  b ] [ e  f ]   [ a·e + b·g   a·f + b·h ]
[ c  d ] [ g  h ] = [ c·e + d·g   c·f + d·h ]
```

- The columns of `M` are the images of `[1, 0]` and `[0, 1]`.
- `(M·N)·v = M·(N·v)`: first `N`, then `M`. In general `M·N ≠ N·M`.
- `det([[a, b], [c, d]]) = ad − bc` is the area scale factor. A negative
  determinant means a reflection.

**The rotation matrix:**

```text
R(θ) = [ cos θ   −sin θ ]
       [ sin θ    cos θ ]
```

**Derivation 1 (axis vectors).** `[1, 0]` rotates to `[cos θ, sin θ]`
(column 1). `[0, 1]` (angle `π/2`) rotates to
`[cos(θ + π/2), sin(θ + π/2)] = [−sin θ, cos θ]` (column 2).

**Derivation 2 (polar coordinates).** `v = r·[cos φ, sin φ]`. The rotation
gives `r·[cos(φ + θ), sin(φ + θ)]`. Expand with the angle-addition formulas:
`x' = x cos θ − y sin θ`, `y' = x sin θ + y cos θ`.

**Why it is a rotation:**

1. `R(θ)·0 = 0` (the origin is fixed).
2. `x'² + y'² = x² + y²` (lengths are kept; the cross terms cancel and
   `sin² + cos² = 1`).
3. `det R(θ) = cos²θ + sin²θ = 1` (no reflection, no area change).

**Properties:**

```text
R(α)·R(β) = R(α + β)
R(θ)ᵀ = R(−θ) = R(θ)⁻¹             (R is orthogonal: RᵀR = I)
R(α)ᵀ·R(β) = R(β − α)              (the relative-rotation identity)
```

---

## 6. Euler's formula — implemented (`src/math/complex.rs`, Entry 6)

```text
e^(iθ) = cos θ + i·sin θ
```

**Derivation.** Put `x = iθ` into `e^x = Σ xⁿ/n!`. The powers of `i` repeat:
`1, i, −1, −i, 1, ...`. The terms without `i` are the series of `cos θ`. The
terms with `i` are the series of `sin θ`.

- `e^(iθ)` is the point at angle `θ` on the unit circle. `|e^(iθ)| = 1`.
- Polar form: `z = r·e^(iφ)` with `r = |z|`, `φ = arg(z)`.
- Euler's identity: `e^(iπ) + 1 = 0`.

**Rotation by multiplication:**

```text
z·e^(iθ) = r·e^(i(φ + θ))
(x + iy)(cos θ + i sin θ) = (x cos θ − y sin θ) + i(x sin θ + y cos θ)
```

This is exactly `R(θ)·[x, y]`. Multiplication by `c + di` equals the matrix
`[[c, −d], [d, c]]`.

| Matrix form | Complex form |
|---|---|
| `R(α)R(β) = R(α + β)` | `e^(iα)e^(iβ) = e^(i(α+β))` |
| `R(θ)ᵀ = R(−θ)` | `conj(e^(iθ)) = e^(−iθ)` |
| `R(θ)ⁿ = R(nθ)` | `(e^(iθ))ⁿ = e^(inθ)` (de Moivre) |

---

## 7. Floating-point precision — implemented in tests (all Stage 2 entries)

- An `f64` has 53 significant bits (about 16 decimal digits).
  `f64::EPSILON ≈ 2.2e-16`.
- `+ − × ÷ √` are correctly rounded. Small integer results are exact.
- `0.1 + 0.2 = 0.30000000000000004`. Compare with a tolerance.
- **Project rule:** `approx_eq(a, b, tol)` tests
  `|a − b| ≤ tol · max(1, |a|, |b|)`, with `tol = 1e-12` by default
  (absolute near zero, relative for large values).
- `PI` in `f64` is `π − 1.22e-16`. Thus `sin(PI) = 1.22e-16` and
  `cos(PI/2) = 6.12e-17`, not 0. The function is correct; the input is not
  exactly `π`.
- **The absolute error of an angle grows with its size:** about
  `|θ| · 1.1e-16` in `f64`, and about `|θ| · 6e-8` in `f32`. Measured:
  `sin(2π · 10⁶) = −4.46e-10` instead of 0.
- **Repeated rotations drift.** 10 000 rotations by `0.001` give a length
  error of `7.8e-14`. One rotation by `10.0` gives a length error of 0.
  Calculate `R(m·θ)` directly, not as `R(θ)ᵐ`.
- `hypot(a, b)` avoids the overflow of `√(a² + b²)` for very large values.

---

## 8. RoPE frequencies, positions and angles — implemented (Stage 3, Entries 7–9)

```text
θⱼ = b^(−2j/d),      j = 0, 1, ..., d/2 − 1
```

Planned topics: why the frequencies form a geometric sequence, and the
wavelength of each pair, `λⱼ = 2π / θⱼ`.

> **Status update (2026-11-01):** implemented in Stage 3
> (`src/rope/frequencies.rs`, `src/rope/position.rs`, explain.md Entries
> 7–9). The original planned text above is kept unchanged. The complete
> derivation follows.

### 8.1 Variables

| Symbol | Meaning |
|---|---|
| `d` | The rotary embedding dimension. Even, `d ≥ 2`. |
| `P = d/2` | The number of dimension pairs. |
| `j` | The pair index, `0 ≤ j < P`. Pair `j` holds dimensions `2j` and `2j + 1`. (The task text and some papers write `i`.) |
| `b` | The frequency base, finite, `b > 1`. Default `10000`. |
| `θⱼ` | The inverse frequency of pair `j`: radians per position. |
| `λⱼ` | The wavelength of pair `j`: positions per full turn. |

### 8.2 The formula

```text
θⱼ = 1 / b^(2j/d) = b^(−2j/d),        j = 0, 1, ..., d/2 − 1
```

### 8.3 Why different frequencies

One speed is not sufficient. A fast speed (`θ = 1`) wraps around after
`2π ≈ 6.28` positions: `44 mod 2π ≈ 0.018`, so position 44 looks like
position 0. A slow speed (`θ = 10⁻⁴`) does not wrap around for 62 832
positions, but neighbouring positions differ by only `10⁻⁴` radian. RoPE uses
`d/2` speeds from fast to slow, like the hands of a clock.

### 8.4 Why the exponent is `2j/d`

- `2j` is the index of the first dimension of pair `j`. The exponent is
  "dimension index / dimension", the same as the sinusoidal encoding of
  Vaswani et al. (2017): `PE(pos, 2i) = sin(pos / 10000^(2i/d))`.
- `2j/d = j/(d/2)`: the fraction of the pairs before pair `j`. It goes from
  `0` to `(d − 2)/d < 1`.
- With `j/d`, the exponent would stop at about `0.5`, and the slowest speed
  would be `1/√b` instead of almost `1/b`.
- **The division by `d` normalizes** the exponent to `[0, 1)`. The speed
  range is then `(1/b, 1]` for every `d`. Without it, `d = 128` would give
  `θ₆₃ = 10000^(−126) = 10^(−504)`, which is `0` in `f64`.

### 8.5 Monotonicity

`b^x = e^(x·ln b)` and `ln b > 0` for `b > 1`. Thus `b^x` increases with
`x`, the exponent `2j/d` increases with `j`, and `θⱼ = 1/b^(2j/d)`
**decreases** with `j`:

```text
1 = θ₀ > θ₁ > ... > θ_(d/2−1) = b^(−(d−2)/d) > 1/b
```

`b = 1` gives `θⱼ = 1` for all `j`. `0 < b < 1` reverses the order. Thus the
implementation requires `b > 1`.

### 8.6 Geometric sequence and wavelength

```text
θ_(j+1) / θⱼ = b^(−2/d)      (constant)       →  θⱼ = (b^(−2/d))^j
ln θⱼ = −(2j/d)·ln b          (linear in j: log-uniform spacing)
λⱼ = 2π / θⱼ = 2π·b^(2j/d)    (from 2π ≈ 6.28 to almost 2π·b)
```

### 8.7 Numerical derivation: `d = 8`, `b = 10000 = 10⁴`

`b^x = 10^(4x)`. Four pairs, `j = 0, 1, 2, 3`:

| `j` | `2j/d` | `b^(2j/d)` | `θⱼ` | `λⱼ` |
|---|---|---|---|---|
| 0 | 0/8 = 0 | 10⁰ = 1 | 1 | 6.28 |
| 1 | 2/8 = 0.25 | 10¹ = 10 | 0.1 | 62.83 |
| 2 | 4/8 = 0.5 | 10² = 100 | 0.01 | 628.32 |
| 3 | 6/8 = 0.75 | 10³ = 1000 | 0.001 | 6283.19 |

Ratio `10000^(−2/8) = 0.1`. For `d = 64`: ratio `10^(−1/8) ≈ 0.7499`, last
`θ ≈ 1.3335e−4`. For `d = 128`: ratio `10^(−1/16) ≈ 0.8660`, last
`θ ≈ 1.1548e−4`.

### 8.8 Positional angles

```text
φ(m, j) = m · θⱼ          (radians = positions × radians per position)
```

`m` is the token position (a non-negative integer). Derivation: each step of
one position adds the same angle `θⱼ`, so `φ(m, j) = φ(0, j) + m·θⱼ` with
`φ(0, j) = 0`.

Angle table, `d = 8`, `b = 10000`:

| `m` | `j = 0` | `j = 1` | `j = 2` | `j = 3` |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 1 | 1 | 0.1 | 0.01 | 0.001 |
| 2 | 2 | 0.2 | 0.02 | 0.002 |
| 3 | 3 | 0.3 | 0.03 | 0.003 |
| 4 | 4 | 0.4 | 0.04 | 0.004 |

- **Position 0:** `0·θⱼ = 0`, so `R = I`. The first token is not changed.
- **Same position, different pairs:** `θⱼ` decreases strictly, so
  `m·θⱼ` decreases strictly for `m ≥ 1`. All angles of one token differ.
- **Fixed pair, increasing position:** linear, step `θⱼ`.

### 8.9 From the angle to the rotation matrix

```text
R(m, j) = [ cos(m·θⱼ)   −sin(m·θⱼ) ]
          [ sin(m·θⱼ)    cos(m·θⱼ) ]
```

`(d, b, j)` give the speed `θⱼ`. `m` gives how far: `m·θⱼ`. Section 5 gives
the matrix. Properties (tested in `angle_table_defines_rope_rotation_matrices`):

```text
R(0, j) = I,     R(m, j) = R(1, j)^m,     R(m, j)ᵀ R(n, j) = R(n − m, j)
```

### 8.10 Precision of positions and angles

- `f64` stores every integer `m ≤ 2^53` exactly. Above, `2^53 + 1` converts
  to `2^53`. Thus `Position` requires `m ≤ 2^53`.
- `3 · 0.1 = 0.30000000000000004` in `f64`: `0.1` is not exact in binary.
- The absolute error of `m·θ` is about `|m·θ| · 1.1e−16` in `f64`. Measured
  for `θ = 10000^(−2/64)`: the `f32` angle at `m = 10⁶` differs from the
  `f64` angle by `0.022` radian.

## 9. The relative-position property — per pair tested (Stage 3); full vector planned (Stage 4)

Already proved for one pair of dimensions in Stage 2:

```text
Matrix form:   (R(mθ)·q)ᵀ (R(nθ)·k) = qᵀ R((n − m)θ) k
Complex form:  Re[q e^(imθ) · conj(k e^(inθ))] = Re[q conj(k) e^(i(m − n)θ)]
```

Both are tested in `tests/numerical_tests.rs`. Planned: the extension to all
`d/2` pairs with the block-diagonal rotation matrix.

> **Status update (2026-11-01):** Stage 3 tests the property for each pair
> with matrices built from the `AngleTable` (section 8.9). The extension to
> the full `d`-dimensional vector moves to Stage 4, because Stage 3 does not
> rotate vectors.

## 10. Dimension layouts — planned (Stage 4)

Planned topics: the "interleaved" layout (pairs `(0,1), (2,3), ...`) and the
"split-half" layout (pairs `(j, j + d/2)`).

## 11. Context extension — planned (Stage 5)

Planned topics: position interpolation, NTK-aware scaling and YaRN.
