"""Independent PyTorch reference implementation of RoPE (validation tooling).

This file is NOT the production implementation. The production
implementation is the Rust library in `src/`. This file exists only to
check the Rust results with an independent calculation.

Independence rules:
- This file never calls Rust code.
- It uses a different (but mathematically equal) formula convention for the
  inverse frequencies: 1 / base^(arange(0, r, 2) / r), the form of many
  PyTorch implementations. Rust uses base.powf(-2j / r).
- It uses whole-tensor PyTorch operations (broadcasting, reshape, stack,
  cat), not a loop over pairs.

Notation (the same as the Rust code and explain.md):
- x: tensor with the shape [B, S, H, D] (batch, sequence, heads, head dim).
- r: rotary dimension (even, 2 <= r <= D). r = D means "full RoPE".
- m: token position. The token at sequence index s has position start + s.
- theta_j = base^(-2j / r): inverse frequency of pair j, j = 0 .. r/2 - 1.
- phi(m, j) = m * theta_j: rotation angle.
"""

from __future__ import annotations

import math

import torch

INTERLEAVED = "interleaved"
SPLIT_HALF = "split_half"


def inverse_frequencies(rotary_dim: int, base: float, dtype: torch.dtype) -> torch.Tensor:
    """theta_j = 1 / base^(2j / r) for j = 0 .. r/2 - 1. Shape [r/2]."""
    if rotary_dim <= 0 or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be positive and even, got {rotary_dim}")
    if not base > 1.0:
        raise ValueError(f"base must be > 1, got {base}")
    # arange(0, r, 2) = [0, 2, 4, ..., r - 2] = 2j. Divide by r: the exponent.
    exponents = torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim
    return 1.0 / (torch.tensor(base, dtype=dtype) ** exponents)


def positions(start: int, length: int, dtype: torch.dtype) -> torch.Tensor:
    """[start, start + 1, ..., start + length - 1]. Shape [S]."""
    return torch.arange(start, start + length, dtype=dtype)


def angles(position_ids: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """phi[s, j] = position[s] * theta[j] (outer product). Shape [S, r/2]."""
    return torch.outer(position_ids, inv_freq)


def scaling_parameters(
    spec: str, *, rotary_dim: int, base: float, start: int, length: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Returns (position_ids, inv_freq, attention_factor) for a scaling spec (Stage 8).

    spec formats (the same text that Rust writes):
      none
      pi:<L>:<L'>                  position interpolation: positions * L/L'
      linear:<s>                   inv_freq / s
      ntk:<s>                      base' = base * s^(r/(r-2))
      yarn:<s>:<L>:<alpha>:<beta>  YaRN paper formulation (Peng et al. 2023)

    Independent of Rust: position interpolation is applied to the POSITIONS
    here (Rust applies it to the frequencies), and YaRN uses a vectorized
    clamp for the ramp.
    """
    pos = positions(start, length, dtype)
    kind, *args = spec.split(":")
    if kind == "none":
        return pos, inverse_frequencies(rotary_dim, base, dtype), 1.0
    if kind == "pi":
        original, target = int(args[0]), int(args[1])
        return pos * (original / target), inverse_frequencies(rotary_dim, base, dtype), 1.0
    if kind == "linear":
        factor = float(args[0])
        return pos, inverse_frequencies(rotary_dim, base, dtype) / factor, 1.0
    if kind == "ntk":
        factor = float(args[0])
        new_base = base * factor ** (rotary_dim / (rotary_dim - 2))
        return pos, inverse_frequencies(rotary_dim, new_base, dtype), 1.0
    if kind == "yarn":
        factor, original = float(args[0]), int(args[1])
        alpha, beta = float(args[2]), float(args[3])
        inv_freq = inverse_frequencies(rotary_dim, base, dtype)
        rotations = original * inv_freq / (2 * math.pi)  # r = L / lambda, lambda = 2pi/theta
        gamma = torch.clamp((rotations - alpha) / (beta - alpha), 0.0, 1.0)
        scaled = (1 - gamma) * inv_freq / factor + gamma * inv_freq
        return pos, scaled, 0.1 * math.log(factor) + 1.0
    raise ValueError(f"unknown scaling spec {spec!r}")


def _rotate_interleaved(x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pairs (x[2j], x[2j+1]). x_rot: [B, S, H, r]; cos, sin: [1, S, 1, r/2]."""
    pairs = x_rot.reshape(*x_rot.shape[:-1], -1, 2)  # [B, S, H, r/2, 2]
    a, b = pairs[..., 0], pairs[..., 1]  # x[2j], x[2j+1]
    out_a = a * cos - b * sin
    out_b = a * sin + b * cos
    return torch.stack((out_a, out_b), dim=-1).reshape(x_rot.shape)


def _rotate_split_half(x_rot: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pairs (x[j], x[j + r/2]), in the common 'rotate_half' form:
    x' = x * [cos, cos] + rotate_half(x) * [sin, sin],
    rotate_half([first, second]) = [-second, first].
    """
    half = x_rot.shape[-1] // 2
    first, second = x_rot[..., :half], x_rot[..., half:]
    rotated_half = torch.cat((-second, first), dim=-1)
    cos_full = torch.cat((cos, cos), dim=-1)
    sin_full = torch.cat((sin, sin), dim=-1)
    return x_rot * cos_full + rotated_half * sin_full


def apply_rope(
    x: torch.Tensor,
    *,
    base: float,
    rotary_dim: int,
    start: int,
    layout: str = INTERLEAVED,
    position_ids: torch.Tensor | None = None,
    inv_freq: torch.Tensor | None = None,
    attention_factor: float = 1.0,
) -> torch.Tensor:
    """Applies RoPE to x with the shape [B, S, H, D].

    Only the first `rotary_dim` features of each head vector are rotated.
    The features rotary_dim .. D are copied unchanged (partial RoPE).

    `position_ids` and `inv_freq` can be given explicitly (used by the
    scaling methods); otherwise they are the standard ones.
    """
    if x.dim() != 4:
        raise ValueError(f"x must have the shape [B, S, H, D], got {tuple(x.shape)}")
    d = x.shape[-1]
    if rotary_dim > d:
        raise ValueError(f"rotary_dim {rotary_dim} > head dimension {d}")
    dtype = x.dtype
    if inv_freq is None:
        inv_freq = inverse_frequencies(rotary_dim, base, dtype)
    if position_ids is None:
        position_ids = positions(start, x.shape[1], dtype)
    phi = angles(position_ids.to(dtype), inv_freq.to(dtype))  # [S, r/2]
    # broadcast over B and H; YaRN scales cos and sin by the attention factor
    cos = (torch.cos(phi) * attention_factor)[None, :, None, :]
    sin = (torch.sin(phi) * attention_factor)[None, :, None, :]

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    if layout == INTERLEAVED:
        rotated = _rotate_interleaved(x_rot, cos, sin)
    elif layout == SPLIT_HALF:
        rotated = _rotate_split_half(x_rot, cos, sin)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return torch.cat((rotated, x_pass), dim=-1)
