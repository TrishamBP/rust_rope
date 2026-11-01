"""Compares the Rust RoPE outputs with the independent PyTorch reference.

Validation tooling, not production code.

Workflow:
    cargo run --example export_reference_cases      # writes reference/data/rust_cases.txt
    reference/.venv/Scripts/python reference/compare_rope.py

For every case, the script:
1. Reads the configuration, the input and the Rust output.
2. Recomputes the output with PyTorch in float64 (strict mode) and in
   float32 (precision study).
3. Calculates the maximum absolute error, the mean absolute error and the
   maximum relative error, and checks a tolerance.

The exit code is 0 only if every case passes in both modes.

Tolerance model (see explain.md, Entry 13):
    tol = 8 * eps * (1 + max_angle) * max(1, max|x|)
- eps: machine epsilon of the reference dtype (2.2e-16 for float64,
  1.19e-7 for float32).
- max_angle: the largest rotation angle of the case, max(m) * max(theta).
  The absolute rounding error of an angle grows with its size.
- max|x|: the largest input magnitude. The output error scales with it.
- 8: a safety factor for the few roundings in theta, phi, cos, sin and the
  multiply-add (about 4 to 6 roundings).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytorch_rope as ref  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "rust_cases.txt"


@dataclass
class Case:
    fields: dict
    input: list
    output: list

    @property
    def name(self) -> str:
        return self.fields["name"]

    @property
    def shape(self) -> tuple:
        return tuple(int(v) for v in self.fields["shape"].split(","))


def read_cases(path: Path) -> list[Case]:
    cases, fields, values = [], None, {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        if head == "case":
            fields = dict(item.split("=", 1) for item in rest.split())
            values = {}
        elif head in ("input", "output"):
            # float() parses the shortest round-trip text exactly.
            values[head] = [float(v) for v in rest.split()]
        elif head == "end":
            cases.append(Case(fields, values["input"], values["output"]))
    return cases


def reference_output(case: Case, dtype: torch.dtype) -> torch.Tensor:
    f = case.fields
    x = torch.tensor(case.input, dtype=torch.float64).reshape(case.shape).to(dtype)
    base, rotary_dim, start = float(f["base"]), int(f["rotary_dim"]), int(f["start"])
    position_ids, inv_freq, attention_factor = ref.scaling_parameters(
        f.get("scaling", "none"), rotary_dim=rotary_dim, base=base,
        start=start, length=case.shape[1], dtype=dtype,
    )
    out = ref.apply_rope(
        x, base=base, rotary_dim=rotary_dim, start=start,
        layout=f.get("layout", ref.INTERLEAVED),
        position_ids=position_ids, inv_freq=inv_freq, attention_factor=attention_factor,
    )
    return out.to(torch.float64)


def max_angle(case: Case) -> float:
    """The largest |phi| of the case: (start + S - 1) * theta_0, theta_0 = 1."""
    f = case.fields
    last_position = int(f["start"]) + case.shape[1] - 1
    return float(last_position) * float(f.get("max_inv_freq", 1.0))


def errors(rust: torch.Tensor, reference: torch.Tensor) -> tuple[float, float, float]:
    diff = (rust - reference).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # Relative error only where the reference is not close to zero. Near
    # zero, |diff| / |ref| becomes huge even for a tiny absolute error.
    floor = 1e-3 * max(reference.abs().max().item(), 1e-300)
    mask = reference.abs() >= floor
    max_rel = (diff[mask] / reference.abs()[mask]).max().item() if mask.any() else 0.0
    return max_abs, mean_abs, max_rel


def main() -> int:
    if not DATA.exists():
        print(f"missing {DATA}; run `cargo run --example export_reference_cases` first")
        return 2
    cases = read_cases(DATA)
    modes = {"float64": (torch.float64, 2.0**-52), "float32": (torch.float32, 2.0**-23)}
    worst = {mode: [0.0, 0.0, 0.0] for mode in modes}
    failures = []
    print(f"{len(cases)} cases from {DATA.name}")
    print(f"{'case':<46} {'mode':<8} {'max abs':>10} {'mean abs':>10} {'max rel':>10} {'tol':>10}")
    for case in cases:
        rust = torch.tensor(case.output, dtype=torch.float64).reshape(case.shape)
        # YaRN multiplies the output by its attention factor (<= 1.4 here).
        scale = max(1.0, max(abs(v) for v in case.input)) * (1.5 if case.fields.get("scaling", "none").startswith("yarn") else 1.0)
        for mode, (dtype, eps) in modes.items():
            reference = reference_output(case, dtype)
            max_abs, mean_abs, max_rel = errors(rust, reference)
            tol = 8.0 * eps * (1.0 + max_angle(case)) * scale
            ok = max_abs <= tol
            w = worst[mode]
            w[0], w[1], w[2] = max(w[0], max_abs), max(w[1], mean_abs), max(w[2], max_rel)
            if not ok:
                failures.append((case.name, mode, max_abs, tol))
            flag = "" if ok else "  FAIL"
            print(f"{case.name:<46} {mode:<8} {max_abs:10.2e} {mean_abs:10.2e} {max_rel:10.2e} {tol:10.2e}{flag}")
    print()
    for mode, (mx, mean, rel) in worst.items():
        print(f"worst over all cases, {mode}: max abs {mx:.3e}, largest mean abs {mean:.3e}, max rel {rel:.3e}")
    if failures:
        print(f"\n{len(failures)} FAILURES")
        for f in failures:
            print("  ", f)
        return 1
    print(f"\nall {len(cases)} cases pass in float64 and float32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
