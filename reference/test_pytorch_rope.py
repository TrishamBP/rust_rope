"""Self-tests of the PyTorch reference (validation tooling).

Run: reference/.venv/Scripts/python -m unittest discover -s reference -v

These tests check the reference itself against hand-calculated values and
mathematical properties. They also contain negative controls: they check
that the Rust comparison FAILS when the reference uses a wrong
configuration. A comparison that cannot fail proves nothing.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_rope  # noqa: E402
import pytorch_rope as ref  # noqa: E402

F64 = torch.float64


def deterministic(shape, seed=0.0):
    n = math.prod(shape)
    k = torch.arange(n, dtype=F64)
    return (2.0 * torch.sin(0.731 * k + seed) + 0.25 * torch.cos(1.37 * k)).reshape(shape)


class ReferenceTests(unittest.TestCase):
    def test_inverse_frequencies_d8(self):
        theta = ref.inverse_frequencies(8, 10_000.0, F64)
        expected = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=F64)
        self.assertTrue(torch.allclose(theta, expected, rtol=1e-14, atol=0.0))

    def test_position_zero_is_identity(self):
        x = deterministic((2, 1, 3, 8))
        for layout in (ref.INTERLEAVED, ref.SPLIT_HALF):
            y = ref.apply_rope(x, base=10_000.0, rotary_dim=8, start=0, layout=layout)
            self.assertTrue(torch.equal(x, y), layout)

    def test_worked_example_entry_10(self):
        x = torch.arange(1.0, 9.0, dtype=F64).reshape(1, 1, 1, 8)
        y = ref.apply_rope(x, base=10_000.0, rotary_dim=8, start=1).flatten()
        expected = [-1.142639664, 1.922075597, 2.585678829, 4.279516911,
                    4.939751002, 6.049699169, 6.991996501, 8.006995999]
        for a, e in zip(y.tolist(), expected):
            self.assertAlmostEqual(a, e, places=8)

    def test_pair_norms_preserved_both_layouts(self):
        x = deterministic((2, 5, 2, 8), 1.0)
        for layout, pairs in ((ref.INTERLEAVED, lambda t: t.reshape(2, 5, 2, 4, 2)),
                              (ref.SPLIT_HALF, lambda t: torch.stack((t[..., :4], t[..., 4:]), -1))):
            y = ref.apply_rope(x, base=10_000.0, rotary_dim=8, start=100, layout=layout)
            nx = (pairs(x) ** 2).sum(-1)
            ny = (pairs(y) ** 2).sum(-1)
            self.assertTrue(torch.allclose(nx, ny, rtol=1e-13, atol=1e-13), layout)

    def test_split_half_equals_interleaved_after_permutation(self):
        # Split-half pair j = (x[j], x[j + r/2]). Interleaved pair j = (x[2j], x[2j+1]).
        # Permute so that x[j] -> slot 2j and x[j + r/2] -> slot 2j+1.
        r = 8
        perm = [v for j in range(r // 2) for v in (j, j + r // 2)]  # [0,4,1,5,2,6,3,7]
        x = deterministic((1, 4, 2, r), 2.0)
        split = ref.apply_rope(x, base=10_000.0, rotary_dim=r, start=3, layout=ref.SPLIT_HALF)
        inter = ref.apply_rope(x[..., perm], base=10_000.0, rotary_dim=r, start=3)
        self.assertTrue(torch.allclose(split[..., perm], inter, rtol=0, atol=1e-15))

    def test_partial_rope_keeps_the_tail(self):
        x = deterministic((1, 3, 2, 12), 0.5)
        for layout in (ref.INTERLEAVED, ref.SPLIT_HALF):
            y = ref.apply_rope(x, base=10_000.0, rotary_dim=4, start=7, layout=layout)
            self.assertTrue(torch.equal(y[..., 4:], x[..., 4:]))
            self.assertFalse(torch.equal(y[..., :4], x[..., :4]))

    def test_relative_position_property(self):
        q = deterministic((1, 1, 1, 8), 0.3)
        k = deterministic((1, 1, 1, 8), -1.0)
        def score(m, n):
            qm = ref.apply_rope(q, base=10_000.0, rotary_dim=8, start=m)
            kn = ref.apply_rope(k, base=10_000.0, rotary_dim=8, start=n)
            return (qm * kn).sum().item()
        self.assertAlmostEqual(score(0, 5), score(100, 105), places=11)

    def test_invalid_configurations(self):
        x = deterministic((1, 1, 1, 8))
        with self.assertRaises(ValueError):
            ref.apply_rope(x, base=10_000.0, rotary_dim=10, start=0)
        with self.assertRaises(ValueError):
            ref.inverse_frequencies(7, 10_000.0, F64)
        with self.assertRaises(ValueError):
            ref.inverse_frequencies(8, 1.0, F64)


class NegativeControls(unittest.TestCase):
    """The comparison must detect wrong conventions."""

    @classmethod
    def setUpClass(cls):
        if not compare_rope.DATA.exists():
            raise unittest.SkipTest("run `cargo run --example export_reference_cases` first")
        # Select by configuration fields, not by name (names changed between stages).
        cls.cases = [c for c in compare_rope.read_cases(compare_rope.DATA)
                     if c.fields["shape"] == "2,5,3,8" and c.fields["start"] == "3"
                     and c.fields["base"] == "10000.0" and c.fields["layout"] == "interleaved"
                     and c.fields.get("scaling", "none") == "none"]
        assert cls.cases, "the selected reference case is missing from the export"

    def max_error_with(self, **override):
        case = self.cases[0]
        f = dict(case.fields, **override)
        x = torch.tensor(case.input, dtype=F64).reshape(case.shape)
        y = ref.apply_rope(x, base=float(f["base"]), rotary_dim=int(f["rotary_dim"]),
                           start=int(f["start"]), layout=f["layout"])
        rust = torch.tensor(case.output, dtype=F64).reshape(case.shape)
        return (y - rust).abs().max().item()

    def test_correct_configuration_matches(self):
        self.assertLess(self.max_error_with(), 1e-13)

    def test_wrong_layout_is_detected(self):
        self.assertGreater(self.max_error_with(layout=ref.SPLIT_HALF), 1e-2)

    def test_wrong_base_is_detected(self):
        self.assertGreater(self.max_error_with(base="500000.0"), 1e-3)

    def test_wrong_start_position_is_detected(self):
        self.assertGreater(self.max_error_with(start="4"), 1e-2)


class ScalingReferenceTests(unittest.TestCase):
    """Stage 8: the reference scaling methods and negative controls."""

    def test_position_interpolation_equals_linear(self):
        # PI scales positions; linear scales frequencies. Same angles.
        pi = ref.scaling_parameters("pi:2048:8192", rotary_dim=8, base=10_000.0,
                                    start=4096, length=3, dtype=F64)
        lin = ref.scaling_parameters("linear:4.0", rotary_dim=8, base=10_000.0,
                                     start=4096, length=3, dtype=F64)
        self.assertTrue(torch.allclose(ref.angles(pi[0], pi[1]), ref.angles(lin[0], lin[1]),
                                       rtol=1e-15, atol=0))
        # Position 4096 with L/L' = 1/4 behaves like position 1024.
        self.assertEqual(pi[0][0].item(), 1024.0)

    def test_ntk_endpoints(self):
        _, theta, _ = ref.scaling_parameters("none", rotary_dim=64, base=10_000.0,
                                             start=0, length=1, dtype=F64)
        _, ntk, _ = ref.scaling_parameters("ntk:4.0", rotary_dim=64, base=10_000.0,
                                           start=0, length=1, dtype=F64)
        self.assertEqual(ntk[0].item(), 1.0)
        self.assertAlmostEqual(ntk[-1].item(), theta[-1].item() / 4.0, delta=1e-17)

    def test_yarn_bands_and_factor(self):
        _, theta, _ = ref.scaling_parameters("none", rotary_dim=8, base=10_000.0,
                                             start=0, length=1, dtype=F64)
        _, y, factor = ref.scaling_parameters("yarn:4.0:2048:1.0:32.0", rotary_dim=8,
                                              base=10_000.0, start=0, length=1, dtype=F64)
        self.assertEqual(y[0].item(), theta[0].item())
        self.assertEqual(y[3].item(), theta[3].item() / 4.0)
        self.assertAlmostEqual(factor, 0.1 * math.log(4.0) + 1.0, places=15)

    def test_missing_scaling_is_detected(self):
        if not compare_rope.DATA.exists():
            self.skipTest("run the Rust export first")
        cases = [c for c in compare_rope.read_cases(compare_rope.DATA)
                 if c.fields.get("scaling", "none") != "none" and int(c.fields["start"]) >= 4096
                 and not c.fields["scaling"].startswith("linear:1.0")]
        self.assertGreater(len(cases), 10)
        for case in cases:
            rust = torch.tensor(case.output, dtype=F64).reshape(case.shape)
            correct = compare_rope.reference_output(case, F64)
            wrong_case = compare_rope.Case(dict(case.fields, scaling="none"), case.input, case.output)
            wrong = compare_rope.reference_output(wrong_case, F64)
            self.assertLess((rust - correct).abs().max().item(), 1e-9, case.name)
            self.assertGreater((rust - wrong).abs().max().item(), 1e-3, case.name)


if __name__ == "__main__":
    unittest.main()
