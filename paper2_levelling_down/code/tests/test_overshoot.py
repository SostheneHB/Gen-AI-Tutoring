"""The price program must not overshoot equality (review of 16 October 2026, point 6)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2.certificate import margin_plan  # noqa: E402
from p2.frontier import fair_frontier  # noqa: E402


class Inst:
    def __init__(self, hk, hd, ak, ad, n):
        self.hk, self.hd, self.ak, self.ad, self.n, self.T = hk, hd, ak, ad, n, len(hk[0])

    def outcomes(self, v):
        K = {g: self.ak[g] + self.hk[g] @ v[g] for g in (0, 1)}
        D = {g: self.ad[g] + self.hd[g] @ v[g] for g in (0, 1)}
        return K, D, D[1] - D[0], sum(self.n[g] * v[g].sum() for g in (0, 1))


def instance():
    # group 1 is behind (gap -0.2); its reference teaches the step with a small effect on D and a large
    # effect on K; moving that teaching to the other step keeps K and saves budget but raises D a lot
    hk = {0: np.array([1.0, 1.0]), 1: np.array([2.0, 1.0])}
    hd = {0: np.array([1.0, 1.0]), 1: np.array([2.0, 0.1])}
    M = Inst(hk, hd, {0: 0.0, 1: 0.0}, {0: 0.0, 1: -0.4}, {0: 1, 1: 1})
    v = {0: np.array([0.0, 0.0]), 1: np.array([0.0, 1.0])}    # gap -0.3; moving 0.5 to step 0 keeps K1, saves 0.5, adds 0.9 to D1
    return M, v


def test_price_program_does_not_overshoot():
    M, v = instance()
    _, _, gap_ref, _ = M.outcomes(v)
    assert gap_ref < 0
    r = fair_frontier(M, v, [0.1])[0]
    assert r["feasible"]
    _, _, gap, _ = M.outcomes(r["plan"])
    narrowing = abs(gap_ref) - abs(gap)
    assert abs(gap) <= abs(gap_ref) + 1e-9 and narrowing >= 0.1 - 1e-9


def test_margin_plan_does_not_overshoot():
    M, v = instance()
    _, _, gap_ref, _ = M.outcomes(v)
    p = margin_plan(M, v, 0.1, 0.0)
    _, _, gap, _ = M.outcomes(p)
    assert abs(gap) <= abs(gap_ref) + 1e-9
