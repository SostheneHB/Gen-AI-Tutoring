"""Tests for the free decomposition, the welfare solver and the conformal margin."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2.certificate import conformal_margin, score  # noqa: E402
from p2.decomposition import FREE_FACTORS, free_decomposition  # noqa: E402


def random_models(rng, n=5):
    out = {}
    for g in (0, 1):
        A = rng.normal(size=(n, n))
        out[g] = dict(A=0.9 * A / np.abs(np.linalg.eigvals(A)).max(), a=rng.normal(size=n))
    return out


def test_free_decomposition_adds_up():
    rng = np.random.default_rng(0)
    for _ in range(20):
        models = random_models(rng)
        m0 = {g: rng.normal(size=5) for g in (0, 1)}
        c = rng.normal(size=5)
        d = free_decomposition(models, m0, 25, c, 0.3)
        assert abs(sum(d["shapley"][f] for f in FREE_FACTORS) - d["gap"]) < 1e-10
        assert abs(sum(d["sequential"][f] for f in FREE_FACTORS) - d["gap"]) < 1e-10


def test_free_decomposition_same_groups_is_initial_only():
    rng = np.random.default_rng(1)
    models = random_models(rng)
    models[1] = models[0]
    m0 = {g: rng.normal(size=5) for g in (0, 1)}
    d = free_decomposition(models, m0, 10, rng.normal(size=5))
    assert abs(d["shapley"]["A"]) < 1e-12 and abs(d["shapley"]["a"]) < 1e-12


def test_conformal_margin_coverage():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(2000):
        s = rng.normal(size=101)
        hits += s[-1] <= conformal_margin(s[:-1], 0.1)
    assert 0.88 <= hits / 2000 <= 0.93          # between 1 - alpha and 1 - alpha + 1/(m+1)


def test_conformal_margin_too_few():
    assert conformal_margin(np.zeros(5), 0.1) == float("inf")


def test_score_infinite_when_a_group_loses():
    assert score((0, 0, 1.0), (-1e-6, 0.1, 1.0)) == float("inf")
    assert abs(score((0, 0, 1.0), (0.0, 0.1, 0.9)) - 0.1) < 1e-12
