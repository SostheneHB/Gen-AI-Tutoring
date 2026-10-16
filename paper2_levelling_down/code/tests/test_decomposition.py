"""The splits must add up to the model gap (random linear models, random controls)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p2.decomposition import gap_decomposition, proposition1_terms, rollout_mean  # noqa: E402


def random_case(seed, n=6, T=12):
    rng = np.random.default_rng(seed)
    models = {g: dict(A=0.9 * rng.standard_normal((n, n)) / np.sqrt(n), B=rng.standard_normal(n),
                      a=rng.standard_normal(n)) for g in (0, 1)}
    m0 = {g: rng.standard_normal(n) for g in (0, 1)}
    v = {g: rng.uniform(-1, 1, T) for g in (0, 1)}
    return models, m0, v, rng.standard_normal(n)


def test_all_splits_add_up():
    for seed in range(50):
        models, m0, v, c = random_case(seed)
        gap = c @ (rollout_mean(m0[1], **{k: models[1][k] for k in "ABa"}, v=v[1])[-1]
                   - rollout_mean(m0[0], **{k: models[0][k] for k in "ABa"}, v=v[0])[-1])
        out = gap_decomposition(models, m0, v, c)
        p1 = proposition1_terms(models, m0, v, c)
        for split in (out["sequential"], out["shapley"], p1):
            assert abs(sum(split.values()) - gap) < 1e-9 * max(1, abs(gap))


def test_null_factor_gets_zero():
    models, m0, v, c = random_case(0)
    models[1]["B"] = models[0]["B"].copy()         # same response to teaching
    out = gap_decomposition(models, m0, v, c)
    assert abs(out["shapley"]["R"]) < 1e-12
