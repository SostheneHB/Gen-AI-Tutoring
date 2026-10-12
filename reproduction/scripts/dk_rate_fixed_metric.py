"""
Estimation rate of the dynamical-similarity score D_K with a FIXED metric
(reviewer point 6 and priority experiment 5).

System: the fixed-dictionary tutoring simulator of Section 4.1 (Eq. simulator),
exploratory control u ~ U[-1, 1], 25 turns, 10-feature quadratic dictionary.
For each group s the affine model z+ = A_s z + B_s u + a_s is fitted by least
squares and K^(s) = A_s^T is its coefficient operator.

* Dictionary scaling and Gamma = G_ref + 1e-3 lambda_max(G_ref) I are computed
  ONCE from an independent reference sample (seed 7) and frozen for every size
  and seed.
* Reference operators are fitted on independent samples of n_ref trajectories
  per group, for three values of n_ref, to quantify the finite-reference floor.
* A second arm re-estimates the scaling and Gamma from every sample, to show the
  additional metric-perturbation term that Proposition 3 does not cover.
"""
import json
import numpy as np

T = 25
SIZES = [20, 40, 80, 160, 320, 640, 1280, 2560]
N_SEEDS = 50
N_REFS = [5000, 20000, 80000]
sig = lambda x: 1 / (1 + np.exp(-x))


def simulate(n, s, rng):
    x = np.clip(np.array([0.3, 0.3, 0.4, 0.0]) + 0.05 * rng.standard_normal((n, 4)), 0, 1)
    k, e, p, d = x.T
    X, U = [np.stack([k, e, p, d], 1)], []
    for _ in range(T):
        u = rng.uniform(-1, 1, n)
        k1 = np.clip(0.85 * k + 0.15 * sig(3 * e - 1.5) + 0.20 * u - 0.06 * s + 0.03 * rng.standard_normal(n), 0, 1)
        e1 = np.clip(0.80 * e + 0.10 * np.tanh(p) + 0.15 * u + 0.03 * rng.standard_normal(n), 0, 1)
        p1 = sig(6 * (k1 - 0.5))
        d1 = 0.90 * d - 0.05 * s + 0.05 * u + 0.02 * rng.standard_normal(n)
        k, e, p, d = k1, e1, p1, d1
        X.append(np.stack([k, e, p, d], 1))
        U.append(u)
    return np.stack(X, 1), np.stack(U, 1)


def psi(X):
    k, e, p, d = (X[..., i] for i in range(4))
    return np.stack([k, e, p, d, k * k, e * e, d * d, k * e, d * e, k * d], -1)


def fit_K(X, U, mu, sd):
    Z = (psi(X[:, :-1]) - mu) / sd
    Zp = (psi(X[:, 1:]) - mu) / sd
    N = Z.shape[-1]
    Z, Zp, u = Z.reshape(-1, N), Zp.reshape(-1, N), U.reshape(-1, 1)
    Xa = np.hstack([Z, u, np.ones_like(u)])
    coef, *_ = np.linalg.lstsq(Xa, Zp, rcond=None)
    return coef[:N]  # rows of coef = A^T, so this is K = A^T


def scaling_and_metric(X):
    P = psi(X[:, :-1]).reshape(-1, 10)
    mu, sd = P.mean(0), P.std(0)
    Z = (P - mu) / sd
    G = Z.T @ Z / len(Z)
    Gamma = G + 1e-3 * np.linalg.eigvalsh(G).max() * np.eye(10)
    w, V = np.linalg.eigh(Gamma)
    return mu, sd, (V * np.sqrt(w)) @ V.T, (V / np.sqrt(w)) @ V.T


def DK(K0, K1, Gh, Gih):
    n = lambda L: np.linalg.norm(Gh @ L @ Gih)
    return 1 - n(K1 - K0) / (n(K0) + n(K1))


def slope(ns, errs_by_seed, rng, B=2000, kind="rmse"):
    ns = np.asarray(ns, float)
    E = np.asarray(errs_by_seed)            # (sizes, seeds) signed errors
    f = (lambda e: np.sqrt((e ** 2).mean(1))) if kind == "rmse" else (lambda e: e.std(1, ddof=1))
    est = np.polyfit(np.log(ns), np.log(f(E)), 1)[0]
    bs = []
    for _ in range(B):
        idx = rng.integers(0, E.shape[1], E.shape[1])
        bs.append(np.polyfit(np.log(ns), np.log(f(E[:, idx])), 1)[0])
    return float(est), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    out = {}
    ref_rng = np.random.default_rng(7)
    Xr, _ = simulate(20000, np.repeat([0, 1], 10000), ref_rng)
    mu, sd, Gh, Gih = scaling_and_metric(Xr)
    refs = {}
    for nref in N_REFS:
        rng = np.random.default_rng(1000 + nref)
        K = [fit_K(*simulate(nref, s, rng), mu, sd) for s in (0, 1)]
        refs[nref] = DK(K[0], K[1], Gh, Gih)
    out["D_ref"] = {str(k): float(v) for k, v in refs.items()}
    D_best = refs[max(N_REFS)]
    fixed, refit, fixed_raw = [], [], []
    for n in SIZES:
        ef, er, raw = [], [], []
        for sd_ in range(N_SEEDS):
            rng = np.random.default_rng(10_000 * n + sd_)
            X0, U0 = simulate(n, 0, rng)
            X1, U1 = simulate(n, 1, rng)
            D = DK(fit_K(X0, U0, mu, sd), fit_K(X1, U1, mu, sd), Gh, Gih)
            raw.append(D)
            ef.append(D - D_best)
            m2, s2, Gh2, Gih2 = scaling_and_metric(np.concatenate([X0, X1]))
            er.append(DK(fit_K(X0, U0, m2, s2), fit_K(X1, U1, m2, s2), Gh2, Gih2) - D_best)
        fixed.append(ef)
        refit.append(er)
        fixed_raw.append(raw)
    rng = np.random.default_rng(3)
    out["sizes"] = SIZES
    out["n_seeds"] = N_SEEDS
    out["fixed_metric"] = {
        "rmse": [float(np.sqrt(np.mean(np.square(e)))) for e in fixed],
        "sd": [float(np.std(e, ddof=1)) for e in fixed],
        "bias": [float(np.mean(e)) for e in fixed],
        "slope_rmse": slope(SIZES, fixed, rng, kind="rmse"),
        "slope_sd": slope(SIZES, fixed, rng, kind="sd"),
    }
    # floor: RMSE relative to each reference
    out["rmse_vs_reference"] = {
        str(nref): [float(np.sqrt(np.mean((np.array(r) - refs[nref]) ** 2))) for r in fixed_raw]
        for nref in N_REFS}
    out["slope_rmse_vs_reference"] = {
        str(nref): slope(SIZES, [np.array(r) - refs[nref] for r in fixed_raw], rng, kind="rmse")
        for nref in N_REFS}
    out["refit_metric"] = {
        "rmse": [float(np.sqrt(np.mean(np.square(e)))) for e in refit],
        "sd": [float(np.std(e, ddof=1)) for e in refit],
        "slope_rmse": slope(SIZES, refit, rng, kind="rmse"),
        "slope_sd": slope(SIZES, refit, rng, kind="sd"),
    }
    json.dump(out, open("../out/dk_rate_fixed_metric.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
