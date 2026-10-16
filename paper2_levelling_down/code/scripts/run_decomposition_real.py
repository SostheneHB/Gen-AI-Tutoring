"""Experiment 05: where does the gap come from in real learning logs (Eedi, OULAD)?

    python code/scripts/run_decomposition_real.py

No teaching control is observed in these logs, so the terms R and P of Proposition 2 are
zero and the gap splits into the initial state I, the persistence A and the drift a
(Shapley over the 3! orders).  Lifting and decoder as in the companion paper [A] (fit on training students
only); one model per group on training students; initial states and observed gaps on
held-out students.  Five splits (class-disjoint for Eedi, student split for OULAD).
Robustness: least squares shrinks A towards 0 when the states are noisy (attenuation), which
shortens the memory of the initial state and moves weight to the drift.  Every split is also
fitted with z_{t-1} as an instrument for z_t ("iv").
Writes outputs/05_decomposition_real/results.json (aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.decomposition import FREE_FACTORS, fit_free_group_models, fit_free_group_models_iv, free_decomposition  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import assistments_pipeline as ap  # noqa: E402

T = ap.N_TRANS
OUT = {"k": 0, "d": 3}                       # final knowledge proxy k_T and residual correctness d_T
DATASETS = {"eedi_premium": ("eedi_features", "S", "school", 20260914 + 200),
            "oulad_deprivation": ("oulad_features", "S_imd", "student", 20260914 + 300),
            "oulad_disability": ("oulad_features", "S_disability", "student", 20260914 + 300)}


def one_split(df, split, seed):
    rng = np.random.default_rng(seed)
    if split == "school":
        tr, _, te = ap.split_schools(df, rng)
    else:
        tr, _, te = ap.split_students(df["user_id"].unique(), rng)
    _, Xtr, Gtr, _ = ap.trajectories(df, tr)
    _, Xte, Gte, _ = ap.trajectories(df, te)
    fit = ap.fit_edmd(Xtr)                                    # shared lifting, scaling and decoder
    Z = ap.encode(fit, Xtr[:, :-1]).reshape(-1, fit.A.shape[0])
    Zn = ap.encode(fit, Xtr[:, 1:]).reshape(-1, fit.A.shape[0])
    fits = {"ls": fit_free_group_models(Z, Zn, np.repeat(Gtr, T)), "iv": fit_free_group_models_iv(ap.encode(fit, Xtr), Gtr)}
    Z0 = ap.encode(fit, Xte[:, 0])
    m0 = {g: Z0[Gte == g].mean(0) for g in (0, 1)}
    res = {}
    for est, models in fits.items():
        r = dict(rho_A={str(g): float(np.abs(np.linalg.eigvals(models[g]["A"])).max()) for g in (0, 1)})
        for name, idx in OUT.items():
            dec = free_decomposition(models, m0, T, fit.C[idx], fit.c[idx])
            obs = float(Xte[Gte == 1, T, idx].mean() - Xte[Gte == 0, T, idx].mean())
            init = float(Xte[Gte == 1, 0, idx].mean() - Xte[Gte == 0, 0, idx].mean())
            r[name] = dict(observed=obs, observed_initial=init, model=dec["gap"], error=obs - dec["gap"],
                           shapley=dec["shapley"], sequential=dec["sequential"])
        res[est] = r
    return res


def summarise(vals):
    v = np.asarray(vals, float)
    h = 2.776 * v.std(ddof=1) / np.sqrt(len(v))              # t(0.975, 4)
    return dict(mean=float(v.mean()), ci=[float(v.mean() - h), float(v.mean() + h)])


def main():
    out = {}
    cache = {}
    for name, (feat, col, split, seed) in DATASETS.items():
        if feat not in cache:
            cache[feat] = ap.keep_full_trajectories(ap.load_features(str(PATHS[feat])))
        df = cache[feat].copy()
        df["G"] = df[col].astype(int)
        all_runs = [one_split(df, split, seed + j) for j in range(5)]
        out[name] = dict(n_students=int(df.user_id.nunique()), group_share=float(df.drop_duplicates("user_id").G.mean()))
        for est in ("ls", "iv"):
            runs = [r[est] for r in all_runs]
            out[name][est] = dict(spectral_radius={g: summarise([r["rho_A"][g] for r in runs]) for g in ("0", "1")},
                                  per_split=[{o: dict(model=r[o]["model"], observed=r[o]["observed"],
                                                      shapley=r[o]["shapley"]) for o in OUT} for r in runs])
            for o in OUT:
                e = {k: summarise([r[o][k] for r in runs]) for k in ("observed", "observed_initial", "model", "error")}
                e["shapley"] = {f: summarise([r[o]["shapley"][f] for r in runs]) for f in FREE_FACTORS}
                e["sequential"] = {f: summarise([r[o]["sequential"][f] for r in runs]) for f in FREE_FACTORS}
                out[name][est][o] = e
                sh = e["shapley"]
                print(f"{name:18s} {est} {o}: observed gap {e['observed']['mean']:+.4f} (at start {e['observed_initial']['mean']:+.4f}) "
                      f"model {e['model']['mean']:+.4f} | Shapley I {sh['I']['mean']:+.4f} {np.round(sh['I']['ci'], 4)} "
                      f"A {sh['A']['mean']:+.4f} {np.round(sh['A']['ci'], 4)} a {sh['a']['mean']:+.4f} {np.round(sh['a']['ci'], 4)} "
                      f"| error {e['error']['mean']:+.4f}", flush=True)
    d = ROOT / "outputs" / "05_decomposition_real"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
