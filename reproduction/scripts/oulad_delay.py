"""
OULAD: sensitivity to the time at which an assessment score becomes known (review of 13 October 2026, item 5).

In the main features a score enters the state k_t in the week it was submitted.  Here a score submitted in week w
becomes known only in week w + delay, for delay in {0, 1, 2, 4} weeks; since k_t is the running mean of the scores
submitted up to week t, the delayed proxy is k_{t - delay} (0.5 before), and p_t = 1{k_t >= 0.4} is recomputed.
For each delay, attribute (deprivation, disability, gender) and the five student splits of the main study:
  * observed gap of k at week 25 on held-out students, and the one-step predicted gaps of the original EDMD model,
    the group-aware model and the projected model;
  * one-step error on k of the projected model divided by that of the original model, over all transitions and
    separately over the transitions in which a new score becomes known (k_{t+1} != k_t) and the others.

    DATA_ROOT=/path/to/root python oulad_delay.py      -> out_reviewer/oulad_delay.json (aggregates only)
"""
import json
import os

import numpy as np

import assistments_pipeline as ap
import rank_matched_controls as rm
from dynamics_only import GroupAware

ROOT = os.environ.get("DATA_ROOT", ".")
DELAYS = (0, 1, 2, 4)
ATTS = ("imd", "disability", "gender")
SEED = 20260914


def delayed(df, delay):
    df = df.sort_values(["user_id", "step"]).copy()
    if delay:
        df["k_t"] = df.groupby("user_id")["k_t"].shift(delay).fillna(0.5)
        df["p_t"] = (df["k_t"] >= 0.4).astype(int)
    return df


def gap(v, g):
    return float(v[g == 1].mean() - v[g == 0].mean())


def one_split(df, seed):
    _, Xtr, Gtr, _, Xte, Gte, _ = rm.prepare(df, seed, "student", "given")
    fit = ap.fit_edmd(Xtr)
    Gamma, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
    c, _ = ap.contrast(fit, Xtr, Gtr)
    P = ap.build_projector(fit.K, Gamma, c, rank=4)[0]
    models = {"original": rm.LinPredictor(fit, fit.A, fit.a),
              "projected": rm.LinPredictor(fit, P.T @ fit.A @ P.T, P.T @ fit.a),
              "group_aware": GroupAware(fit, Xtr, Gtr)}
    T = ap.N_TRANS
    cur = Xte[:, :-1].reshape(-1, Xte.shape[-1])
    nxt = Xte[:, 1:, 0].reshape(-1)
    g = np.repeat(Gte, T)
    new = (Xte[:, 1:, 0] != Xte[:, :-1, 0]).reshape(-1)
    out = dict(observed_gap=gap(Xte[:, -1, 0], Gte), share_new_score=float(new.mean()))
    err = {}
    for name, m in models.items():
        pred = m.predict(cur, 1, g)[:, 0]
        e = (pred - nxt) ** 2
        err[name] = dict(all=float(e.mean()), new=float(e[new].mean()), other=float(e[~new].mean()))
        out[f"gap_{name}"] = gap(m.predict(Xte[:, -2], 1, Gte)[:, 0], Gte)
    out["mse"] = err
    out["ratio"] = {k: err["projected"][k] / err["original"][k] for k in ("all", "new", "other")}
    return out


def main():
    base = ap.keep_full_trajectories(ap.load_features(os.path.join(ROOT, "Datasets", "OULAD", "oulad_features_weekly.csv")))
    res = {}
    for delay in DELAYS:
        dfd = delayed(base, delay)
        for att in ATTS:
            df = dfd.copy()
            df["S"] = df[f"S_{att}"]
            df["G"] = df["S"].astype(int)
            runs = [one_split(df, SEED + 300 + j) for j in range(5)]
            mean = lambda f: float(np.mean([f(r) for r in runs]))
            rng_ = lambda f: [float(min(f(r) for r in runs)), float(max(f(r) for r in runs))]
            res[f"{att}|{delay}"] = dict(
                attribute=att, delay=delay, splits=runs,
                observed_gap=mean(lambda r: r["observed_gap"]),
                gap_original=mean(lambda r: r["gap_original"]), gap_group_aware=mean(lambda r: r["gap_group_aware"]),
                gap_projected=mean(lambda r: r["gap_projected"]), share_new_score=mean(lambda r: r["share_new_score"]),
                ratio_all=rng_(lambda r: r["ratio"]["all"]), ratio_new=rng_(lambda r: r["ratio"]["new"]),
                ratio_other=rng_(lambda r: r["ratio"]["other"]),
                mse_original_all=mean(lambda r: r["mse"]["original"]["all"]))
            s = res[f"{att}|{delay}"]
            print(f"delay {delay} {att:10s} observed {s['observed_gap']:+.4f} original {s['gap_original']:+.4f} "
                  f"group-aware {s['gap_group_aware']:+.4f} projected {s['gap_projected']:+.4f} | ratio all {s['ratio_all']} "
                  f"new-score weeks {s['ratio_new']} other {s['ratio_other']} (share new {s['share_new_score']:.3f})", flush=True)
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(res, open("out_reviewer/oulad_delay.json", "w"), indent=1)


if __name__ == "__main__":
    main()
