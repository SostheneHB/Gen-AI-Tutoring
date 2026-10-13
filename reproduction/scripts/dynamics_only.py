"""
Dynamics-only correction (referee report 2, item 5), on the same splits as the rank-matched controls.

Group-aware model, in the encoded EDMD features z of the raw fit and with the centred group gc = g - p1:
        z_{t+1} = A z_t + a + gc (A_G z_t + a_G)          (least squares on the training students)
Dynamics-only correction: the same fit, used with gc = 0, i.e. z_{t+1} = A z_t + a.  Two students in the same
state then receive the same forecast ("equal treatment given the state"), but the state itself is untouched,
so the part of the gap that the state carries is kept.

Compared arms: raw EDMD, projected EDMD (rank 4, the paper's construction), group-aware, dynamics-only.
Reported on held-out students: MSE of knowledge at h = 1 and 10 (overall, worst group), the free-running
predicted terminal gap against the observed one, the one-step difference between the forecasts for g = 1 and
g = 0 at the same state (group-aware model), and the share of the group-aware free-running gap that the
dynamics-only model keeps (the state part).

    DATA_ROOT=/path/to/root python dynamics_only.py      -> out_rank_matched/dynamics_only.json (aggregates only)
"""
import json
import os

import numpy as np

import assistments_pipeline as ap
import rank_matched_controls as rm

ROOT = os.environ.get("DATA_ROOT", ".")


class GroupAware:
    def __init__(self, fit, Xtr, Gtr):
        Z = ap.encode(fit, Xtr)
        cur, nxt = Z[:, :-1].reshape(-1, Z.shape[-1]), Z[:, 1:].reshape(-1, Z.shape[-1])
        self.p1 = float(Gtr.mean())
        gc = np.repeat(Gtr - self.p1, ap.N_TRANS)[:, None]
        X = np.hstack([cur, np.ones((len(cur), 1)), gc * cur, gc])
        coef, *_ = np.linalg.lstsq(X, nxt, rcond=None)
        d = cur.shape[1]
        self.A, self.a = coef[:d].T, coef[d]
        self.AG, self.aG = coef[d + 1:2 * d + 1].T, coef[2 * d + 1]
        self.fit, self.use_group = fit, True

    def step(self, z, gc):
        out = z @ self.A.T + self.a
        if self.use_group:
            out = out + gc[:, None] * (z @ self.AG.T + self.aG)
        return out

    def predict(self, X0, h, g=None):
        z = ap.encode(self.fit, X0)
        gc = (np.zeros(len(z)) if g is None else g - self.p1)
        for _ in range(h):
            z = self.step(z, gc)
        return z @ self.fit.C.T + self.fit.c


def metrics(model, Xte, Gte):
    m = {}
    for h in (1, 10):
        e = ap.horizon_errors(model, Xte, Gte, h)[:, 0]
        m[f"mse_k_h{h}"] = float(e.mean())
        m[f"mse_k_h{h}_worst_group"] = float(max(e[Gte == 0].mean(), e[Gte == 1].mean()))
    term = model.predict(Xte[:, 0], ap.N_TRANS, Gte)[:, 0]
    m["free_run_gap"] = ap.gap(term, Gte)
    pk = model.predict(Xte[:, -2], 1, Gte)[:, 0]
    m["replay_gap_abs_error"] = abs(ap.gap(pk, Gte) - ap.gap(Xte[:, -1, 0], Gte))
    return m


def one_split(df, seed, split, strata):
    _, Xtr, Gtr, _, Xte, Gte, _ = rm.prepare(df, seed, split, strata)
    fit = ap.fit_edmd(Xtr)
    Gamma, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
    c, _ = ap.contrast(fit, Xtr, Gtr)
    P = ap.build_projector(fit.K, Gamma, c, rank=4)[0]
    ga = GroupAware(fit, Xtr, Gtr)
    do = GroupAware(fit, Xtr, Gtr); do.use_group = False
    arms = {"raw": rm.LinPredictor(fit, fit.A, fit.a), "projected": rm.LinPredictor(fit, P.T @ fit.A @ P.T, P.T @ fit.a),
            "group_aware": ga, "dynamics_only": do}
    out = {"observed_terminal_gap": ap.gap(Xte[:, -1, 0], Gte), "arms": {k: metrics(v, Xte, Gte) for k, v in arms.items()}}
    starts = Xte[:, :-1].reshape(-1, 4)
    out["ga_same_state_diff_k"] = float(np.abs(ga.predict(starts, 1, np.ones(len(starts)))[:, 0]
                                              - ga.predict(starts, 1, np.zeros(len(starts)))[:, 0]).mean())
    out["state_share_of_ga_gap"] = out["arms"]["dynamics_only"]["free_run_gap"] / out["arms"]["group_aware"]["free_run_gap"]
    rel = lambda M: np.linalg.norm(M) / np.linalg.norm(ga.A)
    out["relative_size_AG"] = float(rel(ga.AG))
    return out


def main():
    seed = 20260914
    res = {}
    df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv"))
    res["assistments"] = [one_split(df, seed + 200 + j, "school", "train") for j in range(5)]
    df = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/Eedi_NeurIPS2020/eedi_features_time.csv"))
    res["eedi"] = [one_split(df, seed + 200 + j, "school", "given") for j in range(5)]
    base = ap.keep_full_trajectories(ap.load_features(f"{ROOT}/Datasets/OULAD/oulad_features_weekly.csv"))
    for att in ("imd", "disability", "gender"):
        d = base.copy(); d["S"] = d[f"S_{att}"]; d["G"] = d["S"].astype(int)
        res[f"oulad_{att}"] = [one_split(d, seed + 300 + j, "student", "given") for j in range(5)]
    summ = {}
    for k, runs in res.items():
        s = {"observed_terminal_gap": float(np.mean([r["observed_terminal_gap"] for r in runs])),
             "ga_same_state_diff_k": float(np.mean([r["ga_same_state_diff_k"] for r in runs])),
             "state_share_of_ga_gap": float(np.mean([r["state_share_of_ga_gap"] for r in runs])),
             "relative_size_AG": float(np.mean([r["relative_size_AG"] for r in runs]))}
        for arm in runs[0]["arms"]:
            s[arm] = {q: float(np.mean([r["arms"][arm][q] for r in runs])) for q in runs[0]["arms"][arm]}
        summ[k] = s
        print(f"== {k}: observed gap {s['observed_terminal_gap']:+.4f}, same-state diff {s['ga_same_state_diff_k']:.4f}, "
              f"state share {s['state_share_of_ga_gap']:.2f}, |A_G|/|A| {s['relative_size_AG']:.3f}")
        for arm in runs[0]["arms"]:
            a = s[arm]
            print(f"   {arm:14s} mse1 {a['mse_k_h1']:.4f} (worst {a['mse_k_h1_worst_group']:.4f}) mse10 {a['mse_k_h10']:.4f} "
                  f"free gap {a['free_run_gap']:+.4f} replay err {a['replay_gap_abs_error']:.4f}")
    os.makedirs(rm.OUT, exist_ok=True)
    json.dump({"splits": res, "summary": summ}, open(os.path.join(rm.OUT, "dynamics_only.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
