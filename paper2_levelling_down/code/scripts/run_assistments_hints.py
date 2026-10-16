"""Experiment 16: a learned model with a real control — hints in ASSISTments 2012-2013 (exploratory).

    python code/scripts/extract_assistments_hints.py --raw <raw csv>      # once, local cache in data/
    python code/scripts/run_assistments_hints.py [--splits 20]

State: the the companion paper [A] features (k, e, p, d) at steps 0..25, groups = school-performance strata rebuilt from training
students on the label window (steps > 25), school-disjoint splits, as in the companion paper [A].  Control: u_t = 1 if the
student asked for at least one hint on the problem of step t.  In ASSISTments a hint marks the problem wrong, and
k_t already contains the answer of step t, so the mechanical effect of u_t is in x_t; the model x_{t+1} =
A_g x_t + B_g u_t + a_g (lifted, least squares per group) gives B_g the effect of the hint on what follows.
Target and protected outcome: final knowledge k_25 (D = K, as in the main experiments).

Hints are chosen by the students, so B_g is not a causal effect.  Placebo: the coefficient of u_{t+2} (a later
choice) in the regression of z_{t+1}; it should be zero if the hint choices carried no information on the state
beyond x_t.  Per split (training schools 60%, evaluation 40%) the script reports:
  * the sign of the Markov parameters h^k_{g,t} (does help raise final knowledge in the model?),
  * the observed and the predicted knowledge gap on evaluation schools under the observed hint use,
  * (A1)-(A5) at the efficient reference with the observed budget, the capacity, the price of half of it,
  * the placebo coefficient (decoded on k) next to the hint coefficient.
Writes outputs/16_assistments_hints/results.json (aggregate numbers only).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from p2.decomposition import fit_group_models  # noqa: E402
from p2.frontier import GroupModel, efficient_plan, knapsack_price  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402
from run_assumption_cases import assumptions  # noqa: E402

use_paper1_code()
import assistments_pipeline as ap  # noqa: E402

T = ap.N_TRANS
KIDX = ap.FEATURES.index("k_t")
HINTS = ROOT / "data" / "ASSISTments_2012_2013" / "hints_by_log.parquet"
SEED = 20261016


def trajectories(df, ids):
    sub = df[df.user_id.isin(ids) & (df.step <= T)].sort_values(["user_id", "step"])
    X = sub[ap.FEATURES].to_numpy(float).reshape(-1, T + 1, 4)
    U = sub.u.to_numpy(float).reshape(-1, T + 1)[:, :T]
    G = sub.groupby("user_id", sort=True).G.first().to_numpy(int)
    return X, U, G


def one(df_all, seed):
    rng = np.random.default_rng(seed)
    tr_ids, va_ids, te_ids = ap.split_schools(df_all, rng)
    ev_ids = np.r_[va_ids, te_ids]
    df, _ = ap.rebuild_school_strata(df_all, tr_ids, True)
    df = ap.rebuild_dt(df, tr_ids)
    Xtr, Utr, Gtr = trajectories(df, tr_ids)
    Xev, Uev, Gev = trajectories(df, ev_ids)
    fit = ap.fit_edmd(Xtr)
    Z = ap.encode(fit, Xtr)
    n = Z.shape[-1]
    cur, nxt = Z[:, :-1].reshape(-1, n), Z[:, 1:].reshape(-1, n)
    G_ = np.repeat(Gtr, T)
    mods = fit_group_models(cur, Utr.reshape(-1), nxt, G_)
    ck, ck0 = fit.C[KIDX], fit.c[KIDX]
    # placebo: z_{t+1} on z_t, u_t, u_{t+2}, t = 0..T-3
    plac = {}
    for g in (0, 1):
        m = Gtr == g
        Zc, Zn = Z[m, :T - 2].reshape(-1, n), Z[m, 1:T - 1].reshape(-1, n)
        u0, u2 = Utr[m, :T - 2].reshape(-1), Utr[m, 2:T].reshape(-1)
        C = np.linalg.lstsq(np.column_stack([Zc, u0, u2, np.ones(len(Zc))]), Zn, rcond=None)[0]
        plac[g] = dict(hint=float(ck @ C[n]), later_hint=float(ck @ C[n + 1]))
    # model on evaluation schools
    Z0 = ap.encode(fit, Xev[:, 0])
    m0 = {g: Z0[Gev == g].mean(0) for g in (0, 1)}
    nn = {g: int((Gev == g).sum()) for g in (0, 1)}
    M = GroupModel.separate(mods, m0, ck, ck0, ck, ck0, nn, T)
    v_obs = {g: Uev[Gev == g].mean(0) for g in (0, 1)}
    K_obs, _, gap_pred, B_obs = M.outcomes(v_obs)
    gap_obs = float(Xev[Gev == 1, T, KIDX].mean() - Xev[Gev == 0, T, KIDX].mean())
    v_ref = efficient_plan(M, B_obs)
    _, _, gap_ref, _ = M.outcomes(v_ref)
    a_ = assumptions(M, v_ref)
    cap = knapsack_price(M, v_ref, 0.0)[1]
    target = 0.5 * min(cap, abs(gap_ref))
    price = knapsack_price(M, v_ref, target)[0] / B_obs if target > 0 else None
    hk = {g: M.hk[g] for g in (0, 1)}
    return dict(seed=seed, n_train=int(len(Xtr)), n_eval=int(len(Xev)), p1_eval=float(Gev.mean()),
                hint_rate={str(g): float(Uev[Gev == g].mean()) for g in (0, 1)},
                hk_sum={str(g): float(hk[g].sum()) for g in (0, 1)}, hk_share_positive={str(g): float((hk[g] > 0).mean()) for g in (0, 1)},
                hk_min={str(g): float(hk[g].min()) for g in (0, 1)}, hk_max={str(g): float(hk[g].max()) for g in (0, 1)},
                gap_obs=gap_obs, gap_pred_obs_hints=float(gap_pred), gap_ref=float(gap_ref),
                K_obs={str(g): float(K_obs[g]) for g in (0, 1)}, K_obs_true={str(g): float(Xev[Gev == g, T, KIDX].mean()) for g in (0, 1)},
                assumptions=a_, all_assumptions=bool(all(a_.values())), capacity=float(cap), target=float(target),
                price_half=None if price is None or not np.isfinite(price) else float(price),
                placebo={str(g): plac[g] for g in (0, 1)})


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--splits", type=int, default=20)
    a = a_.parse_args()
    df = ap.keep_full_trajectories(ap.load_features(str(PATHS["assistments_features"])))
    h = pd.read_parquet(HINTS)
    df = df.merge(h[["problem_log_id", "hint_count"]], on="problem_log_id", how="left")
    print(f"students {df.user_id.nunique()}, schools {df.school_id.nunique()}, logs without hint information "
          f"{df.hint_count.isna().mean():.4f}", flush=True)
    df["u"] = (df.hint_count.fillna(0) > 0).astype(float)
    rows = []
    for j in range(a.splits):
        r = one(df, SEED + j)
        rows.append(r)
        print(f"split {j}: gap obs {r['gap_obs']:+.3f} pred {r['gap_pred_obs_hints']:+.3f} ref {r['gap_ref']:+.3f} | "
              f"sum h^k {r['hk_sum']['0']:+.3f}/{r['hk_sum']['1']:+.3f} pos {r['hk_share_positive']['0']:.2f}/{r['hk_share_positive']['1']:.2f} | "
              f"A {r['assumptions']} cap {r['capacity']:.4f} price {r['price_half']} | placebo k: hint "
              f"{r['placebo']['0']['hint']:+.4f} later {r['placebo']['0']['later_hint']:+.4f}", flush=True)
    q = lambda v: [float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95))]
    summ = dict(splits=len(rows), n_students=int(df.user_id.nunique()), gap_obs=q([r["gap_obs"] for r in rows]),
                gap_pred_obs_hints=q([r["gap_pred_obs_hints"] for r in rows]), gap_ref=q([r["gap_ref"] for r in rows]),
                hint_rate={g: q([r["hint_rate"][g] for r in rows]) for g in ("0", "1")},
                hk_sum={g: q([r["hk_sum"][g] for r in rows]) for g in ("0", "1")},
                hk_share_positive={g: q([r["hk_share_positive"][g] for r in rows]) for g in ("0", "1")},
                assumptions_hold=float(np.mean([r["all_assumptions"] for r in rows])),
                each_assumption={k: float(np.mean([r["assumptions"][k] for r in rows])) for k in ("A1", "A2", "A3", "A4", "A5")},
                capacity=q([r["capacity"] for r in rows]),
                price_half=q([r["price_half"] for r in rows if r["price_half"] is not None]) if any(r["price_half"] is not None for r in rows) else None,
                placebo={g: {k: q([r["placebo"][g][k] for r in rows]) for k in ("hint", "later_hint")} for g in ("0", "1")})
    print(json.dumps(summ, indent=1))
    d = ROOT / "outputs" / "16_assistments_hints"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summ, rows=rows), open(d / "results.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
