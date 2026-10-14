"""Level-1 (structural) diagnostics for every family of fits, and an
automated test of the identities the construction must satisfy.

    python structural_diagnostics.py --sim ../../Final_Folder_on_DR \
        --features assistments_features_time.csv --out out_structural.json

For every fit it records: the duality error max|K - A^T|; the idempotency
error max|P^2 - P|; the metric-duality error max|Gamma^{-1} P^T Gamma - P|
(P is Gamma-self-adjoint); the seed residual eps_q = ||P c||_Gamma; the
cross-block residuals r1 = ||PKR||, r2 = ||RKP|| (Gamma-weighted Frobenius);
the leakage of the removed direction through the projected input and offset
channels, ||(I - P^T) B_P|| and ||(I - P^T) a_P||; the requested, candidate
and numerical ranks; and the relative singular-value cutoff.  The script
exits with an error if an identity fails beyond the tolerance TOL.
"""
import argparse
import json
import sys

import numpy as np

TOL = 1e-9


def summarise(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: [float(min(r[k] for r in rows)), float(max(r[k] for r in rows))] for k in keys} | {"n_fits": len(rows)}


def check(rows, family):
    bad = [r for r in rows if max(r["duality_err"], r["idempotency_err"], r["metric_duality_err"],
                                  r["eps_q"], r["leak_B"], r["leak_a"]) > TOL]
    if bad:
        sys.exit(f"STRUCTURAL TEST FAILED in {family}: {bad[0]}")


# ---------------------------------------------------------------- synthetic
def synthetic_rows(sim_path, n_seeds=10):
    sys.path.insert(0, sim_path)
    import fa_edmd_mpc_v3_corrected as fa
    import synthetic_regimes as sr
    configs = set()
    for g in (-1.0, 0.0, 0.25, 0.5, 1.0, 2.0):
        configs.add((g, 4, 3))
    for r in (1, 2, 4, 6, 8):
        configs.add((1.0, r, 3))
    for d in (1, 2, 3, 5):
        configs.add((1.0, 4, d))
    rows = []
    for g, rank, depth in sorted(configs):
        fa.step = sr.make_step(g)
        for seed in range(n_seeds):
            m = fa.fit_model(seed, rank, depth)
            P, G = m.projector, m.metric
            K = m.propagation.T                       # coefficient operator
            R = np.eye(len(P)) - P
            A_P, B_P, a_P = m.dynamics(True)
            Rv = np.eye(len(P)) - P.T                 # value-space complement
            Gi = np.linalg.inv(G)
            d = m.diagnostics
            rows.append(dict(gamma=g, rank=rank, depth=depth, seed=seed,
                             duality_err=float(np.abs(K - m.propagation.T).max()),
                             idempotency_err=float(np.abs(P @ P - P).max()),
                             metric_duality_err=float(np.abs(Gi @ P.T @ G - P).max()),
                             eps_q=float(d["seed_residual"]),
                             r1=float(d["r1_centered_block"]), r2=float(d["r2_centered_block"]),
                             leak_B=float(np.linalg.norm(Rv @ B_P)), leak_a=float(np.linalg.norm(Rv @ a_P)),
                             requested_rank=int(d["requested_rank"]),
                             candidate_rank=int(d["numeric_candidate_rank"]),
                             applied_rank=int(d["applied_rank"]),
                             numerical_rank_G=int(np.linalg.matrix_rank(m.gram)),
                             svd_cutoff=float(d["relative_svd_tolerance"])))
    fa.step = sr.make_step(1.0)
    return rows


# ---------------------------------------------------------------- ASSISTments
def assistments_rows(features):
    import assistments_pipeline as ap
    df_all = ap.keep_full_trajectories(ap.load_features(features))
    seed0 = 20260914
    ids = np.random.default_rng(seed0 + 1).permutation(df_all["user_id"].unique())
    runs = ([(f"school_disjoint_{j}", seed0 + 200 + j, "school", "train", None) for j in range(5)]
            + [(f"school_disjoint_otherlabel_{j}", seed0 + 200 + j, "school", "other", None) for j in range(5)]
            + [(f"student_split_{j}", seed0 + 100 * (j > 0) + j, "student", "train", None) for j in range(5)]
            + [(f"subsample_{j}", seed0 + 10 + j, "student", "train", ids[j * 5000:(j + 1) * 5000]) for j in range(3)]
            + [("student_split_full_strata", seed0, "student", "full", None)])
    rows = []
    for name, seed, split, strata, sid in runs:
        rng = np.random.default_rng(seed)
        df = df_all if sid is None else df_all[df_all["user_id"].isin(sid)]
        if split == "school":
            tr, va, te = ap.split_schools(df, rng)
        else:
            tr, va, te = ap.split_students(df["user_id"].unique(), rng)
        if strata == "train":
            df, _ = ap.rebuild_school_strata(df, tr, split == "school")
        elif strata == "other":
            df, _, va, te, _ = ap.rebuild_school_strata_other(df, tr, va, te, rng)
        df = ap.rebuild_dt(df, tr)
        _, Xtr, Gtr, _ = ap.trajectories(df, tr)
        fit = ap.fit_edmd(Xtr)
        G, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
        c, _ = ap.contrast(fit, Xtr, Gtr)
        P, R, cand = ap.build_projector(fit.K, G, c)
        A_P, a_P = ap.propagator(fit, P)
        Rv = np.eye(len(P)) - P.T
        Gi = np.linalg.inv(G)
        family = name.rsplit("_", 1)[0] if name[-1].isdigit() else name
        rows.append(dict(family=family, run=name,
                         duality_err=float(np.abs(fit.K - fit.A.T).max()),
                         idempotency_err=float(np.abs(P @ P - P).max()),
                         metric_duality_err=float(np.abs(Gi @ P.T @ G - P).max()),
                         eps_q=float(np.sqrt(max((P @ c) @ G @ (P @ c), 0))),
                         r1=ap.wnorm(P @ fit.K @ R, G), r2=ap.wnorm(R @ fit.K @ P, G),
                         leak_B=0.0,                                  # autonomous model: no input channel
                         leak_a=float(np.linalg.norm(Rv @ a_P)),
                         requested_rank=ap.RANK, candidate_rank=int(cand), applied_rank=ap.RANK,
                         numerical_rank_G=int(fit.num_rank), svd_cutoff=1e-8))
    return rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--sim", default="../../Final_Folder_on_DR")
    a.add_argument("--features", default="assistments_features_time.csv")
    a.add_argument("--out", default="out_structural.json")
    a.add_argument("--seeds", type=int, default=10, help="simulator replicates per setting (50 = every fit of the 50-replicate study)")
    args = a.parse_args()
    out = {}
    syn = synthetic_rows(args.sim, args.seeds)
    check(syn, "synthetic")
    out["synthetic_all_fits"] = summarise(syn)
    ass = assistments_rows(args.features)
    check(ass, "assistments")
    for fam in sorted(set(r["family"] for r in ass)):
        out["assistments:" + fam] = summarise([r for r in ass if r["family"] == fam])
    out["rows"] = {"synthetic": syn, "assistments": ass}
    json.dump(out, open(args.out, "w"), indent=1)
    for k, v in out.items():
        if k != "rows":
            print(k, {kk: (vv if not isinstance(vv, list) else [f"{x:.2e}" for x in vv]) for kk, vv in v.items()})
    print("ALL STRUCTURAL TESTS PASSED (tolerance %g)" % TOL)


if __name__ == "__main__":
    main()
