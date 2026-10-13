"""
Shuffled-label control at the level of the school for ASSISTments (referee report 5, item 8).

The ASSISTments group is a property of the school, so shuffling labels between students breaks the dependence
structure. Here the training schools' labels are permuted between schools (the number of schools in each group
is kept), every student takes the label of the school, the contrast is refitted and the rank-4 projection is
built with the paper's construction. 50 draws per split, the five school-disjoint splits of the paper.

    python school_permutation.py     -> out_rank_matched/school_permutation.json (aggregates only)
"""
import json
import os

import numpy as np

import assistments_pipeline as ap
import rank_matched_controls as rm

N_DRAWS = 50


def one_split(df, seed):
    _, Xtr, Gtr, Str, Xte, Gte, _ = rm.prepare(df, seed, "school", "train")
    fit = ap.fit_edmd(Xtr)
    Gamma, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
    raw = rm.LinPredictor(fit, fit.A, fit.a)
    raw_term = raw.predict(Xte[:, 0], ap.N_TRANS)[:, 0]
    mse = lambda P: rm.pred_metrics(rm.LinPredictor(fit, P.T @ fit.A @ P.T, P.T @ fit.a), Xte, Gte, raw_term)[0]["mse_k_h1"]
    c, _ = ap.contrast(fit, Xtr, Gtr)
    group = mse(ap.build_projector(fit.K, Gamma, c, rank=4)[0])
    schools = np.unique(Str)
    lab = {s: int(Gtr[Str == s][0]) for s in schools}
    rng = np.random.default_rng(seed + 11)
    draws_school, draws_student = [], []
    for _ in range(N_DRAWS):
        perm = dict(zip(schools, rng.permutation([lab[s] for s in schools])))
        Gp = np.array([perm[s] for s in Str])
        cp, _ = ap.contrast(fit, Xtr, Gp)
        draws_school.append(mse(ap.build_projector(fit.K, Gamma, cp, rank=4)[0]))
        cs, _ = ap.contrast(fit, Xtr, rng.permutation(Gtr))
        draws_student.append(mse(ap.build_projector(fit.K, Gamma, cs, rank=4)[0]))
    return dict(raw=float(rm.pred_metrics(raw, Xte, Gte, raw_term)[0]["mse_k_h1"]), group=float(group),
                school_perm_mean=float(np.mean(draws_school)), school_perm_q=[float(np.percentile(draws_school, q)) for q in (2.5, 50, 97.5)],
                student_perm_mean=float(np.mean(draws_student)),
                group_percentile_in_school_perm=float((np.array(draws_school) < group).mean()),
                n_schools=int(len(schools)))


def main():
    df = ap.keep_full_trajectories(ap.load_features("assistments_features_time.csv"))
    runs = [one_split(df, 20260914 + 200 + j) for j in range(5)]
    for j, r in enumerate(runs):
        print(f"split {j}: raw {r['raw']:.4f} group {r['group']:.4f} school-perm {r['school_perm_mean']:.4f} {[round(x, 4) for x in r['school_perm_q']]} "
              f"student-perm {r['student_perm_mean']:.4f} group above {100 * r['group_percentile_in_school_perm']:.0f}% of school draws", flush=True)
    m = lambda k: float(np.mean([r[k] for r in runs]))
    summ = {k: m(k) for k in ("raw", "group", "school_perm_mean", "student_perm_mean", "group_percentile_in_school_perm")}
    print(summ)
    json.dump({"draws": N_DRAWS, "splits": runs, "summary": summ}, open(os.path.join(rm.OUT, "school_permutation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
