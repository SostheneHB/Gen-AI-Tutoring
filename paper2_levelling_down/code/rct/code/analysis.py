"""
Policy evaluation on the ASSISTments randomized experiments, as fixed in notes/analysis_plan.md.

    python analysis.py --sim effect     # simulated outcomes with a known effect (pipeline test)
    python analysis.py --sim null       # simulated outcomes with no effect (false-positive check)
    python analysis.py --real           # real outcomes: refused unless notes/APPROVED exists

In simulation mode the real outcome columns are never read.  Policies (plan, Section 5):
  P0 control for all, P1 treatment for all, P2 best arm (reference), P3 best arm per group,
  P4 personalised (T-learner), P5 gap-aware with floors (low group gets its best arm; a group whose
  estimated value under P5 falls below its value under P2 on the training folds is reset to P2).
Values: doubly robust with the known assignment probability 1/2, cross-fitted over 5 folds of classes.
Uncertainty and pooling: student-weighted values over all experiments, joint cluster bootstrap of classes within experiments.
Outputs (aggregates only): outputs/policy_<mode>.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "outputs")
COV = ["student_prior_assignments_started", "student_prior_assignments_percent_completed",
       "student_prior_average_problem_correctness", "student_prior_average_problem_attempt_count",
       "student_prior_average_problem_hint_count", "student_prior_median_ln_problem_time_on_task"]
POLICIES = ["P0", "P1", "P2", "P3", "P4", "P5"]
P = 0.5
GROUPS = ("low", "high", "all")


def load(real, sens=None):
    """Sample of the plan (rules R1-R4) and groups (R5).  Outcome read only if real."""
    from scipy import stats
    c = pd.read_csv(os.path.join(D, "experiment_conditions.csv"), index_col=0)
    cols = ["assignment_log_id", "class_id", "student_id", "assignment_start_time", "target_sequence"] + COV
    if real:
        cols.append("assignment_completed")
    s = pd.read_csv(os.path.join(D, "experiment_prior_stats_and_targets.csv"), usecols=cols)
    m = s.merge(c, on="assignment_log_id")
    m = m[(m.in_control + m.in_treatment) == 1]
    m = m.sort_values("assignment_start_time").drop_duplicates(["target_sequence", "student_id"])
    per = m.groupby("target_sequence").agg(n=("student_id", "size"), t=("in_treatment", "sum"))
    per["p"] = [stats.binomtest(int(t), int(n), .5).pvalue for t, n in zip(per.t, per.n)]
    size = (per.t >= 56) & (per.n - per.t >= 56)
    keep = per.index[size] if sens == "propensity" else per.index[(per.p >= .1) & size]   # plan, sensitivity 1
    m = m[m.target_sequence.isin(keep)].copy()
    m["pA"] = 0.5 if sens != "propensity" else m.target_sequence.map(per.t / per.n)
    if sens == "min10classes":                                                   # plan, cluster sensitivity
        k = m.groupby("target_sequence").class_id.nunique()
        m = m[m.target_sequence.isin(k.index[k >= 10])].copy()
    x = (m.student_prior_assignments_percent_completed if sens == "groups_completion"     # plan, sensitivity 2
         else m.student_prior_average_problem_correctness)
    q1, q2 = np.nanquantile(x, [1 / 3, 2 / 3])
    m["group"] = np.where(x.isna(), "missing", np.where(x <= q1, "low", np.where(x > q2, "high", "middle")))
    m["A"] = m.in_treatment.astype(int)
    return m.reset_index(drop=True)


def simulate(m, kind, rng):
    """Outcome with a known structure.  Baseline from prior completion; effect by group if kind == 'effect'."""
    z = (m.student_prior_assignments_percent_completed.fillna(.6) - .6) * 3
    base = 1 / (1 + np.exp(-(0.6 + z)))
    eff = np.zeros(len(m))
    if kind == "effect":
        eff = np.where(m.group == "low", 0.08, np.where(m.group == "high", -0.03, 0.0))
    p1 = np.clip(base + eff, 0.01, 0.99)
    p0 = base.to_numpy()
    y = np.where(m.A == 1, rng.random(len(m)) < p1, rng.random(len(m)) < p0).astype(int)
    return y, p0, p1


def features(d):
    X = d[COV].copy()
    for g in ("low", "middle", "high"):
        X[f"g_{g}"] = (d.group == g).astype(float)
    return X.to_numpy(float)


def fit_outcome(X, a, y, arm):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, random_state=0)
    clf.fit(X[a == arm], y[a == arm])
    return clf


def policies_on(test, mu0, mu1, train_mu0, train_mu1, train):
    """Assignments (0/1) of every policy for the test students; choices learnt on the training folds."""
    best = int(train_mu1.mean() > train_mu0.mean())
    out = {"P0": np.zeros(len(test), int), "P1": np.ones(len(test), int), "P2": np.full(len(test), best)}
    p3 = np.full(len(test), best)
    for g in ("low", "middle", "high"):
        tr = train.group.to_numpy() == g
        if tr.sum():
            p3[test.group.to_numpy() == g] = int(train_mu1[tr].mean() > train_mu0[tr].mean())
    out["P3"] = p3
    out["P4"] = (mu1 > mu0).astype(int)
    # P5: P4, but the low group gets its best arm; floors checked on the training folds against P2
    p5 = out["P4"].copy()
    low_te = test.group.to_numpy() == "low"
    p5[low_te] = p3[low_te]
    tr_p4 = (train_mu1 > train_mu0).astype(int)
    tr_p5 = tr_p4.copy()
    low_tr = train.group.to_numpy() == "low"
    tr_p5[low_tr] = int(train_mu1[low_tr].mean() > train_mu0[low_tr].mean()) if low_tr.sum() else best
    for g in ("low", "middle", "high"):
        tr = train.group.to_numpy() == g
        if not tr.sum():
            continue
        v5 = np.where(tr_p5[tr] == 1, train_mu1[tr], train_mu0[tr]).mean()
        v2 = (train_mu1 if best else train_mu0)[tr].mean()
        if v5 < v2:
            p5[test.group.to_numpy() == g] = best
    out["P5"] = p5
    return out


def dr_scores(pol, a, y, mu0, mu1, pA=P, ipw=False):
    prob = np.where(pol == 1, pA, 1 - pA)
    if ipw:                                                                      # plan, sensitivity 4
        return (a == pol) / prob * y
    mu_pol = np.where(pol == 1, mu1, mu0)
    mu_a = np.where(a == 1, mu1, mu0)
    return mu_pol + (a == pol) / prob * (y - mu_a)


def experiment(d, y, seed, ipw=False):
    X, a = features(d), d.A.to_numpy()
    scores = {k: np.full(len(d), np.nan) for k in POLICIES}
    gkf = GroupKFold(n_splits=min(5, d.class_id.nunique()))   # plan: 5 folds, or one per class if fewer than 5
    for tr, te in gkf.split(X, y, groups=d.class_id):
        m0, m1 = fit_outcome(X[tr], a[tr], y[tr], 0), fit_outcome(X[tr], a[tr], y[tr], 1)
        mu0, mu1 = m0.predict_proba(X[te])[:, 1], m1.predict_proba(X[te])[:, 1]
        tmu0, tmu1 = m0.predict_proba(X[tr])[:, 1], m1.predict_proba(X[tr])[:, 1]
        pols = policies_on(d.iloc[te], mu0, mu1, tmu0, tmu1, d.iloc[tr])
        for k, pol in pols.items():
            scores[k][te] = dr_scores(pol, a[te], y[te], mu0, mu1, d.pA.to_numpy()[te], ipw)
    return scores


def values(scores, groups, w=None):
    """Value per group and gap for every policy (w: bootstrap weights per student)."""
    w = np.ones(len(groups)) if w is None else w
    out = {}
    for k, s in scores.items():
        v = {}
        for g in ("low", "high"):
            m = groups == g
            v[g] = float((w[m] * s[m]).sum() / w[m].sum())
        v["all"] = float((w * s).sum() / w.sum())
        v["gap"] = v["high"] - v["low"]
        out[k] = v
    return out


def boot_weights(classes, rng, B):
    u, inv = np.unique(classes, return_inverse=True)
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B)
    return W[:, inv].astype(float)


def dersimonian_laird(est, var):
    est, var = np.asarray(est), np.asarray(var)
    w = 1 / var
    fixed = (w * est).sum() / w.sum()
    q = (w * (est - fixed) ** 2).sum()
    tau2 = max(0.0, (q - (len(est) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (var + tau2)
    mean = (wr * est).sum() / wr.sum()
    se = np.sqrt(1 / wr.sum())
    return dict(mean=float(mean), ci=[float(mean - 1.96 * se), float(mean + 1.96 * se)], tau2=float(tau2), k=len(est))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", choices=["effect", "null"])
    g.add_argument("--real", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sens", choices=["propensity", "groups_completion", "ipw", "min10classes"], default=None)
    args = ap.parse_args()
    if args.real and not os.path.exists(os.path.join(HERE, "..", "notes", "APPROVED")):
        raise SystemExit("Real outcomes are locked: the analysis plan has not been approved (notes/APPROVED is missing).")
    rng = np.random.default_rng(20261012)
    m = load(args.real, args.sens)
    if args.real:
        y_all, truth = m.assignment_completed.to_numpy(int), None
    else:
        y_all, p0, p1 = simulate(m, args.sim, rng)
        m["p0"], m["p1"] = p0, p1
    per_exp, S, grp_all, cls_all, exp_all = {}, {k: [] for k in POLICIES}, [], [], []
    for e, d in m.groupby("target_sequence"):
        d = d.reset_index(drop=True)
        y = y_all[m.target_sequence.to_numpy() == e]
        sc = experiment(d, y, 0, ipw=args.sens == "ipw")
        for k in POLICIES:
            S[k].append(sc[k])
        grp_all.append(d.group.to_numpy()); cls_all.append((e + "_" + d.class_id.astype(str)).to_numpy())
        exp_all.append(np.full(len(d), e))
        rec = {"n": len(d), "values": values(sc, d.group.to_numpy()),
               "policies_differ_from_P2": {k: bool(np.any(np.abs(sc[k] - sc["P2"]) > 0)) for k in POLICIES if k != "P2"}}
        if not args.real:
            rec["true_effect_low"] = float((d.p1 - d.p0)[d.group == "low"].mean())
            rec["true_effect_high"] = float((d.p1 - d.p0)[d.group == "high"].mean())
        per_exp[e] = rec
    S = {k: np.concatenate(v) for k, v in S.items()}
    grp, cls, ex = np.concatenate(grp_all), np.concatenate(cls_all), np.concatenate(exp_all)
    point = values(S, grp)
    draws = {k: {q: [] for q in ("low", "high", "all", "gap")} for k in POLICIES if k != "P2"}
    idx_by_exp = [np.flatnonzero(ex == e) for e in np.unique(ex)]
    for start in range(0, args.boot, 100):
        nb = min(100, args.boot - start)
        W = np.zeros((nb, len(grp)))
        for idx in idx_by_exp:
            W[:, idx] = boot_weights(cls[idx], rng, nb)
        for w in W:
            v = values(S, grp, w)
            for k in draws:
                for q in draws[k]:
                    draws[k][q].append(v[k][q] - v["P2"][q])
    pooled = {k: {q: dict(mean=point[k][q] - point["P2"][q],
                          ci=[float(np.percentile(draws[k][q], 2.5)), float(np.percentile(draws[k][q], 97.5))],
                          upper95_one_sided=float(np.percentile(draws[k][q], 95)),
                          lower05_one_sided=float(np.percentile(draws[k][q], 5)))
                  for q in draws[k]} for k in draws}
    res = dict(mode=("real" if args.real else f"sim_{args.sim}") + (f"_{args.sens}" if args.sens else ""), experiments=len(per_exp), students=int(len(m)),
               pooling="student-weighted over experiments; joint cluster bootstrap of classes within experiments",
               values=point, pooled_vs_P2=pooled, per_experiment=per_exp)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, f"policy_{res['mode']}.json"), "w"), indent=1)
    print(f"[{res['mode']}] {res['experiments']} experiments, {res['students']} students; pooled difference with P2 (reference):")
    for k, v in pooled.items():
        print(f"  {k}: " + " | ".join(f"{q} {v[q]['mean']:+.4f} [{v[q]['ci'][0]:+.4f},{v[q]['ci'][1]:+.4f}]" for q in ("low", "high", "all", "gap")))


if __name__ == "__main__":
    main()
