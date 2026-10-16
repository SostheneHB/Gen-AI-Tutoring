"""Experiment 15: the certificate on real decisions — ASSISTments randomized experiments (exploratory).

    python code/scripts/run_rct_certificate.py [--splits 100] [--cohorts 40]

Data: the 30 randomized ASSISTments experiments of the companion study (OSF k8ph9; selection rules and groups
of its analysis plan, loaded with its own code).  This analysis is NOT part of that registered plan: it is an
exploratory use of the same data for this project.  Only aggregate numbers are written; the data stay local.

One decision per student and experiment.  Groups: low and high terciles of prior correctness.  Target: the gap
in assignment completion (high minus low), pooled over experiments.
  reference  in each experiment, the arm with the higher completion on the training classes, for everyone;
  plans      the low group is switched to its other arm in the experiments where the training classes estimate
             the largest gain for it, in this order (the knapsack filling, without budget), until the promised
             narrowing reaches rho / (1 - delta); the high group keeps the reference, so its mean is unchanged.
  promise    the training-class estimate of the gain of the low group (difference in means, randomized arms).
Cohorts: the evaluation classes are split into K disjoint cohorts.  In each cohort the realised change of the
low group's mean completion under a plan is estimated without bias by inverse-probability weighting (assignment
probability 1/2): mean over its low students of (1{A=o}/0.5 - 1{A=r}/0.5) y over the switched experiments.  The
score of a cohort is the worst shortfall over the family, +inf if the estimated change is negative (a loss).
Each split: half of the classes train, the other half form K cohorts; K/2 calibrate (conformal margin, and Learn
then Test) and K/2 test.  Repeated over random class splits.
Writes outputs/15_rct_certificate/results_K<cohorts>.json (aggregate numbers only).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[2]
RCT = ROOT / "code" / "rct"
sys.path.insert(0, str(RCT / "code"))
import analysis as rct  # noqa: E402

ALPHA, ETA = 0.1, 0.1
DELTAS = np.round(np.arange(0, 0.51, 0.05), 3)


def conformal_margin(scores, alpha):
    m = len(scores)
    k = math.ceil((m + 1) * (1 - alpha))
    return float("inf") if k > m else float(np.sort(scores)[k - 1])


def one_split(d, rng, K):
    classes = d.class_id.unique()
    rng.shuffle(classes)
    tr_cl = set(classes[: len(classes) // 2])
    tr = d[d.class_id.isin(tr_cl)]
    ev = d[~d.class_id.isin(tr_cl)].copy()
    exps = sorted(d.target_sequence.unique())
    # reference arm per experiment (best overall on training classes) and the low group's estimated gain
    ref, other, gain, w = {}, {}, {}, {}
    n_low_ev = (ev.group == "low").sum()
    for e in exps:
        t = tr[tr.target_sequence == e]
        m1, m0 = t[t.A == 1].y.mean(), t[t.A == 0].y.mean()
        r = 1 if m1 >= m0 else 0
        tl = t[t.group == "low"]
        if (tl.A == 1).sum() < 2 or (tl.A == 0).sum() < 2:
            continue
        ref[e], other[e] = r, 1 - r
        gain[e] = tl[tl.A == other[e]].y.mean() - tl[tl.A == r].y.mean()
        w[e] = (ev[(ev.target_sequence == e)].group == "low").sum() / max(n_low_ev, 1)
    order = [e for e in sorted(gain, key=lambda e: -gain[e]) if gain[e] > 0]
    max_promise = sum(w[e] * gain[e] for e in order)
    if max_promise <= 0:
        return None
    rho = 0.5 * max_promise
    family = []
    for dl in DELTAS:
        need, S, prom = rho / (1 - dl), [], 0.0
        for e in order:
            if prom >= need:
                break
            S.append(e); prom += w[e] * gain[e]
        if prom >= need - 1e-12:
            family.append((float(dl), S, prom))
    if not family:
        return None
    # cohorts of evaluation classes
    ev_cl = ev.class_id.unique()
    rng.shuffle(ev_cl)
    coh = {c: i % K for i, c in enumerate(ev_cl)}
    ev["cohort"] = ev.class_id.map(coh)
    low = ev[ev.group == "low"].copy()
    low["r"] = low.target_sequence.map(ref)
    low["o"] = low.target_sequence.map(other)
    low["ipw"] = ((low.A == low.o).astype(float) - (low.A == low.r).astype(float)) / 0.5 * low.y
    n_low_coh = low.groupby("cohort").size()
    real = np.zeros((K, len(family)))
    for j, (_, S, _) in enumerate(family):
        s = low[low.target_sequence.isin(S)].groupby("cohort").ipw.sum()
        real[:, j] = (s.reindex(range(K), fill_value=0.0) / n_low_coh.reindex(range(K))).to_numpy()
    prom = np.array([f[2] for f in family])
    score = np.array([np.inf if (real[c] < 0).any() else float(np.round(np.max(1 - real[c] / prom), 12)) for c in range(K)])
    ok = (real >= rho)                                  # no loss and narrowing >= rho, per cohort and plan
    cal, test = np.arange(K // 2), np.arange(K // 2, K)
    q = conformal_margin(score[cal], ALPHA)
    jc = next((j for j, f in enumerate(family) if f[0] >= q), None)
    # Learn then Test, fixed sequence from the largest margin down
    jl = None
    for j in sorted(range(len(family)), key=lambda j: -family[j][0]):
        if binom.cdf(int((~ok[cal, j]).sum()), len(cal), ALPHA) <= ETA:
            jl = j
        else:
            break
    pooled = lambda j: float(low[low.target_sequence.isin(family[j][1]) & low.cohort.isin(test)].ipw.sum()
                             / max((low.cohort.isin(test)).sum(), 1))
    # replication check on all evaluation classes: promise of the full filling (training) vs its realised change (IPW)
    real_full = float(low[low.target_sequence.isin(order)].ipw.sum() / len(low))
    out = dict(rho=rho, max_promise=max_promise, real_full=real_full, n_positive=len(order), family=len(family), q=q, cover=float((score[test] <= q).mean()),
               naive_success=float(ok[test, 0].mean()), naive_pooled=pooled(0),
               conformal=None if jc is None else dict(delta=family[jc][0], success=float(ok[test, jc].mean()), pooled=pooled(jc),
                                                      switched=len(family[jc][1])),
               ltt=None if jl is None else dict(delta=family[jl][0], success=float(ok[test, jl].mean()), pooled=pooled(jl)),
               finite_scores=float(np.isfinite(score).mean()), score_sd=float(np.std(score[np.isfinite(score)])) if np.isfinite(score).any() else None,
               real_sd=float(real[:, 0].std()), gap_eval=float(ev[ev.group == "high"].y.mean() - ev[ev.group == "low"].y.mean()))
    return out


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--splits", type=int, default=100)
    a_.add_argument("--cohorts", type=int, default=40)
    a = a_.parse_args()
    d = rct.load(real=True)
    d = d[d.group.isin(["low", "high"])].copy()
    d["y"] = d.assignment_completed.astype(float)
    rng = np.random.default_rng(20261016)
    runs = [r for r in (one_split(d, rng, a.cohorts) for _ in range(a.splits)) if r is not None]
    iv = [r for r in runs if r["conformal"]]
    lt = [r for r in runs if r["ltt"]]
    f = lambda v: [float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95))]
    summ = dict(splits=len(runs), cohorts=a.cohorts, students=int(len(d)), experiments=int(d.target_sequence.nunique()),
                classes=int(d.class_id.nunique()), max_promise=f([r["max_promise"] for r in runs]),
                real_full=f([r["real_full"] for r in runs]), share_real_full_positive=float(np.mean([r["real_full"] > 0 for r in runs])),
                n_positive=f([r["n_positive"] for r in runs]), gap_eval=f([r["gap_eval"] for r in runs]), rho=f([r["rho"] for r in runs]),
                cohort_sd_of_realised_change=f([r["real_sd"] for r in runs]), finite_scores=f([r["finite_scores"] for r in runs]),
                cover=f([r["cover"] for r in runs]), naive_success=f([r["naive_success"] for r in runs]),
                naive_pooled=f([r["naive_pooled"] for r in runs]),
                conformal_intervention_rate=len(iv) / len(runs),
                conformal_success=f([r["conformal"]["success"] for r in iv]) if iv else None,
                conformal_pooled=f([r["conformal"]["pooled"] for r in iv]) if iv else None,
                ltt_intervention_rate=len(lt) / len(runs),
                ltt_success=f([r["ltt"]["success"] for r in lt]) if lt else None)
    print(json.dumps(summ, indent=1))
    d_ = ROOT / "outputs" / "15_rct_certificate"
    d_.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=summ, runs=runs), open(d_ / f"results_K{a.cohorts}.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
