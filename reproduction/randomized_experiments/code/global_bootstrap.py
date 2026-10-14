"""
Correction of the interval method (recorded in notes/analysis_plan.md, Deviations), real outcomes.

analysis.py resamples classes within each experiment.  547 of the 1,788 classes (58% of the records) appear
in several experiments, so one class can be drawn independently in each, which ignores this dependence.
Here every class gets one bootstrap weight, used in every experiment where it appears.  Point estimates are
those of analysis.py (same folds, same seed); only the intervals change.

    python global_bootstrap.py [--sens propensity|groups_completion|ipw|min10classes]
    -> outputs/policy_real[_<sens>]_globalclass.json
"""
import argparse
import json
import os

import numpy as np

from analysis import OUT, POLICIES, boot_weights, experiment, load, values

B = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", choices=["propensity", "groups_completion", "ipw", "min10classes"], default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(20261012)
    m = load(True, args.sens)
    y_all = m.assignment_completed.to_numpy(int)
    S, grp, cls = {k: [] for k in POLICIES}, [], []
    for e, d in m.groupby("target_sequence"):
        d = d.reset_index(drop=True)
        sc = experiment(d, y_all[m.target_sequence.to_numpy() == e], 0, ipw=args.sens == "ipw")
        for k in POLICIES:
            S[k].append(sc[k])
        grp.append(d.group.to_numpy()); cls.append(d.class_id.to_numpy())
    S = {k: np.concatenate(v) for k, v in S.items()}
    grp, cls = np.concatenate(grp), np.concatenate(cls)
    point = values(S, grp)
    draws = {k: {q: [] for q in ("low", "high", "all", "gap")} for k in POLICIES if k != "P2"}
    for start in range(0, B, 200):
        for w in boot_weights(cls, rng, min(200, B - start)):        # one weight per class, all experiments
            v = values(S, grp, w)
            for k in draws:
                for q in draws[k]:
                    draws[k][q].append(v[k][q] - v["P2"][q])
    pooled = {k: {q: dict(mean=point[k][q] - point["P2"][q],
                          ci=[float(np.percentile(draws[k][q], 2.5)), float(np.percentile(draws[k][q], 97.5))],
                          lower05_one_sided=float(np.percentile(draws[k][q], 5)))
                  for q in draws[k]} for k in draws}
    tag = "real" + (f"_{args.sens}" if args.sens else "") + "_globalclass"
    json.dump(dict(mode=tag, students=int(len(m)), classes=int(len(np.unique(cls))),
                   bootstrap="classes resampled once over all experiments", values=point, pooled_vs_P2=pooled),
              open(os.path.join(OUT, f"policy_{tag}.json"), "w"), indent=1)
    print(f"[{tag}] {len(m)} records, {len(np.unique(cls))} classes; difference with P2, points:")
    for k, v in pooled.items():
        print(f"  {k}: " + " | ".join(f"{q} {100*v[q]['mean']:+.1f} [{100*v[q]['ci'][0]:+.1f},{100*v[q]['ci'][1]:+.1f}]"
                                       for q in ("low", "high", "all", "gap"))
              + f" | low lo5 {100*v['low']['lower05_one_sided']:+.1f}, high lo5 {100*v['high']['lower05_one_sided']:+.1f}")


if __name__ == "__main__":
    main()
