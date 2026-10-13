"""
Post-hoc sensitivity (listed in the plan's Deviations, 13 October 2026): dependence between experiments.

The planned bootstrap resamples classes within each experiment, independently across experiments. A student, or
a class, can appear in several experiments. This script reuses the planned doubly robust scores (same code, same
cross-fitting) and recomputes the pooled differences with P2 under two other resampling schemes:
  * classes resampled jointly across experiments (a class keeps all its records in every experiment);
  * students resampled jointly across experiments (a student keeps all his or her records).
It also counts the students and classes that appear in more than one experiment.

    python dependence_check.py      -> outputs/dependence_check.json
"""
import json
import os

import numpy as np

import analysis as A

B = 2000


def weights(ids, rng, nb):
    u, inv = np.unique(ids, return_inverse=True)
    W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=nb)
    return W[:, inv].astype(float)


def main():
    m = A.load(True)
    y_all = m.assignment_completed.to_numpy(int)
    S, grp, cls, stu = {k: [] for k in A.POLICIES}, [], [], []
    for e, d in m.groupby("target_sequence"):
        d = d.reset_index(drop=True)
        sc = A.experiment(d, y_all[m.target_sequence.to_numpy() == e], 0)
        for k in A.POLICIES:
            S[k].append(sc[k])
        grp.append(d.group.to_numpy()); cls.append(d.class_id.astype(str).to_numpy()); stu.append(d.student_id.astype(str).to_numpy())
    S = {k: np.concatenate(v) for k, v in S.items()}
    grp, cls, stu = np.concatenate(grp), np.concatenate(cls), np.concatenate(stu)
    per_stu = m.groupby("student_id").target_sequence.nunique()
    per_cls = m.groupby("class_id").target_sequence.nunique()
    overlap = dict(records=int(len(m)), students=int(len(per_stu)), students_in_several=int((per_stu > 1).sum()),
                   share_records_of_repeat_students=float(m.student_id.isin(per_stu.index[per_stu > 1]).mean()),
                   classes=int(len(per_cls)), classes_in_several=int((per_cls > 1).sum()),
                   share_records_of_repeat_classes=float(m.class_id.isin(per_cls.index[per_cls > 1]).mean()))
    point = A.values(S, grp)
    out = {"overlap": overlap}
    rng = np.random.default_rng(20261013)
    for name, ids in (("classes_joint", cls), ("students_joint", stu)):
        draws = {k: {q: [] for q in ("low", "high", "gap")} for k in ("P1", "P3", "P4", "P5")}
        for start in range(0, B, 100):
            for w in weights(ids, rng, 100):
                v = A.values(S, grp, w)
                for k in draws:
                    for q in draws[k]:
                        draws[k][q].append(v[k][q] - v["P2"][q])
        out[name] = {k: {q: dict(mean=float(point[k][q] - point["P2"][q]),
                                 ci=[float(np.percentile(draws[k][q], 2.5)), float(np.percentile(draws[k][q], 97.5))],
                                 lower05_one_sided=float(np.percentile(draws[k][q], 5)))
                         for q in draws[k]} for k in draws}
    json.dump(out, open(os.path.join(A.OUT, "dependence_check.json"), "w"), indent=1)
    print(json.dumps(overlap, indent=1))
    for name in ("classes_joint", "students_joint"):
        for k in ("P1", "P3", "P5"):
            v = out[name][k]
            print(f"{name:15s} {k}: " + " | ".join(f"{q} {100 * v[q]['mean']:+.1f} [{100 * v[q]['ci'][0]:+.1f},{100 * v[q]['ci'][1]:+.1f}] (lower5 {100 * v[q]['lower05_one_sided']:+.1f})" for q in ("low", "high", "gap")))


if __name__ == "__main__":
    main()
