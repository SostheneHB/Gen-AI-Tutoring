"""Level-2 audit on Eedi with a real sensitive attribute (PremiumPupil).

    python run_eedi.py --features eedi_features_time.csv --out out_eedi

Reuses assistments_pipeline.run unchanged.  The group is the student's own
PremiumPupil flag (strata="given": the label is not rebuilt from outcomes), and
the sampling unit is the class (GroupId), so splits are class-disjoint and the
bootstrap resamples classes.
"""
import argparse
import json
import os

import pandas as pd

import assistments_pipeline as ap


def strict_split(classes_of):
    """Class-disjoint split that also removes validation and test students who
    answered in any class of a training student (students belong to several
    classes; the cluster used for the split is each student's main class)."""
    base = ap.split_schools

    def split(df, rng):
        tr, va, te = base(df, rng)
        tr_classes = set().union(*(classes_of.get(u, set()) for u in tr))
        clean = lambda ids: [u for u in ids if not (classes_of.get(u, set()) & tr_classes)]
        va2, te2 = clean(va), clean(te)
        split.removed.append((len(va) - len(va2), len(te) - len(te2), len(te)))
        return tr, va2, te2
    split.removed = []
    return split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="eedi_features_time.csv")
    p.add_argument("--out", default="out_eedi")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--strict", default=None,
                   help="Eedi data folder: run the strict variant (no test student in any training class)")
    a = p.parse_args()
    df = ap.keep_full_trajectories(ap.load_features(a.features))
    summary = {"n_students_full_traj": int(df.user_id.nunique()), "n_classes": int(df.school_id.nunique()),
               "premium_share": float(df.drop_duplicates("user_id").G.mean())}
    print(summary, flush=True)
    name = "class_disjoint"
    if a.strict:
        ans = pd.read_csv(os.path.join(a.strict, "train_data/train_task_1_2.csv"), usecols=["UserId", "AnswerId"])
        meta = pd.read_csv(os.path.join(a.strict, "metadata/answer_metadata_task_1_2.csv"), usecols=["AnswerId", "GroupId"])
        m = ans.merge(meta, on="AnswerId").dropna()
        classes_of = m.groupby("UserId").GroupId.apply(set).to_dict()
        ap.split_schools = strict_split(classes_of)
        name = "class_disjoint_strict"
    runs = [ap.run(df, f"{name}_{j}", a.seed + 200 + j, "school", "given",
                   ridge_sweep=(j == 0 and not a.strict), outdir=a.out) for j in range(a.splits)]
    summary[name] = ap.summarise(runs)
    if a.strict:
        summary["removed_val_test_total"] = ap.split_schools.removed
    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps({k: (v["mean"], v["t_ci"]) for k, v in summary[name].items()
                      if isinstance(v, dict) and "mean" in v and not k.startswith("mse")}, indent=1))


if __name__ == "__main__":
    main()
