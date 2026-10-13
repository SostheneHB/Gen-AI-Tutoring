"""
Eedi: pupils with and without a recorded PremiumPupil value (referee report 5, item 8). Compares the two sets on
observables available for everyone: number of answers, accuracy, gender and year of birth (task 1/2 data).

    DATA_ROOT=/path/to/root python eedi_inclusion.py     -> out_reviewer/eedi_inclusion.json (aggregates only)
"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.environ.get("DATA_ROOT", ".")
DATA = os.path.join(ROOT, "Datasets", "Eedi_NeurIPS2020", "data")


def main():
    meta = pd.read_csv(glob.glob(os.path.join(DATA, "**", "student_metadata_task_1_2.csv"), recursive=True)[0])
    ans = pd.read_csv(glob.glob(os.path.join(DATA, "**", "train_task_1_2.csv"), recursive=True)[0], usecols=["UserId", "IsCorrect"])
    per = ans.groupby("UserId").IsCorrect.agg(n="size", acc="mean").reset_index()
    m = meta.merge(per, on="UserId", how="inner")
    m["recorded"] = m.PremiumPupil.notna()
    m["year"] = pd.to_datetime(m.DateOfBirth, errors="coerce").dt.year
    out = {"pupils": int(len(m)), "recorded": int(m.recorded.sum()), "share_recorded": float(m.recorded.mean())}
    for lab, d in (("recorded", m[m.recorded]), ("missing", m[~m.recorded])):
        out[lab] = dict(n=int(len(d)), answers_median=float(d.n.median()), answers_mean=float(d.n.mean()), accuracy_mean=float(d.acc.mean()),
                        accuracy_sd=float(d.acc.std()), share_gender_known=float((d.Gender.fillna(0) != 0).mean()),
                        share_female=float((d.Gender == 1).sum() / max((d.Gender.isin([1, 2])).sum(), 1)),
                        year_of_birth_median=float(d.year.median()) if d.year.notna().any() else None,
                        share_year_known=float(d.year.notna().mean()))
    a, b = m[m.recorded], m[~m.recorded]
    sd = lambda x, y: float((x.mean() - y.mean()) / np.sqrt((x.var() + y.var()) / 2))
    out["std_mean_difference"] = dict(answers=sd(np.log(a.n), np.log(b.n)), accuracy=sd(a.acc, b.acc))
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open("out_reviewer/eedi_inclusion.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
