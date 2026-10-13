"""
OULAD: timing and missing data of the weekly features (referee report 5, item 8). Shares of student-weeks before
the first score, students without a score by week 25, students who withdraw before week 26, inactive weeks and
assessments without a score.

    DATA_ROOT=/path/to/root python oulad_timing.py      -> out_reviewer/oulad_timing.json (aggregates only)
"""
import json
import os

import pandas as pd

ROOT = os.environ.get("DATA_ROOT", ".")
D = os.path.join(ROOT, "Datasets", "OULAD")


def main():
    f = pd.read_csv(os.path.join(D, "oulad_features_weekly.csv"), usecols=["user_id", "problem_id", "k_t", "correct"]).sort_values(["user_id", "problem_id"])
    first = f.groupby("user_id").k_t.apply(lambda v: int((v.values != 0.5).argmax()) if (v.values != 0.5).any() else 26)
    week = f.groupby("user_id").cumcount()
    out = dict(students=int(f.user_id.nunique()),
               share_weeks_before_first_score=float((week < f.user_id.map(first)).mean()),
               share_students_no_score_by_week25=float((first >= 26).mean()),
               share_inactive_weeks=float(1 - f.correct.mean()))
    reg, info = pd.read_csv(os.path.join(D, "studentRegistration.csv")), pd.read_csv(os.path.join(D, "studentInfo.csv"))
    fst = info.sort_values(["id_student", "code_presentation", "code_module"]).drop_duplicates("id_student")
    fst = fst[fst.id_student.isin(f.user_id.unique())]
    r = fst.merge(reg, on=["id_student", "code_module", "code_presentation"], how="left")
    out["share_withdrawn_before_week26"] = float((pd.to_numeric(r.date_unregistration, errors="coerce") < 182).mean())
    sa = pd.read_csv(os.path.join(D, "studentAssessment.csv"))
    out["share_assessment_rows_missing_score"] = float(pd.to_numeric(sa.score, errors="coerce").isna().mean())
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open("out_reviewer/oulad_timing.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
