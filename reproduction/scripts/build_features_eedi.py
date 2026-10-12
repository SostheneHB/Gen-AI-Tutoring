"""Features for the Eedi NeurIPS 2020 data (tasks 1/2), in the format of
assistments_features_time.csv, so that assistments_pipeline.py can be reused.

    python build_features_eedi.py --data <Eedi folder>/data --out eedi_features_time.csv

State x_t = (k_t, e_t, p_t, d_t), rows of each student in DateAnswered order
(ties broken by AnswerId):
  k_t  rolling mean of IsCorrect over the last 10 answers (rolling-correctness proxy)
  e_t  self-reported confidence rescaled to [-0.5, 0.5]; when an answer has no
       confidence, the student's last reported confidence is carried forward,
       and 0 before the first report (Eedi has no affect detectors)
  p_t  1{k_t > 0.5}
  d_t  IsCorrect minus the mean correctness of the question's most specific
       subject (the analysis pipeline rebuilds it from training students only)
Group S = PremiumPupil (eligible for free school meals or pupil premium);
students with a missing value are removed.  Cluster school_id = the class
(GroupId) in which the student answered most often.

The data are licensed CC BY-NC-ND 4.0: the output file is a derived data set
and must not be redistributed.
"""
import argparse
import ast
import glob
import os

import numpy as np
import pandas as pd


def find(data, pattern):
    hits = sorted(glob.glob(os.path.join(data, "**", pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eedi_features_time.csv")
    a = ap.parse_args()
    ans = pd.read_csv(find(a.data, "train_task_1_2.csv"))
    meta = pd.read_csv(find(a.data, "answer_metadata_task_1_2.csv"))
    stu = pd.read_csv(find(a.data, "student_metadata_task_1_2.csv"))
    qm = pd.read_csv(find(a.data, "question_metadata_task_1_2.csv"))
    subj = pd.read_csv(find(a.data, "subject_metadata.csv"))
    print("answers", len(ans), "students", ans.UserId.nunique(), flush=True)

    level = subj.set_index("SubjectId")["Level"].to_dict()
    qm["skill_id"] = qm["SubjectId"].map(lambda s: max(ast.literal_eval(s), key=lambda x: level.get(x, -1)))

    d = ans.merge(meta, on="AnswerId", how="left").merge(qm[["QuestionId", "skill_id"]], on="QuestionId", how="left")
    d["start_time"] = pd.to_datetime(d["DateAnswered"], errors="coerce")
    n0 = len(d)
    d = d.dropna(subset=["start_time"])
    d = d.merge(stu[["UserId", "PremiumPupil"]], on="UserId", how="left")
    d = d.dropna(subset=["PremiumPupil"])
    print(f"rows kept after time and PremiumPupil filters: {len(d)} of {n0}", flush=True)
    d = d.sort_values(["UserId", "start_time", "AnswerId"]).reset_index(drop=True)

    d["correct"] = d["IsCorrect"].astype(float)
    d["k_t"] = d.groupby("UserId")["correct"].transform(lambda v: v.rolling(10, min_periods=1).mean())
    conf = d["Confidence"] / 100.0 - 0.5
    d["e_t"] = conf.groupby(d["UserId"]).ffill().fillna(0.0)
    d["p_t"] = (d["k_t"] > 0.5).astype(int)
    d["d_t"] = d["correct"] - d["skill_id"].map(d.groupby("skill_id")["correct"].mean())
    d["S"] = d["PremiumPupil"].astype(int)
    modal = d.dropna(subset=["GroupId"]).groupby("UserId")["GroupId"].agg(lambda g: g.value_counts().index[0])
    d["school_id"] = d["UserId"].map(modal)
    d = d.dropna(subset=["school_id"])
    out = d.rename(columns={"UserId": "user_id", "QuestionId": "problem_id", "AnswerId": "problem_log_id"})
    out = out[["user_id", "problem_id", "k_t", "e_t", "p_t", "d_t", "S", "correct", "skill_id",
               "school_id", "problem_log_id", "start_time"]]
    out.to_csv(a.out, index=False)
    n = out.groupby("user_id").size()
    print(f"students {out.user_id.nunique()} (>=26 answers: {(n >= 26).sum()}), classes {out.school_id.nunique()}, "
          f"premium share {out.drop_duplicates('user_id').S.mean():.3f}, confidence reported {d['Confidence'].notna().mean():.3f}")


if __name__ == "__main__":
    main()
