"""
Build the ASSISTments 2012-2013 state features from the raw log.

    python build_features.py --raw 2012-2013-data-with-predictions-4-final.csv \
        --order time --out assistments_features_time.csv

`--order problem_id` reproduces the historical upstream file
`assistments_features_with_pid.csv` exactly (all columns, all 2,711,753 rows):
rows are sorted stably by (user_id, problem_id) in raw-file order.
`--order time` sorts every student's rows by (start_time, problem_log_id), which
is the chronological order required for transition pairs (x_t, x_{t+1}).

Definitions (identical in both orders except for the row order):
  k_t  rolling mean of `correct` over the last 10 rows of the student
       (a rolling-correctness proxy, not a latent knowledge estimate)
  e_t  Average_confidence(CONCENTRATING) - Average_confidence(BORED)
  p_t  1{k_t > 0.5}
  d_t  correct - mean correctness of the skill over the corpus
       (the analysis pipeline rebuilds d_t from training students only)
  S    1{school mean correctness > median of school means} (full corpus;
       the analysis pipeline rebuilds it from training students only)
The response, skill, school identifier and timestamp are written as extra
columns, so the pipeline needs no separate school / time files and does not
have to recover `correct` from the sign of d_t.
"""
from __future__ import annotations

import argparse
import hashlib

import pandas as pd

COLS = ["problem_log_id", "user_id", "problem_id", "skill_id", "correct",
        "start_time", "school_id",
        "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)"]


def load_raw(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=COLS)
    return pd.read_csv(path, usecols=COLS, low_memory=False)


def build(raw: pd.DataFrame, order: str) -> pd.DataFrame:
    d = raw.dropna(subset=["skill_id", "correct", "school_id"]).copy()
    d["e_t"] = d["Average_confidence(CONCENTRATING)"] - d["Average_confidence(BORED)"]
    if order == "problem_id":
        keys = ["user_id", "problem_id"]
    elif order == "time":
        d["start_time"] = pd.to_datetime(d["start_time"], format="ISO8601")
        keys = ["user_id", "start_time", "problem_log_id"]
    else:
        raise ValueError(order)
    d = d.sort_values(keys, kind="stable").reset_index(drop=True)
    d["k_t"] = d.groupby("user_id")["correct"].transform(
        lambda v: v.rolling(10, min_periods=1).mean())
    d["p_t"] = (d["k_t"] > 0.5).astype(int)
    d["d_t"] = d["correct"] - d["skill_id"].map(d.groupby("skill_id")["correct"].mean())
    sch = d.groupby("school_id")["correct"].mean()
    d["S"] = (d["school_id"].map(sch) > sch.median()).astype(int)
    return d[["user_id", "problem_id", "k_t", "e_t", "p_t", "d_t", "S",
              "correct", "skill_id", "school_id", "problem_log_id", "start_time"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="2012-2013-data-with-predictions-4-final.csv")
    ap.add_argument("--order", choices=["problem_id", "time"], default="time")
    ap.add_argument("--out", default="assistments_features_time.csv")
    args = ap.parse_args()
    out = build(load_raw(args.raw), args.order)
    out.to_csv(args.out, index=False)
    h = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(f"{args.out}: {len(out):,} rows, {out.user_id.nunique():,} students, sha256 {h}")


if __name__ == "__main__":
    main()
