"""Sample flow of the ASSISTments study, from the feature file to the analysed set.

    python sample_flow.py --features assistments_features_time.csv --out sample_flow.csv

Every rule below uses row counts only, never outcomes of the evaluated steps:
  1. students with at least 20 rows (load_features);
  2. students with at least 26 rows, i.e. a full window x_0..x_25 (keep_full_trajectories);
  3. schools with at least one row after step 25 (the label window).  A school
     without such a row cannot receive a group label, so its students are removed
     (rebuild_school_strata drops rows whose label is missing).
The script also lists the removed schools and the students in them.
"""
import argparse

import pandas as pd

import assistments_pipeline as ap


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--features", default="assistments_features_time.csv")
    a.add_argument("--out", default="sample_flow.csv")
    args = a.parse_args()
    raw = pd.read_csv(args.features, usecols=["user_id", "school_id"])
    df1 = ap.load_features(args.features)
    df2 = ap.keep_full_trajectories(df1)
    labelled = df2.loc[df2["step"] > ap.N_TRANS, "school_id"].unique()
    df3 = df2[df2["school_id"].isin(labelled)]
    rows = []
    for name, d, rule in [
            ("feature file", raw, "students with a skill and a school"),
            (">= 20 rows", df1, "at least 20 attempts"),
            (">= 26 rows", df2, "full window x_0..x_25 (T = 25 transitions)"),
            ("labelled schools", df3, "school has at least one step after t = 25 (label window)")]:
        rows.append(dict(stage=name, rule=rule, students=d["user_id"].nunique(),
                         schools=d["school_id"].nunique(), rows=len(d)))
    flow = pd.DataFrame(rows)
    flow.to_csv(args.out, index=False)
    print(flow.to_string(index=False))
    gone = df2[~df2["school_id"].isin(labelled)].groupby("school_id")["user_id"].nunique()
    print("\nremoved schools (students per school):", gone.to_dict())
    n = df2.groupby("user_id")["step"].max() + 1
    print("students with exactly 26 steps:", int((n == ap.N_TRANS + 1).sum()))


if __name__ == "__main__":
    main()
