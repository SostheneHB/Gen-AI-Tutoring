"""Level-2 audit on OULAD with declared sensitive attributes.

    python run_oulad.py --features oulad_features_weekly.csv --out out_oulad

For each attribute (deprivation, gender, disability) the group column S is
set from the feature file, and five random student splits are run with
assistments_pipeline.run unchanged (strata="given").  There are only 22
module presentations, too few for presentation-disjoint splits, so students
are split at random and the bootstrap resamples module presentations
(school_id), the unit that shares the content, calendar and assessments.
"""
import argparse
import json
import os

import assistments_pipeline as ap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="oulad_features_weekly.csv")
    p.add_argument("--out", default="out_oulad")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--attributes", default="imd,gender,disability")
    a = p.parse_args()
    base = ap.keep_full_trajectories(ap.load_features(a.features))
    summary = {}
    for att in a.attributes.split(","):
        df = base.copy()
        df["S"] = df[f"S_{att}"]
        df["G"] = df["S"].astype(int)
        name = f"{att}_student_split"
        runs = [ap.run(df, f"{name}_{j}", a.seed + 300 + j, "student", "given",
                       ridge_sweep=False, outdir=os.path.join(a.out, att)) for j in range(5)]
        summary[att] = ap.summarise(runs)
        summary[att]["group_share"] = float(df.drop_duplicates("user_id").G.mean())
        print(att, {k: (round(v["mean"], 4), [round(x, 4) for x in v["t_ci"]])
                    for k, v in summary[att].items() if isinstance(v, dict) and k.startswith(("observed", "replay_gap"))},
              flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
