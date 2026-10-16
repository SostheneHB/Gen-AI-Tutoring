"""Help use per problem log, ASSISTments 2012-2013 (local cache for experiment 16).

    python code/scripts/extract_assistments_hints.py --raw <2012-2013-data-with-predictions-4-final.csv>

Reads only the columns problem_log_id, hint_count, bottom_hint and original of the raw file, keeps the logs of
the the companion paper [A] features file, and writes data/ASSISTments_2012_2013/hints_by_log.parquet.  The file stays in data/
(ignored by git): ASSISTments data and files derived from them are not redistributed.
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "ASSISTments_2012_2013" / "assistments_features_time.csv"
OUT = ROOT / "data" / "ASSISTments_2012_2013" / "hints_by_log.parquet"


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--raw", required=True)
    a = a_.parse_args()
    keep = set(pd.read_csv(FEAT, usecols=["problem_log_id"]).problem_log_id.astype("int64"))
    parts = []
    cols = ["problem_log_id", "hint_count", "bottom_hint", "original"]
    for ch in pd.read_csv(a.raw, usecols=cols, chunksize=1_000_000, dtype=str):
        ch = ch.dropna(subset=["problem_log_id"])
        ch["problem_log_id"] = pd.to_numeric(ch.problem_log_id, errors="coerce")
        ch = ch[ch.problem_log_id.isin(keep)]
        for c in cols[1:]:
            ch[c] = pd.to_numeric(ch[c], errors="coerce")
        parts.append(ch)
    h = pd.concat(parts).drop_duplicates("problem_log_id")
    h.to_parquet(OUT, index=False)
    print(f"{len(h)} of {len(keep)} logs matched; hint used in {(h.hint_count > 0).mean():.3f}, bottom hint in "
          f"{(h.bottom_hint > 0).mean():.3f}; original problems {(h.original == 1).mean():.3f}")


if __name__ == "__main__":
    main()
