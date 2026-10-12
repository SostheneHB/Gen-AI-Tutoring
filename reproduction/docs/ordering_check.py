"""How many transition pairs change between problem-id order and timestamp order.

    python ordering_check.py assistments_features_pid.csv assistments_features_time.csv
"""
import sys

import numpy as np
import pandas as pd

N_STEPS = 26


def pairs(df, keep):
    df = df[df.user_id.isin(keep)].copy()
    df["step"] = df.groupby("user_id").cumcount()
    df = df[df.step < N_STEPS]
    nxt = df.groupby("user_id")["problem_log_id"].shift(-1)
    m = nxt.notna()
    return set(zip(df.problem_log_id[m], nxt[m].astype(np.int64))), set(df.problem_log_id)


pid, tim = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
c = tim.user_id.value_counts()
c = c[c >= 20]
keep = c[c >= N_STEPS].index
pp, rp = pairs(pid, keep)
pt, rt = pairs(tim, keep)
print(f"students analysed: {len(keep):,}; transition pairs: {len(pt):,}")
print(f"pairs consecutive in time among problem-id pairs: {len(pt & pp) / len(pp):.4f}")
print(f"rows of the first {N_STEPS} steps shared by both orders: {len(rt & rp) / len(rt):.4f}")
print(f"repeated (student, problem) rows: {int(tim.duplicated(['user_id', 'problem_id']).sum()):,}")
