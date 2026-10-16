"""Merge the outputs of experiment 17 (sequential run + per-model parallel runs) and print the summary by size.

    python code/scripts/merge_certificate_estimated.py

Reads outputs/17_certificate_estimated/results_<regime>_sequential.json and results_<regime>_g*_s*.json, keeps one row
per (n, gamma, seed) (the sequential one when both exist), writes results_<regime>.json and prints, for each regime and
cohort size, how many models each procedure intervenes in, the largest risk and whether the bounds always held.
"""
import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "outputs" / "17_certificate_estimated"
PROC = ("oracle", "plugin", "hoeffding_serfling", "empirical_bernstein", "ltt_oracle", "ltt_eb")

for regime in ("heterogeneous", "hard"):
    rows = {}
    for f in [D / f"results_{regime}_sequential.json"] + sorted(D.glob(f"results_{regime}_g*_s*.json")):
        for r in json.load(open(f))["rows"]:
            rows.setdefault((r["n"], r["gamma"], r["seed"]), r)
    rows = [rows[k] for k in sorted(rows)]
    json.dump(dict(rows=rows, note="merged: sequential run and per-model parallel runs (own random stream)"),
              open(D / f"results_{regime}.json", "w"), indent=1, default=float)
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        held = min(min(r["bounds_held"].values()) for r in rs)
        txt = " | ".join(f"{p}: intervenes in {sum(r[p]['intervention'] > 0 for r in rs)}/{len(rs)} models "
                         f"(mean rate {sum(r[p]['intervention'] for r in rs) / len(rs):.2f}), max risk {max(r[p]['risk'] for r in rs):.3f}"
                         for p in PROC)
        print(f"{regime} n={n}: bounds held in at least {held:.0%} of calibration cohorts | {txt}")
        for r in rs:
            print(f"    gamma={r['gamma']} seed={r['seed']} rho={r['rho']:.3f}: " +
                  " ".join(f"{p[:6]}={r[p]['intervention']:.2f}" for p in PROC))
