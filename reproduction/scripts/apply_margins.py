"""
Post-hoc application of the margin rule of Table tab:margins (referee report 3, item 4). The margins were NOT fixed
before the experiments; they are chosen here, once, and applied to every split:
    eps_abs = 0.01 (knowledge-score units), eps_rel = 10% of |delta_ref|, eps_Delta = max(eps_abs, eps_rel |delta_ref|),
    eps_MSE = 10% of the one-step MSE of the raw EDMD model in the same split,
with delta_ref the observed held-out terminal gap (the training gap is not stored; with these values eps_abs
dominates in every split).
Error on the gap: the school/class-cluster interval of (predicted - observed) terminal gap of each model.
Increase of the MSE: the paired one-step MSE difference (knowledge) with the raw model and its cluster interval.
A split has a "clear gap" when the interval of its observed terminal gap excludes zero.

    DATA_ROOT=/path/to/root python apply_margins.py      -> out_reviewer/margins.json (aggregates only)
"""
import json
import os

ROOT = os.environ.get("DATA_ROOT", ".")
SETS = {"ASSISTments": [f"out_time/school_disjoint_{j}/results.json" for j in range(5)],
        "Eedi": [f"{ROOT}/Datasets/Eedi_NeurIPS2020/out_eedi/class_disjoint_{j}/results.json" for j in range(5)],
        **{f"OULAD {a}": [f"{ROOT}/Datasets/OULAD/out_oulad/{a}/{a}_student_split_{j}/results.json" for j in range(5)]
           for a in ("imd", "disability", "gender")}}
MODELS = {"edmd_proj": "projected", "edmd_groupcal": "recalibrated per group", "edmd_raw": "raw"}
EPS_ABS, EPS_REL, EPS_MSE_REL = 0.01, 0.10, 0.10


def gap_decision(ci, eps):
    lo, hi = ci
    if -eps <= lo and hi <= eps:
        return "acceptable"
    if hi < -eps or lo > eps:
        return "matters"
    return "inconclusive"


def mse_decision(ci, eps):
    lo, hi = ci
    if hi <= eps:
        return "acceptable"
    if lo > eps:
        return "matters"
    return "inconclusive"


def main():
    out = {"margins": dict(eps_abs=EPS_ABS, eps_rel=EPS_REL, eps_mse_rel=EPS_MSE_REL), "splits": [], "counts": {}}
    for name, paths in SETS.items():
        for p in paths:
            r = json.load(open(p))["replay"]
            obs, obs_ci = r["terminal_gap_observed"], r["terminal_gap_observed_school_ci"]
            clear = obs_ci[0] > 0 or obs_ci[1] < 0
            eps_d = max(EPS_ABS, EPS_REL * abs(obs))
            eps_m = EPS_MSE_REL * r["models"]["edmd_raw"]["mse"]["1"]["k_t"]
            rec = dict(dataset=name, split=p.split("/")[-2], observed_gap=obs, clear_gap=clear, eps_delta=eps_d, eps_mse=eps_m)
            for m in MODELS:
                g = r["models"][m]["gap_error_school_ci"]
                d = dict(gap_error=r["models"][m]["gap_error"], gap_ci=g, gap_decision=gap_decision(g, eps_d))
                if m != "edmd_raw":
                    pm = r["paired_vs_raw"][m]["1"]
                    d.update(mse_diff=pm["diff"], mse_ci=pm["school_ci"], mse_decision=mse_decision(pm["school_ci"], eps_m))
                rec[m] = d
            out["splits"].append(rec)
    clear = [s for s in out["splits"] if s["clear_gap"]]
    for m in MODELS:
        c = {}
        for key in ("gap_decision", "mse_decision"):
            if key in clear[0][m]:
                for s in clear:
                    c.setdefault(key, {}).setdefault(s[m][key], 0)
                    c[key][s[m][key]] += 1
        out["counts"][MODELS[m]] = c
    out["n_splits"], out["n_clear"] = len(out["splits"]), len(clear)
    out["clear_by_dataset"] = {n: sum(s["clear_gap"] for s in clear if s["dataset"] == n) for n in SETS}
    os.makedirs("out_reviewer", exist_ok=True)
    json.dump(out, open("out_reviewer/margins.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_splits", "n_clear", "clear_by_dataset", "counts")}, indent=1))


if __name__ == "__main__":
    main()
