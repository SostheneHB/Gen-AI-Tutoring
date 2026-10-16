"""Experiment 01: where does the gap come from in the simulator?

    python code/scripts/run_decomposition_simulator.py      (from the project root)

For each group strength gamma and each replicate: fit one lifted linear model per
group on exploratory data, run three policies on the true simulator (no teaching,
original MPC, projected MPC), and decompose the final gap in disparity d and in
knowledge k into I (initial), D (dynamics), R (response), P (policy), plus the model
error E = observed gap - model gap.  Writes outputs/01_decomposition_simulator/results.json
(aggregate numbers only).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from p2.decomposition import FACTORS, fit_group_models, gap_decomposition, proposition1_terms  # noqa: E402
from p2.paths import PATHS, use_paper1_code  # noqa: E402

use_paper1_code()
import synthetic_regimes as sr  # noqa: E402

SEEDS = range(10)
GAMMAS = (0.5, 1.0, 2.0)
T = 25


def run_policy(fa, model, pol, init, noise, groups):
    x, S, U = init.copy(), [init.copy()], []
    for t in range(T):
        u = np.zeros(len(x)) if pol is None else pol(model.encode(x))
        U.append(u)
        x = fa.step(x, u, groups, noise[:, t])
        S.append(x.copy())
    return np.stack(S, 1), np.stack(U, 1)


def one(fa, gamma, seed):
    fa.step = sr.make_step(gamma)
    model = fa.fit_model(seed, 4, 3)
    st, ctrl, grp = fa.training_data(seed)
    Z = model.encode(st[:, :-1].reshape(-1, 4))
    Zn = model.encode(st[:, 1:].reshape(-1, 4))
    G = np.repeat(grp, st.shape[1] - 1)
    models = fit_group_models(Z, ctrl.reshape(-1), Zn, G)
    init, noise, groups = fa.make_cohort(seed, horizon=T)
    arms = {"no_teaching": None,
            "mpc_original": fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset, beta=8.0),
            "mpc_projected": fa.MPCPolicy(*model.dynamics(True), model.decoder, model.decoder_offset, beta=8.0)}
    res = {}
    for name, pol in arms.items():
        S, U = run_policy(fa, model, pol, init, noise, groups)
        m0 = {g: model.encode(S[groups == g, 0]).mean(0) for g in (0, 1)}
        v = {g: U[groups == g].mean(0) for g in (0, 1)}
        res[name] = {}
        for out, idx in (("disparity", fa.D_IDX), ("knowledge", 0)):
            c, c0 = model.decoder[idx], model.decoder_offset[idx]
            dec = gap_decomposition(models, m0, v, c, c0)
            obs = float(S[groups == 1, -1, idx].mean() - S[groups == 0, -1, idx].mean())
            res[name][out] = dict(observed=obs, model=dec["gap"], error=obs - dec["gap"],
                                  shapley=dec["shapley"], sequential=dec["sequential"],
                                  proposition1=proposition1_terms(models, m0, v, c))
    fa.step = sr.make_step(1.0)
    return res


def summarise(vals):
    v = np.asarray(vals, float)
    h = 2.262 * v.std(ddof=1) / np.sqrt(len(v))      # t(0.975, 9)
    return dict(mean=float(v.mean()), ci=[float(v.mean() - h), float(v.mean() + h)])


def main():
    fa = sr.load_sim(str(PATHS["simulator"]))
    out = {}
    for gamma in GAMMAS:
        runs = [one(fa, gamma, s) for s in SEEDS]
        out[str(gamma)] = {}
        for arm in runs[0]:
            out[str(gamma)][arm] = {}
            for o in ("disparity", "knowledge"):
                r = [x[arm][o] for x in runs]
                entry = {k: summarise([x[k] for x in r]) for k in ("observed", "model", "error")}
                entry["shapley"] = {f: summarise([x["shapley"][f] for x in r]) for f in FACTORS}
                entry["proposition1"] = {f: summarise([x["proposition1"][f] for x in r]) for f in FACTORS}
                out[str(gamma)][arm][o] = entry
                sh = entry["shapley"]
                print(f"gamma={gamma} {arm:13s} {o:9s} observed {entry['observed']['mean']:+.3f} model {entry['model']['mean']:+.3f} "
                      f"| Shapley I {sh['I']['mean']:+.3f} D {sh['D']['mean']:+.3f} R {sh['R']['mean']:+.3f} P {sh['P']['mean']:+.3f} "
                      f"| error {entry['error']['mean']:+.3f}", flush=True)
    d = ROOT / "outputs" / "01_decomposition_simulator"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
