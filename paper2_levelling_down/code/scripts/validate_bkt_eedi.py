"""Experiment 22, step 3: how good is the Eedi-fitted knowledge tracing?  (fifth report, point 7)

    python code/scripts/validate_bkt_eedi.py [--users 4000] [--boot 10]

For each group (PremiumPupil 1 / 0):
  1. held-out prediction: the parameters fitted on 4,000 training pupils (same sample and seed as fit_bkt_eedi.py)
     predict the answers of 2,000 other pupils, answer by answer (forward filter, before each answer is seen):
     log-likelihood per answer, AUC and expected calibration error (10 bins), against two baselines: a constant
     rate per group, and BKT without forgetting (F = 0) refitted on the same training pupils;
  2. delays: the calibration of the predictions by the time since the previous answer of the same sequence
     (< 1 day, 1-7 days, 7-30 days, > 30 days): mean predicted against observed correctness;
  3. skill grouping: the fit is repeated with a coarser skill (the subject of level 2 instead of the most specific);
  4. parameter uncertainty: the fit is repeated on 10 bootstrap samples of pupils, and each parameter set is run
     through the semi-synthetic system (kappa = 0.5, fitted model of seed 0): true gap, the narrowing of the
     full plan (capacity on the system) and the extra budget for half of what the model can close.
Writes outputs/22_semisynthetic/validation.json (aggregate numbers only; the Eedi data are not redistributed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "scripts"))
from fit_bkt_eedi import NAMES, UPPER, nll  # noqa: E402
from p2.paths import PATHS  # noqa: E402

RAW = PATHS["eedi_raw"]


def load_all(users_train, users_test, rng):
    st = pd.read_csv(RAW / "metadata" / "student_metadata_task_1_2.csv", usecols=["UserId", "PremiumPupil"]).dropna()
    st["g"] = st.PremiumPupil.astype(int)
    # same training sample as fit_bkt_eedi.py (same generator, same calls), then a disjoint test sample
    train = pd.concat([d.sample(min(users_train, len(d)), random_state=int(rng.integers(1 << 30))) for _, d in st.groupby("g")])
    rest = st[~st.UserId.isin(train.UserId)]
    test = pd.concat([d.sample(min(users_test, len(d)), random_state=int(rng.integers(1 << 30))) for _, d in rest.groupby("g")])
    keep = pd.concat([train.assign(split="train"), test.assign(split="test")])
    tr = pd.read_csv(RAW / "train_data" / "train_task_1_2.csv", usecols=["QuestionId", "UserId", "AnswerId", "IsCorrect"])
    tr = tr[tr.UserId.isin(keep.UserId)]
    ids = set(tr.AnswerId.astype("int64"))
    parts = []
    for ch in pd.read_csv(RAW / "metadata" / "answer_metadata_task_1_2.csv", usecols=["AnswerId", "DateAnswered"], chunksize=2_000_000):
        ch = ch.dropna(subset=["AnswerId"])
        ch["AnswerId"] = ch.AnswerId.astype("int64")
        parts.append(ch[ch.AnswerId.isin(ids)])
    tr = tr.merge(pd.concat(parts), on="AnswerId")
    tr["date"] = pd.to_datetime(tr.DateAnswered)
    subj = pd.read_csv(RAW / "metadata" / "subject_metadata.csv", encoding="utf-8-sig").set_index("SubjectId").Level
    q = pd.read_csv(RAW / "metadata" / "question_metadata_task_1_2.csv")
    lists = q.SubjectId.map(eval)
    q["skill"] = lists.map(lambda s: max(s, key=lambda x: subj.get(x, -1)))
    q["skill2"] = lists.map(lambda s: next((x for x in s if subj.get(x, -1) == 2), max(s, key=lambda x: subj.get(x, -1))))
    tr = tr.merge(q[["QuestionId", "skill", "skill2"]], on="QuestionId").merge(keep[["UserId", "g", "split"]], on="UserId")
    return tr


def seqs(df, skill):
    """Padded answers Y and delays (days since the previous answer of the sequence; -1 for the first)."""
    df = df.sort_values(["UserId", skill, "date", "AnswerId"])
    out, gaps = [], []
    for _, s in df.groupby(["UserId", skill], sort=False):
        if 3 <= len(s) <= 50:
            out.append(s.IsCorrect.to_numpy(np.int8))
            dt = s.date.diff().dt.total_seconds().to_numpy() / 86400
            gaps.append(np.nan_to_num(dt, nan=-1.0))
    L = max(len(y) for y in out)
    Y, G = np.full((len(out), L), -1, np.int8), np.full((len(out), L), -1.0)
    for i, (y, g) in enumerate(zip(out, gaps)):
        Y[i, :len(y)], G[i, :len(y)] = y, g
    return Y, G


def fit(Y, fix_F=None):
    best = None
    for start in ([0.4, 0.1, 0.02, 0.2, 0.1], [0.6, 0.2, 0.05, 0.25, 0.15]):
        x0 = logit(np.array(start) / UPPER)
        if fix_F is not None:
            f = lambda x: nll(np.r_[x[:2], logit(max(fix_F, 1e-9) / UPPER[2]), x[2:]], Y)
            r = minimize(f, np.r_[x0[:2], x0[3:]], method="L-BFGS-B")
            x = np.r_[r.x[:2], logit(max(fix_F, 1e-9) / UPPER[2]), r.x[2:]]
        else:
            r = minimize(nll, x0, args=(Y,), method="L-BFGS-B")
            x = r.x
        if best is None or r.fun < best[1]:
            best = (x, r.fun)
    return dict(zip(NAMES, (expit(best[0]) * UPPER).tolist()))


def predict(par, Y):
    """P(correct) before each answer (forward filter)."""
    L0, T, F, G, S = (par[k] for k in NAMES)
    p = np.full(Y.shape[0], L0)
    P = np.full(Y.shape, np.nan)
    for t in range(Y.shape[1]):
        y = Y[:, t]
        obs = y >= 0
        pc = p * (1 - S) + (1 - p) * G
        P[:, t] = np.where(obs, pc, np.nan)
        post = np.where(y == 1, p * (1 - S) / np.clip(pc, 1e-12, None), p * S / np.clip(1 - pc, 1e-12, None))
        post = np.where(obs, post, p)
        p = post * (1 - F) + (1 - post) * T
    return P


def scores(P, Y):
    m = Y >= 0
    p, y = np.clip(P[m], 1e-6, 1 - 1e-6), Y[m].astype(float)
    ll = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
    return dict(loglik_per_answer=ll, auc=auc, ece=ece, answers=int(m.sum()))


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--users", type=int, default=4000)
    a_.add_argument("--test-users", type=int, default=2000)
    a_.add_argument("--boot", type=int, default=10)
    a = a_.parse_args()
    rng = np.random.default_rng(20261016)
    df = load_all(a.users, a.test_users, rng)
    out = {}
    boot_sets = {0: [], 1: []}
    for g in (0, 1):
        dg = df[df.g == g]
        Ytr, _ = seqs(dg[dg.split == "train"], "skill")
        Yte, Gte = seqs(dg[dg.split == "test"], "skill")
        par = fit(Ytr)
        par0 = fit(Ytr, fix_F=0.0)
        rate = float((Ytr[Ytr >= 0] == 1).mean())
        P = predict(par, Yte)
        res = dict(params=par, params_no_forgetting=par0, test_sequences=int(len(Yte)),
                   bkt=scores(P, Yte), bkt_no_forgetting=scores(predict(par0, Yte), Yte),
                   constant=scores(np.where(Yte >= 0, rate, np.nan), Yte))
        m = Yte >= 0
        cal = {}
        for name, lo, hi in (("first", -1.5, -0.5), ("<1 day", 0, 1), ("1-7 days", 1, 7), ("7-30 days", 7, 30), (">30 days", 30, 1e9)):
            sel = m & (Gte >= lo) & (Gte < hi)
            cal[name] = dict(n=int(sel.sum()), predicted=float(P[sel].mean()) if sel.any() else None,
                             observed=float((Yte[sel] == 1).mean()) if sel.any() else None)
        res["calibration_by_delay"] = cal
        Ytr2, _ = seqs(dg[dg.split == "train"], "skill2")
        res["params_coarser_skill"] = fit(Ytr2)
        for _ in range(a.boot):
            boot_sets[g].append(fit(Ytr[rng.integers(0, len(Ytr), len(Ytr))]))
        out[str(g)] = res
        print(f"group {g}: held-out loglik {res['bkt']['loglik_per_answer']:.4f} (no forgetting {res['bkt_no_forgetting']['loglik_per_answer']:.4f}, "
              f"constant {res['constant']['loglik_per_answer']:.4f}) AUC {res['bkt']['auc']:.3f} ECE {res['bkt']['ece']:.3f} | coarser skill "
              + ", ".join(f"{k} {v:.3f}" for k, v in res["params_coarser_skill"].items()), flush=True)
        print("   calibration by delay:", {k: (v["n"], None if v["predicted"] is None else round(v["predicted"], 3), None if v["observed"] is None else round(v["observed"], 3)) for k, v in cal.items()}, flush=True)
    # 4. parameter uncertainty through the semi-synthetic system
    from p2 import semisim
    from p2.config import REF_HELP
    from p2.frontier import GroupModel, efficient_plan, knapsack_plan, knapsack_price
    from p2.paths import use_paper1_code
    use_paper1_code()
    import pareto_fair as pf
    import synthetic_regimes as sr
    fa = sr.load_sim(str(PATHS["simulator"]))
    T = 25
    prop = []
    for b in range(a.boot):
        par = {0: boot_sets[0][b], 1: boot_sets[1][b]}
        semisim.patch(fa, par, 0.5, 0)
        model = fa.fit_model(0, 4, 3)
        A, B, a2, aG = pf.fit_group_aware(fa, model, 0)
        init, _, groups = fa.make_cohort(0, horizon=T)
        n = {g: int((groups == g).sum()) for g in (0, 1)}
        m0 = {g: model.encode(init[groups == g]).mean(0) for g in (0, 1)}
        M = GroupModel(A, B, a2, aG, m0, model.decoder[0], model.decoder_offset[0], model.decoder[0], model.decoder_offset[0], n, T)
        v_ref = efficient_plan(M, REF_HELP * T * sum(n.values()), "min_gap")
        _, _, gap, B_ref = M.outcomes(v_ref)
        cap = knapsack_price(M, v_ref, 0.0)[1]
        close = min(cap, abs(gap))
        full, _ = knapsack_plan(M, v_ref, knapsack_price(M, v_ref, close)[0])
        half_extra = knapsack_price(M, v_ref, 0.5 * close)[0] / B_ref
        tg, tn = [], []
        for j in range(10):
            init_, noise, grp = fa.make_cohort(0, horizon=T, role="evaluation", index=200 + j)
            step = semisim.cohort_step(fa, par, 0.5, 0, "evaluation", 200 + j, len(grp))
            res_ = []
            for plan in (v_ref, full):
                x = init_.copy()
                for t in range(T):
                    x = step(x, np.where(grp == 1, plan[1][t], plan[0][t]), grp, noise[:, t])
                res_.append(x[grp == 1, 0].mean() - x[grp == 0, 0].mean())
            tg.append(res_[0]); tn.append(abs(res_[0]) - abs(res_[1]))
        prop.append(dict(true_gap=float(np.mean(tg)), full_narrowing=float(np.mean(tn)), half_extra_budget=float(half_extra)))
        print(f"bootstrap {b}: true gap {prop[-1]['true_gap']:+.3f} full narrowing {prop[-1]['full_narrowing']:.3f} half extra {half_extra:.2f}", flush=True)
    out["propagation_kappa_0.5"] = dict(runs=prop, ranges={k: [float(min(p[k] for p in prop)), float(max(p[k] for p in prop))]
                                                          for k in ("true_gap", "full_narrowing", "half_extra_budget")})
    json.dump(out, open(ROOT / "outputs" / "22_semisynthetic" / "validation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
