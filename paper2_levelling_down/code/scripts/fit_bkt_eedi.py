"""Experiment 22, step 1: group-specific knowledge tracing (BKT with forgetting) fitted on Eedi.

    python code/scripts/fit_bkt_eedi.py [--users 4000] [--boot 10]

Groups: PremiumPupil = 1 (a UK marker of economic disadvantage) and 0; pupils with a missing flag are left out.
A skill is the most specific subject of a question (the subject of highest level in its list).  A sequence is the
answers of one pupil to the questions of one skill, in time order; sequences of 3 to 50 answers are kept.  For each
group, a random sample of pupils is used, and the five BKT parameters (initial mastery L0, learning T, forgetting F,
guess G, slip S; G, S < 0.5) are fitted by maximum likelihood (forward algorithm, L-BFGS-B), pooled over skills.
Uncertainty: the fit is repeated on bootstrap samples of pupils.
Writes outputs/22_semisynthetic/bkt_params.json: ten numbers per group and their bootstrap ranges (no data file).
The Eedi data are used under CC BY-NC-ND 4.0 and never redistributed.
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
from p2.paths import PATHS  # noqa: E402

RAW = PATHS["eedi_raw"]
NAMES = ("L0", "T", "F", "G", "S")
UPPER = np.array([1.0, 1.0, 1.0, 0.5, 0.5])


def load(users_per_group, rng):
    st = pd.read_csv(RAW / "metadata" / "student_metadata_task_1_2.csv", usecols=["UserId", "PremiumPupil"]).dropna()
    st["g"] = st.PremiumPupil.astype(int)
    keep = pd.concat([d.sample(min(users_per_group, len(d)), random_state=int(rng.integers(1 << 30))) for _, d in st.groupby("g")])
    tr = pd.read_csv(RAW / "train_data" / "train_task_1_2.csv", usecols=["QuestionId", "UserId", "AnswerId", "IsCorrect"])
    tr = tr[tr.UserId.isin(keep.UserId)]
    ids = set(tr.AnswerId.astype("int64"))
    parts = []
    for ch in pd.read_csv(RAW / "metadata" / "answer_metadata_task_1_2.csv", usecols=["AnswerId", "DateAnswered"], chunksize=2_000_000):
        ch = ch.dropna(subset=["AnswerId"])
        ch["AnswerId"] = ch.AnswerId.astype("int64")
        parts.append(ch[ch.AnswerId.isin(ids)])
    tr = tr.merge(pd.concat(parts), on="AnswerId")
    subj = pd.read_csv(RAW / "metadata" / "subject_metadata.csv", encoding="utf-8-sig").set_index("SubjectId").Level
    q = pd.read_csv(RAW / "metadata" / "question_metadata_task_1_2.csv")
    q["skill"] = q.SubjectId.map(lambda s: max(eval(s), key=lambda x: subj.get(x, -1)))
    tr = tr.merge(q[["QuestionId", "skill"]], on="QuestionId").merge(keep[["UserId", "g"]], on="UserId")
    tr = tr.sort_values(["UserId", "skill", "DateAnswered", "AnswerId"])
    return tr


def sequences(df):
    seqs = {}
    for g, d in df.groupby("g"):
        out = []
        for (_, _), s in d.groupby(["UserId", "skill"], sort=False):
            y = s.IsCorrect.to_numpy(np.int8)
            if 3 <= len(y) <= 50:
                out.append(y)
        L = max(len(y) for y in out)
        Y = np.full((len(out), L), -1, np.int8)
        for i, y in enumerate(out):
            Y[i, :len(y)] = y
        seqs[g] = Y
    return seqs


def nll(theta, Y):
    L0, T, F, G, S = expit(theta) * UPPER
    p = np.full(Y.shape[0], L0)
    ll = 0.0
    for t in range(Y.shape[1]):
        y = Y[:, t]
        obs = y >= 0
        pc = p * (1 - S) + (1 - p) * G
        lik = np.where(y == 1, pc, 1 - pc)
        ll += np.log(np.clip(lik[obs], 1e-12, None)).sum()
        post = np.where(y == 1, p * (1 - S) / np.clip(pc, 1e-12, None), p * S / np.clip(1 - pc, 1e-12, None))
        post = np.where(obs, post, p)
        p = post * (1 - F) + (1 - post) * T
    return -ll


def fit(Y):
    best = None
    for start in ([0.4, 0.1, 0.02, 0.2, 0.1], [0.6, 0.2, 0.05, 0.25, 0.15]):
        x0 = logit(np.array(start) / UPPER)
        r = minimize(nll, x0, args=(Y,), method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return dict(zip(NAMES, (expit(best.x) * UPPER).tolist())), float(best.fun)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--users", type=int, default=4000)
    a_.add_argument("--boot", type=int, default=10)
    a = a_.parse_args()
    rng = np.random.default_rng(20261016)
    df = load(a.users, rng)
    seqs = sequences(df)
    out = {}
    for g, Y in seqs.items():
        point, ll = fit(Y)
        boots = []
        for _ in range(a.boot):
            idx = rng.integers(0, len(Y), len(Y))
            boots.append(fit(Y[idx])[0])
        out[str(g)] = dict(params=point, neg_loglik=ll, sequences=int(len(Y)), answers=int((Y >= 0).sum()),
                           first_answer_correct=float((Y[:, 0] == 1).mean()),
                           boot_range={k: [float(min(b[k] for b in boots)), float(max(b[k] for b in boots))] for k in NAMES})
        print(f"group {g} (PremiumPupil={g}): {len(Y)} sequences, " + ", ".join(f"{k} {point[k]:.3f} [{out[str(g)]['boot_range'][k][0]:.3f}, "
                                                                                 f"{out[str(g)]['boot_range'][k][1]:.3f}]" for k in NAMES), flush=True)
    d = ROOT / "outputs" / "22_semisynthetic"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(dict(users_per_group=a.users, boot=a.boot, groups=out), open(d / "bkt_params.json", "w"), indent=1)


if __name__ == "__main__":
    main()
