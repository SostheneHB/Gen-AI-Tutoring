"""Numerical check of Theorem (group modularity as a centered Koopman pairing).

For random connected weighted graphs, random community partitions and random
sensitive labels, compares the configuration-null (group) modularity computed
from its definition with the centered random-walk pairing
<1_U, P_0 1_W>_pi, and checks B = D P_0, detailed balance and the
self-adjointness of P_0 in L^2(pi).

    python verify_modularity.py --cases 1334 --out modularity_check.json
"""
import argparse
import json

import numpy as np


def random_graph(rng, n):
    while True:
        W = rng.random((n, n)) * (rng.random((n, n)) < rng.uniform(0.2, 0.8))
        A = np.triu(W, 1)
        A = A + A.T
        if (A.sum(1) > 0).all():
            # connectivity through powers of the adjacency pattern
            R = (A > 0).astype(float) + np.eye(n)
            if (np.linalg.matrix_power(R, n) > 0).all():
                return A


def check(rng):
    n = int(rng.integers(4, 40))
    A = random_graph(rng, n)
    k = A.sum(1)
    two_m = k.sum()
    pi = k / two_m
    P = A / k[:, None]
    P0 = P - np.outer(np.ones(n), pi)
    B = A - np.outer(k, k) / two_m
    D = np.diag(k)
    Dpi = np.diag(pi)
    K = int(rng.integers(1, min(6, n) + 1))
    comm = rng.integers(0, K, n)
    red = rng.random(n) < rng.uniform(0.1, 0.9)
    err = {"B_eq_DP0": float(np.abs(B - D @ P0).max()),
           "detailed_balance": float(np.abs(Dpi @ P - (Dpi @ P).T).max()),
           "P0_selfadjoint": float(np.abs(Dpi @ P0 - (Dpi @ P0).T).max()),
           "modularity": 0.0}
    Pi0 = np.eye(n) - np.outer(np.ones(n), pi)
    for c in range(K):
        C = (comm == c).astype(float)
        for U in (C, C * red, C * ~red):
            direct = U @ B @ C / two_m
            pairing = U @ Dpi @ P0 @ C
            centered = (Pi0 @ U) @ Dpi @ P0 @ (Pi0 @ C)
            err["modularity"] = max(err["modularity"], abs(direct - pairing), abs(direct - centered))
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=1334)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--out", default="modularity_check.json")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    errs = [check(rng) for _ in range(a.cases)]
    out = {"cases": a.cases, "seed": a.seed,
           **{f"max_{k}": max(e[k] for e in errs) for k in errs[0]}}
    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
