"""Conformal certificate that no group loses on the real system (Theorem 3 of the manuscript).

A finite family of plans is fixed from the model only: plan w^delta is the cheapest change
of the reference with no predicted knowledge loss and a predicted gap reduction of at least
rho / (1 - delta).  On each calibration cohort the reference and every plan of the family are
applied to the same students, and the score is the worst relative shortfall of the promise
    s = max_delta ( 1 - rho_true(w^delta) / rho_hat(w^delta) ),
or +infinity if a group loses knowledge under some plan of the family.  The margin q is the
ceil((m+1)(1-alpha))-th smallest score.  The rule applies the plan with the smallest
delta >= q (or keeps the reference if there is none).
"""
import math

import numpy as np
from scipy.optimize import linprog


def margin_plan(M, v_ref, rho, delta, slack=0.0):
    """min budget  s.t.  h^k_g . w_g >= slack (g = 0, 1),  gap reduction >= rho / (1 - delta), box."""
    T, n = M.T, M.n
    _, _, gap_ref, _ = M.outcomes(v_ref)
    sgn = -np.sign(gap_ref)
    c = np.r_[np.full(T, n[0]), np.full(T, n[1])]
    z = np.zeros(T)
    A_ub = [np.r_[-M.hk[0], z], np.r_[z, -M.hk[1]], -sgn * np.r_[-M.hd[0], M.hd[1]], sgn * np.r_[-M.hd[0], M.hd[1]]]
    b_ub = [-slack, -slack, -rho / (1 - delta), abs(gap_ref)]          # narrowing in [rho/(1-delta), |gap_ref|]
    bounds = [(-v_ref[0][t], 1 - v_ref[0][t]) for t in range(T)] + [(-v_ref[1][t], 1 - v_ref[1][t]) for t in range(T)]
    r = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, bounds=bounds, method="highs")
    if r.status != 0:
        return None
    return {0: v_ref[0] + r.x[:T], 1: v_ref[1] + r.x[T:]}


def predicted_change(M, v_ref, v):
    """(dK0_hat, dK1_hat, rho_hat): the model's promise, which does not depend on the start."""
    K, _, gap, _ = M.outcomes(v)
    K_ref, _, gap_ref, _ = M.outcomes(v_ref)
    return K[0] - K_ref[0], K[1] - K_ref[1], abs(gap_ref) - abs(gap)


def conformal_margin(scores, alpha):
    m = len(scores)
    k = math.ceil((m + 1) * (1 - alpha))
    return float("inf") if k > m else float(np.sort(scores)[k - 1])


def score(pred, true):
    """Relative shortfall of the promised gap reduction; +inf if a group loses knowledge."""
    if min(true[0], true[1]) < 0:
        return float("inf")
    return 1 - true[2] / pred[2]
