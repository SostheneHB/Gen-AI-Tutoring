"""The certificate with estimated effects (Theorem 4 of the manuscript).

On a calibration cohort the true changes are not observed.  The students of each group are split at random into
equal arms (the reference and each plan of the family: complete randomization within the group), and each arm
mean is a sample without replacement from the cohort's potential outcomes.  For every group g and arm a we build
one-sided bounds on the cohort mean mu_{g,a} that hold simultaneously with probability >= 1 - beta (Bonferroni over
the 2 x arms x 2 one-sided statements), and the conservative score is the supremum of the score over this box:

    change         Delta_g(w) in [lo(mu_{g,w}) - hi(mu_{g,ref}), hi(mu_{g,w}) - lo(mu_{g,ref})]
                   (exactly 0 if the plan of g is the reference: same students, no interference)
    narrowing      rho_true(w) = |d| - |d + x| >= its minimum over d (reference gap) and x = Delta_1 - Delta_0
    score          +inf if some loss bound is < 0, else max_w (1 - inf rho_true(w) / rho_hat(w)).

Bounds (outcomes in [0, 1], n draws without replacement from N values):
  hoeffding_serfling   (b - a) sqrt(rho_n log(1/d) / (2 n))                      Bardenet & Maillard (2015), Cor. 2.5
  empirical_bernstein  sigma_hat sqrt(2 rho_n log(1/d') / n) + kappa log(1/d') / n,  d' = d / 5, kappa = 7/3 + 3/sqrt 2
                       (sigma_hat with divisor n)                                  Bardenet & Maillard (2015), Thm 4.3
"""
import math

import numpy as np

KAPPA = 7 / 3 + 3 / math.sqrt(2)


def rho_n(n, N):
    return 1 - (n - 1) / N if n <= N / 2 else (1 - n / N) * (1 + 1 / n)


def width(x, N, d, method):
    """One-sided deviation bound of the mean of the sample x (values in [0, 1]) at error probability d."""
    n = len(x)
    r = rho_n(n, N)
    if method == "hoeffding_serfling":
        return math.sqrt(r * math.log(1 / d) / (2 * n))
    if method == "empirical_bernstein":
        dd = d / 5
        return float(np.std(x)) * math.sqrt(2 * r * math.log(1 / dd) / n) + KAPPA * math.log(1 / dd) / n
    raise ValueError(method)


def assign(rng, groups, arms):
    """Complete randomization within each group into `arms` arms of equal size; returns the arm of each student
    (students left over by the division are not used: arm -1)."""
    out = np.full(len(groups), -1)
    for g in (0, 1):
        idx = rng.permutation(np.where(groups == g)[0])
        size = len(idx) // arms
        for a in range(arms):
            out[idx[a * size:(a + 1) * size]] = a
    return out


def arm_estimates(Y, groups, arm, beta, method):
    """Y: (arms, students) potential outcomes; only Y[arm[i], i] is used.  Returns the point estimates and the
    simultaneous lower and upper bounds of every group-arm mean (shape (2, arms))."""
    A = Y.shape[0]
    J = 2 * A * 2
    est, lo, hi = np.zeros((2, A)), np.zeros((2, A)), np.zeros((2, A))
    for g in (0, 1):
        N = int((groups == g).sum())
        for a in range(A):
            x = Y[a, (groups == g) & (arm == a)]
            est[g, a] = x.mean()
            if method == "plugin":
                lo[g, a] = hi[g, a] = est[g, a]
            else:
                w = width(x, N, beta / J, method)
                lo[g, a], hi[g, a] = max(est[g, a] - w, 0.0), min(est[g, a] + w, 1.0)
    return est, lo, hi


def interval_abs(lo, hi):
    """min and max of |x| for x in [lo, hi]."""
    mn = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    return mn, max(abs(lo), abs(hi))


def inf_narrowing(d_lo, d_hi, x_lo, x_hi):
    """min of |d| - |d + x| over d in [d_lo, d_hi], x in [x_lo, x_hi] (d: reference gap, x: change of the gap).
    The function is piecewise linear in each variable, so the minimum is at a bound or at a kink."""
    best = np.inf
    for d in [d_lo, d_hi] + ([0.0] if d_lo < 0 < d_hi else []):
        xs = [x_lo, x_hi] + ([-d] if x_lo < -d < x_hi else [])
        for x in xs:
            best = min(best, abs(d) - abs(d + x))
    return best


def conservative_score(lo, hi, pred, unchanged):
    """lo, hi: (2, arms) bounds, arm 0 = reference; pred[j] = (dK0_hat, dK1_hat, rho_hat) of plan j (arm j + 1);
    unchanged[j][g] = True if plan j gives group g the reference controls: then its change is exactly 0 (no
    interference, same students), and it is not estimated."""
    d_lo, d_hi = lo[1, 0] - hi[0, 0], hi[1, 0] - lo[0, 0]          # reference gap (group 1 minus group 0)
    s = -np.inf
    for j, p in enumerate(pred):
        a = j + 1
        ch = {}
        for g in (0, 1):
            ch[g] = (0.0, 0.0) if unchanged[j][g] else (lo[g, a] - hi[g, 0], hi[g, a] - lo[g, 0])
            if ch[g][0] < 0:
                return np.inf
        x_lo, x_hi = ch[1][0] - ch[0][1], ch[1][1] - ch[0][0]       # change of the gap
        s = max(s, 1 - inf_narrowing(d_lo, d_hi, x_lo, x_hi) / p[2])
    return float(np.round(s, 12))
