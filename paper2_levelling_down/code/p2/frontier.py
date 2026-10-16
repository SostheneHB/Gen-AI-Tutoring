"""The price of fairness over a horizon: exact linear programs on the group-aware model.

Group g has mean controls v_g in [0,1]^T (non-negative, so the budget n_g * sum_t v_{g,t}
is linear).  Final outcomes are affine in the controls, with Markov parameters
h_{g,t} = c' A^{T-1-t} B:
    knowledge  K_g(v_g) = alpha^k_g + h^k . v_g,   disparity  D_g(v_g) = alpha^d_g + h^d . v_g,
    gap        delta = D_1 - D_0 (the gap in disparity).

efficient_plan : maximise n0 K_0 + n1 K_1 under a budget (the reference that wastes nothing)
fair_frontier  : minimum budget to move the gap by rho with no group losing knowledge;
                 the dual value of the gap constraint is the shadow price of fairness.
"""
import numpy as np
from scipy.optimize import linprog


def markov(A, B, c, T):
    P, out = np.eye(len(B)), []
    for _ in range(T):
        out.append(c @ P @ B)
        P = A @ P
    return np.array(out[::-1])            # h_t for t = 0..T-1


def free_response(A, a, m0, c, T):
    m = np.asarray(m0, float)
    for _ in range(T):
        m = A @ m + a
    return float(c @ m)


class GroupModel:
    """Final knowledge and disparity of each group as affine functions of its mean controls.

    Group-aware model of the companion paper [A] (same A, B; offset a + g*aG).  Use GroupModel.separate for
    fully group-specific dynamics (A_g, B_g, a_g).  Markov parameters are stored per group.
    """

    def __init__(self, A, B, a, aG, m0, ck, ck0, cd, cd0, n, T):
        dyn = {g: (A, B, a + g * aG) for g in (0, 1)}
        self._build(dyn, m0, ck, ck0, cd, cd0, n, T)

    @classmethod
    def separate(cls, models, m0, ck, ck0, cd, cd0, n, T):
        obj = cls.__new__(cls)
        obj._build({g: (models[g]["A"], models[g]["B"], models[g]["a"]) for g in (0, 1)}, m0, ck, ck0, cd, cd0, n, T)
        return obj

    def _build(self, dyn, m0, ck, ck0, cd, cd0, n, T):
        self.T, self.n = T, n
        self.hk = {g: markov(dyn[g][0], dyn[g][1], ck, T) for g in (0, 1)}
        self.hd = {g: markov(dyn[g][0], dyn[g][1], cd, T) for g in (0, 1)}
        self.ak = {g: free_response(dyn[g][0], dyn[g][2], m0[g], ck, T) + ck0 for g in (0, 1)}
        self.ad = {g: free_response(dyn[g][0], dyn[g][2], m0[g], cd, T) + cd0 for g in (0, 1)}

    def outcomes(self, v):
        K = {g: self.ak[g] + self.hk[g] @ v[g] for g in (0, 1)}
        D = {g: self.ad[g] + self.hd[g] @ v[g] for g in (0, 1)}
        return K, D, D[1] - D[0], sum(self.n[g] * v[g].sum() for g in (0, 1))


def efficient_plan(M, budget, rule="solver", tol=1e-11):
    """max n0 K0 + n1 K1  s.t.  n0 sum v0 + n1 sum v1 <= budget, 0 <= v <= 1.

    The efficient plan is often not unique (in the group-aware model both groups have the same h^k).  rule:
      "solver"   the plan returned by HiGHS (used in the experiments of the first versions);
      "min_gap"  among efficient plans (total knowledge >= optimum - tol |optimum|), the one with the smallest |gap|;
      "max_gap"  among efficient plans, the one with the largest |gap| (for the sensitivity analysis only).
    For the last two rules, values within 1e-7 of a bound of the box are set to the bound (numerical clean-up: the
    second program accepts a relative loss of total knowledge of tol, which would otherwise leave entries such as
    1 - 1e-8 that change the sets of steps with room and with teaching).
    """
    snap = lambda x: np.where(np.abs(x) < 1e-7, 0.0, np.where(np.abs(x - 1) < 1e-7, 1.0, x))
    T, n = M.T, M.n
    c = -np.r_[n[0] * M.hk[0], n[1] * M.hk[1]]
    A_ub = [np.r_[np.full(T, n[0]), np.full(T, n[1])]]
    r = linprog(c, A_ub=A_ub, b_ub=[budget], bounds=[(0, 1)] * (2 * T), method="highs")
    if rule == "solver":
        return {0: r.x[:T], 1: r.x[T:]}
    W = -r.fun                                              # optimal value of sum_g n_g h^k_g . v_g
    A_eff = A_ub + [c]
    b_eff = [budget, -W + tol * max(abs(W), 1.0)]
    gv, g0 = np.r_[-M.hd[0], M.hd[1]], M.ad[1] - M.ad[0]    # gap(v) = g0 + gv . v
    if rule == "min_gap":                                   # min s  s.t.  s >= gap, s >= -gap  (variables v, s)
        A2 = [np.r_[a, 0.0] for a in A_eff] + [np.r_[gv, -1.0], np.r_[-gv, -1.0]]
        b2 = b_eff + [-g0, g0]
        r2 = linprog(np.r_[np.zeros(2 * T), 1.0], A_ub=np.array(A2), b_ub=b2, bounds=[(0, 1)] * (2 * T) + [(0, None)], method="highs")
        x = snap(r2.x[:2 * T])
        return {0: x[:T], 1: x[T:]}
    if rule == "max_gap":
        best = None
        for s in (1.0, -1.0):                               # max of s * gap, then keep the larger |gap|
            r2 = linprog(-s * gv, A_ub=np.array(A_eff), b_ub=b_eff, bounds=[(0, 1)] * (2 * T), method="highs")
            val = abs(g0 + gv @ r2.x)
            if best is None or val > best[0]:
                best = (val, r2.x)
        x = snap(best[1])
        return {0: x[:T], 1: x[T:]}
    raise ValueError(rule)


def fair_frontier(M, v_ref, rhos):
    """For each rho: min budget s.t. K_g >= K_g(ref) and the gap moves towards zero by rho.

    Returns a list of dicts with the budget, the plan and the shadow price (d budget / d rho).
    """
    T, n = M.T, M.n
    K_ref, _, gap_ref, B_ref = M.outcomes(v_ref)
    sgn = -np.sign(gap_ref)                      # direction that narrows the gap
    out = []
    for rho in rhos:
        c = np.r_[np.full(T, n[0]), np.full(T, n[1])]
        A_ub = [np.r_[-M.hk[0], np.zeros(T)], np.r_[np.zeros(T), -M.hk[1]],     # floors
                -sgn * np.r_[-M.hd[0], M.hd[1]],                                 # gap moves by >= rho
                sgn * np.r_[-M.hd[0], M.hd[1]]]                                  # ... and by <= |gap_ref| (no overshoot)
        # gap(v) = (ad1 - ad0) + hd.v1 - hd.v0 ;  rho <= sgn * (gap(v) - gap_ref) <= |gap_ref|
        b_ub = [M.ak[0] - K_ref[0], M.ak[1] - K_ref[1],
                sgn * (M.ad[1] - M.ad[0]) - (sgn * gap_ref + rho),
                abs(gap_ref) + sgn * gap_ref - sgn * (M.ad[1] - M.ad[0])]
        r = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, bounds=[(0, 1)] * (2 * T), method="highs")
        if r.status != 0:
            out.append(dict(rho=float(rho), feasible=False))
            continue
        v = {0: r.x[:T], 1: r.x[T:]}
        out.append(dict(rho=float(rho), feasible=True, budget=float(r.fun), extra_budget=float(r.fun - B_ref),
                        shadow_price=float(-r.ineqlin.marginals[2]), plan=v))
    return out


def knapsack_price(M, v_ref, rho):
    """Closed-form price of fairness (Theorem 2 of the manuscript).

    Assumes the channel through group 0 is closed and that extra control for the group
    behind (group 1 when the gap is negative) never lowers its knowledge.  Then the
    cheapest fair plan fills the free room (1 - v_ref) of the group behind in decreasing
    order of h^d: a fractional knapsack.  Returns (extra budget, capacity, initial price).
    """
    _, _, gap_ref, _ = M.outcomes(v_ref)
    g = 1 if gap_ref < 0 else 0                     # the group whose disparity must rise
    h, room = M.hd[g], 1.0 - v_ref[g]
    order = np.argsort(-h)
    need, cost = rho, 0.0
    for t in order:
        if need <= 1e-15:
            break
        if h[t] <= 0 or room[t] <= 1e-12:
            continue
        take = min(room[t], need / h[t])
        cost += M.n[g] * take
        need -= h[t] * take
    capacity = float(sum(h[t] * room[t] for t in range(M.T) if h[t] > 0))
    free = [t for t in order if h[t] > 0 and room[t] > 1e-12]
    price0 = M.n[g] / h[free[0]] if free else float("inf")
    return (cost if need <= 1e-12 else float("inf")), capacity, price0


def welfare_plan(M, v_ref, extra, phi_prime, iters=4000):
    """Prioritarian welfare  max sum_g n_g phi(D_g)  (phi concave) under the knowledge floors,
    the budget B(v_ref) + extra and the box, solved by Frank-Wolfe: each step is an exact LP."""
    T, n = M.T, M.n
    K_ref, _, _, B_ref = M.outcomes(v_ref)
    A_ub = np.array([np.r_[-M.hk[0], np.zeros(T)], np.r_[np.zeros(T), -M.hk[1]],
                     np.r_[np.full(T, n[0]), np.full(T, n[1])]])
    b_ub = [M.ak[0] - K_ref[0], M.ak[1] - K_ref[1], B_ref + extra]
    x = np.r_[v_ref[0], v_ref[1]]
    for k in range(iters):
        D = [M.ad[g] + M.hd[g] @ x[g * T:(g + 1) * T] for g in (0, 1)]
        grad = np.r_[n[0] * phi_prime(D[0]) * M.hd[0], n[1] * phi_prime(D[1]) * M.hd[1]]
        s = linprog(-grad, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1)] * (2 * T), method="highs").x
        gap_fw = grad @ (s - x)
        if gap_fw < 1e-13:
            break
        x = x + 2.0 / (k + 2.0) * (s - x)
    return {0: x[:T], 1: x[T:]}


def best_return_ahead(M, v_ref):
    """Lambda_o of Proposition 3: the largest gain in disparity per unit of budget that the
    group ahead can offer without losing knowledge (raise a step with room, or move teaching
    from a taught step u to a step t with room and pay the difference)."""
    _, _, gap, _ = M.outcomes(v_ref)
    o = 0 if gap < 0 else 1
    rk, rd = M.hk[o], M.hd[o]
    F, S = np.where(v_ref[o] < 1 - 1e-9)[0], np.where(v_ref[o] > 1e-9)[0]
    lam = rd[F].max() / M.n[o] * M.n[o] if len(F) else 0.0
    for t in F:
        for u in S:
            den = 1 / rk[t] - 1 / rk[u]
            if den > 1e-15:
                lam = max(lam, (rd[t] / rk[t] - rd[u] / rk[u]) / den)
    return float(lam)


def knapsack_plan(M, v_ref, extra):
    """The filling of Theorem 2 with a given extra budget. Returns (plan, last step used)."""
    _, _, gap, _ = M.outcomes(v_ref)
    b = 1 if gap < 0 else 0
    v = {g: v_ref[g].copy() for g in (0, 1)}
    left, last = extra / M.n[b], None
    for t in np.argsort(-M.hd[b]):
        if left <= 1e-15:
            break
        if M.hd[b][t] <= 0 or v[b][t] >= 1 - 1e-12:
            continue
        take = min(1 - v[b][t], left)
        v[b][t] += take
        left -= take
        last = t
    return v, last
