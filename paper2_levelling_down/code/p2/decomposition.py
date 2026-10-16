"""Decomposition of the gap between two groups of learners into its sources.

For each group g we fit a lifted linear model  z' = A_g z + B_g u + a_g  (EDMD,
same lifting and decoder for both groups).  With group means m_{g,t} of the lifted
state and group means v_{g,t} of the control, the model gives exactly

    m_{g,t+1} = A_g m_{g,t} + B_g v_{g,t} + a_g,      y_{g,t} = c' m_{g,t} + c0,

and the gap Delta_T = y_{1,T} - y_{0,T} splits into four "factors" that can be
switched from group 0 to group 1:

    I  initial state      m_{g,0}
    D  dynamics           (A_g, a_g)
    R  response to teaching  B_g
    P  policy             (v_{g,t})_t

Two decompositions are given.
  * sequential (Proposition 2 of the manuscript): reference group 0, fixed order;
  * Shapley: average of the marginal effects over the 4! orders in which the
    factors can be switched.  It is the unique split that is additive (the four
    terms sum to the model gap), symmetric and gives zero to a factor that never
    matters (Shapley axioms).
The model error  E = observed gap - model gap  is always reported next to them.
"""
from itertools import permutations
from math import factorial

import numpy as np

FACTORS = ("I", "D", "R", "P")


def fit_group_models(Z, U, Zn, G):
    """Least squares [A B a] for each group, same lifting.  Z, Zn: (N, n); U: (N,); G: (N,)."""
    out = {}
    for g in (0, 1):
        m = G == g
        X = np.column_stack([Z[m], U[m], np.ones(m.sum())])
        C = np.linalg.lstsq(X, Zn[m], rcond=None)[0]
        n = Z.shape[1]
        out[g] = dict(A=C[:n].T, B=C[n], a=C[n + 1])
    return out


def rollout_mean(m0, A, B, a, v):
    """Group-mean trajectory of the linear model with mean controls v (length T)."""
    m = [np.asarray(m0, float)]
    for vt in v:
        m.append(A @ m[-1] + B * vt + a)
    return np.array(m)


def model_outcome(theta, c, c0):
    """Final outcome of the model when each factor is taken from the group given in theta."""
    m0, (A, a), B, v = theta
    return float(c @ rollout_mean(m0, A, B, a, v)[-1] + c0)


def factors_of(models, m0, v, g):
    return (m0[g], (models[g]["A"], models[g]["a"]), models[g]["B"], v[g])


def gap_decomposition(models, m0, v, c, c0=0.0):
    """Return the model gap, the sequential terms and the Shapley terms.

    models: output of fit_group_models; m0[g]: initial mean lifted state of group g;
    v[g]: mean control of group g at t = 0..T-1; c, c0: linear read-out of the outcome.
    """
    f0, f1 = factors_of(models, m0, v, 0), factors_of(models, m0, v, 1)

    def value(switched):              # factors in `switched` come from group 1
        theta = tuple(f1[i] if FACTORS[i] in switched else f0[i] for i in range(4))
        return model_outcome(theta, c, c0)

    gap = value(set(FACTORS)) - value(set())
    # sequential: switch I, then D, then R, then P (reference group 0 for what is not yet switched)
    seq, done, prev = {}, set(), value(set())
    for f in FACTORS:
        done = done | {f}
        cur = value(done)
        seq[f] = cur - prev
        prev = cur
    # Shapley: average marginal contribution over all orders
    shap = {f: 0.0 for f in FACTORS}
    for order in permutations(FACTORS):
        done, prev = set(), value(set())
        for f in order:
            done = done | {f}
            cur = value(done)
            shap[f] += cur - prev
            prev = cur
    shap = {f: shap[f] / factorial(4) for f in FACTORS}
    return dict(gap=gap, sequential=seq, shapley=shap)


def proposition1_terms(models, m0, v, c):
    """The four terms of Proposition 2 (manuscript): propagate the mean gap e_t with A_0,
    along the path of group 1.  I + D + R + P equals the model gap exactly."""
    A0, B0, a0 = models[0]["A"], models[0]["B"], models[0]["a"]
    A1, B1, a1 = models[1]["A"], models[1]["B"], models[1]["a"]
    m1 = rollout_mean(m0[1], A1, B1, a1, v[1])
    T = len(v[1])
    powers = [np.eye(len(a0))]
    for _ in range(T):
        powers.append(A0 @ powers[-1])
    e0 = np.asarray(m0[1]) - np.asarray(m0[0])
    I = float(c @ powers[T] @ e0)
    D = sum(float(c @ powers[T - 1 - t] @ ((A1 - A0) @ m1[t] + (a1 - a0))) for t in range(T))
    R = sum(float(c @ powers[T - 1 - t] @ ((B1 - B0) * v[1][t])) for t in range(T))
    P = sum(float(c @ powers[T - 1 - t] @ (B0 * (v[1][t] - v[0][t]))) for t in range(T))
    return dict(I=I, D=D, R=R, P=P)


FREE_FACTORS = ("I", "A", "a")


def free_decomposition(models, m0, T, c, c0=0.0):
    """Gap decomposition when no control is observed (real logs): the model is
    z' = A_g z + a_g, and the gap splits into initial state I, persistence A and drift a.
    Returns the model gap, the sequential terms (order I, A, a) and the Shapley terms."""
    def value(switched):
        g = {f: 1 if f in switched else 0 for f in FREE_FACTORS}
        m = np.asarray(m0[g["I"]], float)
        A, a = models[g["A"]]["A"], models[g["a"]]["a"]
        for _ in range(T):
            m = A @ m + a
        return float(c @ m + c0)

    gap = value(set(FREE_FACTORS)) - value(set())
    seq, done, prev = {}, set(), value(set())
    for f in FREE_FACTORS:
        done = done | {f}
        cur = value(done)
        seq[f], prev = cur - prev, cur
    shap = {f: 0.0 for f in FREE_FACTORS}
    for order in permutations(FREE_FACTORS):
        done, prev = set(), value(set())
        for f in order:
            done = done | {f}
            cur = value(done)
            shap[f] += cur - prev
            prev = cur
    shap = {f: shap[f] / factorial(len(FREE_FACTORS)) for f in FREE_FACTORS}
    return dict(gap=gap, sequential=seq, shapley=shap)


def fit_free_group_models(Z, Zn, G):
    """Least squares [A a] for each group (no control), same lifting."""
    out = {}
    for g in (0, 1):
        m = G == g
        X = np.column_stack([Z[m], np.ones(m.sum())])
        C = np.linalg.lstsq(X, Zn[m], rcond=None)[0]
        out[g] = dict(A=C[:-1].T, a=C[-1])
    return out


def fit_free_group_models_iv(Zs, G):
    """Same as fit_free_group_models, but the regressor z_t is instrumented by z_{t-1}.
    Least squares on noisy states shrinks A towards 0 (attenuation), which shortens the
    memory of the initial state; the lagged state is a valid instrument when the noise
    is not correlated over time.  Zs: (N, T+1, n) lifted trajectories."""
    out = {}
    n = Zs.shape[-1]
    for g in (0, 1):
        Zg = Zs[G == g]
        W = np.concatenate([Zg[:, :-2], np.ones(Zg.shape[:1] + (Zg.shape[1] - 2, 1))], -1).reshape(-1, n + 1)
        X = np.concatenate([Zg[:, 1:-1], np.ones(Zg.shape[:1] + (Zg.shape[1] - 2, 1))], -1).reshape(-1, n + 1)
        Y = Zg[:, 2:].reshape(-1, n)
        C = np.linalg.lstsq(W.T @ X, W.T @ Y, rcond=None)[0]   # lifting may repeat a feature (p*p = p)
        out[g] = dict(A=C[:-1].T, a=C[-1])
    return out


def fit_free_group_models_iv_lag(Zs, G, lag=1, reduce=True, tol=1e-8):
    """Instrumental-variable fit of z_{t+1} = A_g z_t + a_g with z_{t-lag} as instrument for z_t.
    lag=1 is fit_free_group_models_iv.  Returns the models and, for each group, first-stage
    diagnostics: the condition number of W'X and the smallest first-stage R^2 over the lifted
    coordinates (how well the instrument predicts each regressor).  Zs: (N, T+1, n)."""
    out, diag = {}, {}
    n_full = Zs.shape[-1]
    U = np.eye(n_full)
    if reduce:                     # the lifting can repeat a feature (p*p = p): work in its column space
        flat = Zs.reshape(-1, n_full)
        _, s, Vt = np.linalg.svd(flat - flat.mean(0), full_matrices=False)
        U = Vt[s > tol * s[0]].T
        Zs = Zs @ U
    n = Zs.shape[-1]
    for g in (0, 1):
        Zg = Zs[G == g]
        L = Zg.shape[1]
        ones = np.ones((Zg.shape[0], L - lag - 1, 1))
        W = np.concatenate([Zg[:, : L - lag - 1], ones], -1).reshape(-1, n + 1)
        X = np.concatenate([Zg[:, lag: L - 1], ones], -1).reshape(-1, n + 1)
        Y = Zg[:, lag + 1:].reshape(-1, n)
        C = np.linalg.lstsq(W.T @ X, W.T @ Y, rcond=None)[0]
        Ar, ar = C[:-1].T, C[-1]
        # back to the full coordinates: z = U r + (part orthogonal to the data, kept fixed)
        out[g] = dict(A=U @ Ar @ U.T + (np.eye(n_full) - U @ U.T), a=U @ ar) if reduce else dict(A=Ar, a=ar)
        Pi = np.linalg.lstsq(W, X[:, :-1], rcond=None)[0]
        resid = X[:, :-1] - W @ Pi
        var = X[:, :-1].var(0)
        keep = var > 1e-12
        r2 = 1 - resid[:, keep].var(0) / var[keep]
        diag[g] = dict(cond_WX=float(np.linalg.cond(W.T @ X)), min_first_stage_r2=float(r2.min()),
                       median_first_stage_r2=float(np.median(r2)), spectral_radius=float(np.abs(np.linalg.eigvals(Ar)).max()), rank=int(n))
    return out, diag
