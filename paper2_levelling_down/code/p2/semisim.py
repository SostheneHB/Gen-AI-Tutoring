"""A semi-synthetic tutoring simulator calibrated on Eedi (experiment 22).

The learning of each group follows knowledge tracing with forgetting (BKT), with the five parameters fitted on Eedi
for that group (code/scripts/fit_bkt_eedi.py, groups = PremiumPupil 1 / 0).  Two ingredients cannot be identified
from these logs and are ASSUMED (and varied): the effect of help and the spread of the learning rate between students.

State of a student, as in the simulator of the companion paper, x = (k, e, p, d):
  k  expected mastery (the knowledge outcome),       e  moving average of observed correctness,
  p  last observed correctness (0/1),                 d  unused (0).
One step with help u (u in [0, 1] for decisions; the training data use u in [-1, 1]):
  learning hazard   T_i(u) = 1 - (1 - min(T_g r_i, 0.95)) ** exp(kappa * u)        (help multiplies the hazard exponent)
  k'  = clip( (1 - F_g) k + (1 - k) T_i(u) exp(0.3 eta - 0.045), 0, 1 )            (eta ~ N(0, 1): day-to-day variation)
  y   ~ Bernoulli( k (1 - S_g) + (1 - k) G_g ),  drawn from a second normal noise through its distribution function
  e'  = 0.7 e + 0.3 y,  p' = y.
Initial mastery: k_0 = clip(L0_g + 0.1 z, 0, 1), z ~ N(0, 1); e_0 = p_0 = probability of a correct answer at k_0.
The response factor r_i is log-normal with mean 1 (sigma 0.5), as in the heterogeneous regime of the main experiments.
The noise of a cohort is standard normal; common random numbers make the reference and a plan comparable student by
student.  `patch(fa, ...)` replaces the simulator of the companion code (cohorts, step) so that its model fitting runs
unchanged on the semi-synthetic system.
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

ROOT = Path(__file__).resolve().parents[2]
PARAMS = ROOT / "outputs" / "22_semisynthetic" / "bkt_params.json"
SIGMA_R = 0.5


# The parameters fitted on Eedi (also printed in the appendix of the paper), so that the experiments can be rerun
# without the Eedi data; fit_bkt_eedi.py recomputes them from the data.
FITTED = {0: {'L0': 0.509889, 'T': 0.018159, 'F': 0.035002, 'G': 0.356769, 'S': 0.143993}, 1: {'L0': 0.411041, 'T': 0.013829, 'F': 0.050791, 'G': 0.33565, 'S': 0.155353}}


def load_params(path=PARAMS):
    if not Path(path).exists():
        return FITTED
    g = json.load(open(path))["groups"]
    return {int(k): v["params"] for k, v in g.items()}       # group 1 = PremiumPupil (disadvantaged)


def step_factory(par, kappa, r):
    def step(states, controls, groups, noise):
        k, e, p, d = np.moveaxis(np.asarray(states, float), -1, 0)
        eta, z = noise[..., 0], noise[..., 1]
        u = np.asarray(controls, float)
        g = np.asarray(groups).astype(int)
        T = np.where(g == 1, par[1]["T"], par[0]["T"])
        F = np.where(g == 1, par[1]["F"], par[0]["F"])
        G = np.where(g == 1, par[1]["G"], par[0]["G"])
        S = np.where(g == 1, par[1]["S"], par[0]["S"])
        hazard = 1 - (1 - np.minimum(T * r, 0.95)) ** np.exp(kappa * u)
        k1 = np.clip((1 - F) * k + (1 - k) * hazard * np.exp(0.3 * eta - 0.045), 0, 1)
        y = (ndtr(z) < k * (1 - S) + (1 - k) * G).astype(float)
        return np.stack([k1, 0.7 * e + 0.3 * y, y, np.zeros_like(k1)], axis=-1)
    return step


ROLES = {"training": 1, "evaluation": 2}


def responses(fa, seed, role, index, n):
    """Response factors of a cohort: a stream of their own, independent of the cohort's states and noise."""
    g = np.random.default_rng([seed, ROLES[role], index, 7741])
    return g.lognormal(-SIGMA_R ** 2 / 2, SIGMA_R, n)


def cohort_factory(fa, par):
    def make_cohort(seed, students=150, horizon=25, *, role="evaluation", index=0):
        if students < 2 or students % 2:
            raise ValueError("Use an even cohort size.")
        rs = fa.random_stream(seed, role, index)
        groups = np.repeat([0, 1], students // 2)
        L0 = np.where(groups == 1, par[1]["L0"], par[0]["L0"])
        G = np.where(groups == 1, par[1]["G"], par[0]["G"])
        S = np.where(groups == 1, par[1]["S"], par[0]["S"])
        k0 = np.clip(L0 + 0.1 * rs.standard_normal(students), 0, 1)
        pc = k0 * (1 - S) + (1 - k0) * G
        init = np.stack([k0, pc, pc, np.zeros(students)], axis=-1)
        noise = rs.standard_normal((students, horizon, 3))
        return init, noise, groups
    return make_cohort


def patch(fa, par, kappa, seed):
    """Make the companion simulator code run on the semi-synthetic system (training responses fixed by the seed)."""
    fa.make_cohort = cohort_factory(fa, par)
    fa.step = step_factory(par, kappa, responses(fa, seed, "training", 0, 150))
    return fa


def cohort_step(fa, par, kappa, seed, role, index, n):
    """The step of an evaluation cohort, with its own response factors."""
    return step_factory(par, kappa, responses(fa, seed, role, index, n))
