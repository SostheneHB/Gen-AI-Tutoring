"""
Experiments answering the external review (October 2026).

    python reviewer_experiments.py budget        # E1 equal-budget baselines for the Pareto-fair MPC
    python reviewer_experiments.py solver        # E2 certification of the gap-constrained solver (Eq. 39)
    python reviewer_experiments.py structural    # E3 level-1 checks for Eedi, OULAD and the group-aware model
    python reviewer_experiments.py replicates    # E4 the 28 settings with 50 replicates and Holm correction
    python reviewer_experiments.py oulad         # E5 OULAD with held-out presentations and a time split
    python reviewer_experiments.py scenarios     # E6 scenarios not used to design the method
Outputs (aggregates only): out_reviewer/<part>.json
"""
import json
import os
import sys
import time

import numpy as np
from scipy import stats

import pareto_fair as pf
import synthetic_regimes as sr

SIM = os.path.expanduser("~/Documents/Directive_research_copy_code/Final_Folder_on_DR")
DATA = os.path.expanduser("~/Documents/Directive_research_copy_code/Datasets")
OUT = "out_reviewer"
SEEDS10 = list(range(10))
T = 25


def summ(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return dict(mean=float(v.mean()), ci=[float("nan")] * 2, p=float("nan"), n=len(v))
    h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    p = float(stats.ttest_1samp(v, 0).pvalue) if v.std() > 0 else (0.0 if v.mean() != 0 else 1.0)
    return dict(mean=float(v.mean()), ci=[float(v.mean() - h), float(v.mean() + h)], p=p, n=len(v))


def holm(pvals):
    """Holm adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def rollout(fa, model, pol, init, noise, groups):
    x = init.copy()
    S, U = [x.copy()], []
    for t in range(noise.shape[1]):
        z = model.encode(x)
        u = pol(z, groups) if isinstance(pol, sr.CohortMPC) else pol(z)
        U.append(u)
        x = fa.step(x, u, groups, noise[:, t])
        S.append(x.copy())
    S, U = np.stack(S, 1), np.stack(U, 1)
    return dict(gap=float(fa.contrast(S[:, -1, fa.D_IDX], groups)), k=float(S[:, -1, 0].mean()),
                k0=float(S[groups == 0, -1, 0].mean()), k1=float(S[groups == 1, -1, 0].mean()),
                kgap=float(fa.contrast(S[:, -1, 0], groups)),
                u1=float(np.abs(U).mean()), u2=float((U ** 2).mean()))


# ---------------------------------------------------------------------------
# Policies added for the review
# ---------------------------------------------------------------------------
class TopUp(sr.CohortMPC):
    """Reference group-aware MPC, plus a constant extra control kappa for every
    student of the group behind (clipped to the box)."""

    def __init__(self, base, kappa, behind):
        self.base, self.kappa, self.behind = base, kappa, behind

    def __call__(self, latent, groups):
        u = np.array(self.base(latent, groups), float)
        m = groups == self.behind
        u[m] = np.clip(u[m] + self.kappa, -1, 1)
        return u


class CertifiedCohortMPC(sr.CohortMPC):
    """Gap-constrained MPC with the exact feasibility test of Eq. (39) and an
    expanding multiplier bracket.  Records, for every step: certified
    infeasibility, bracket expansions, and the primal residual |gap|."""

    def __init__(self, fa, model, **kw):
        super().__init__(fa, model, 0.0, "constraint", **kw)
        self.stats = dict(steps=0, infeasible=0, expanded=0, unresolved=0, max_res_feasible=0.0,
                          max_res_infeasible_excess=0.0)
        self.vnorm1 = float(np.abs(self.v).sum())

    def __call__(self, latent, groups):
        n1, n0 = (groups == 1).sum(), (groups == 0).sum()
        s = np.where(groups == 1, 1 / n1, -1 / n0)
        base = latent @ self.base_map + self.base_off
        b = float(s @ base)
        reach = float(np.abs(s).sum() * self.vnorm1)        # ||w||_1 of Eq. (39)
        feasible = abs(b) <= reach
        gapf = lambda U: s @ (base + U @ self.v)
        f = lambda lam: gapf(self._solve(latent, lam * s))
        self.stats["steps"] += 1
        lo, hi = -1e4, 1e4
        flo, fhi = f(lo), f(hi)
        while np.sign(flo) == np.sign(fhi) and hi < 1e12:
            lo, hi = lo * 10, hi * 10
            flo, fhi = f(lo), f(hi)
            self.stats["expanded"] += 1
        if np.sign(flo) == np.sign(fhi):
            lam = lo if abs(flo) < abs(fhi) else hi
            if feasible:
                self.stats["unresolved"] += 1
        else:
            for _ in range(80):
                mid = .5 * (lo + hi)
                fm = f(mid)
                if np.sign(fm) == np.sign(flo):
                    lo, flo = mid, fm
                else:
                    hi = mid
            lam = .5 * (lo + hi)
        U = self._solve(latent, lam * s)
        res = abs(float(gapf(U)))
        if feasible:
            self.stats["max_res_feasible"] = max(self.stats["max_res_feasible"], res)
        else:
            self.stats["infeasible"] += 1
            self.stats["max_res_infeasible_excess"] = max(self.stats["max_res_infeasible_excess"], res - (abs(b) - reach))
        return U[:, 0]


def ga_models(fa, seed):
    model = fa.fit_model(seed, 4, 3)
    A, B, a0, aG = pf.fit_group_aware(fa, model, seed)
    return model, (A, B, a0, aG)


def bisect(fun, lo, hi, target, increasing=True, iters=30):
    for _ in range(iters):
        mid = .5 * (lo + hi)
        v = fun(mid)
        if (v < target) == increasing:
            lo = mid
        else:
            hi = mid
    return .5 * (lo + hi)


# ---------------------------------------------------------------------------
# E1  equal-budget baselines
# ---------------------------------------------------------------------------
def budget(fa, gammas=(0.5, 1.0, 2.0), seeds=SEEDS10, c=2.0, beta=8.0):
    out = {}
    for gamma in gammas:
        fa.step = sr.make_step(gamma)
        per = []
        for seed in seeds:
            model, (A, B, a0, aG) = ga_models(fa, seed)
            init, noise, groups = fa.make_cohort(seed, horizon=T)
            mk = lambda **kw: pf.GroupAwareMPC(fa, model, beta, A, B, a0, aG, **kw)
            ref = rollout(fa, model, mk(fair=False), init, noise, groups)
            fair = rollout(fa, model, mk(fair=True, cH=1, first_budget=True, budget_factor=c), init, noise, groups)
            behind = 1 if ref["gap"] < 0 else 0
            run_k = lambda kap: rollout(fa, model, TopUp(mk(fair=False), kap, behind), init, noise, groups)
            k1 = bisect(lambda kap: run_k(kap)["u1"], 0.0, 1.0, fair["u1"])
            k2 = bisect(lambda kap: run_k(kap)["u2"], 0.0, 1.0, fair["u2"])
            run_r = lambda lr: rollout(fa, model, mk(fair=False, rho=float(np.exp(lr))), init, noise, groups)
            lr1 = bisect(lambda lr: run_r(lr)["u1"], np.log(0.02), 0.0, fair["u1"], increasing=False)
            arms = dict(fair=fair, topup_u1=run_k(k1), topup_u2=run_k(k2), more_u1=run_r(lr1))
            per.append(dict(ref=ref, **arms, kappa_u1=k1, kappa_u2=k2, rho_u1=float(np.exp(lr1))))
            print(f"[budget] gamma={gamma} seed={seed}: |gap| ref {abs(ref['gap']):.3f} fair {abs(fair['gap']):.3f} "
                  f"topup(U1) {abs(arms['topup_u1']['gap']):.3f} topup(U2) {abs(arms['topup_u2']['gap']):.3f} "
                  f"more(U1) {abs(arms['more_u1']['gap']):.3f} | U1 fair {fair['u1']:.3f} topup {arms['topup_u1']['u1']:.3f}", flush=True)
        res = {"gamma": gamma, "c": c}
        for arm in ("fair", "topup_u1", "topup_u2", "more_u1"):
            res[arm] = {q: summ([p[arm][q] - p["ref"][q] for p in per]) for q in ("k", "k0", "k1", "kgap")}
            res[arm]["abs_gap"] = summ([abs(p[arm]["gap"]) - abs(p["ref"]["gap"]) for p in per])
            res[arm]["u1_ratio"] = summ([p[arm]["u1"] / p["ref"]["u1"] for p in per])
            res[arm]["u2_ratio"] = summ([p[arm]["u2"] / p["ref"]["u2"] for p in per])
            res[arm]["worst_group"] = summ([min(p[arm]["k0"] - p["ref"]["k0"], p[arm]["k1"] - p["ref"]["k1"]) for p in per])
        for arm in ("topup_u1", "topup_u2", "more_u1"):
            res[f"fair_minus_{arm}"] = {
                "abs_gap": summ([abs(p["fair"]["gap"]) - abs(p[arm]["gap"]) for p in per]),
                "k0": summ([p["fair"]["k0"] - p[arm]["k0"] for p in per]),
                "k1": summ([p["fair"]["k1"] - p[arm]["k1"] for p in per])}
        res["kappa_u1"] = summ([p["kappa_u1"] for p in per])
        res["rho_u1"] = summ([p["rho_u1"] for p in per])
        out[str(gamma)] = res
    fa.step = sr.make_step(1.0)
    return out


# ---------------------------------------------------------------------------
# E2  solver certification
# ---------------------------------------------------------------------------
def configs28():
    base = dict(sr.BASE)
    grids = {"gamma": [0.0, 0.25, 0.5, 1.0, 2.0, -1.0], "rank": [1, 2, 4, 6, 8],
             "depth": [1, 2, 3, 5], "horizon": [5, 10, 25, 50], "beta": [0.0, 1.0, 8.0, 32.0]}
    seen, cfgs = set(), []
    for f, vals in grids.items():
        for v in vals:
            cfg = dict(base, **{f: v})
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key); cfgs.append(cfg)
    for g in (0.0, 0.5, 1.0, 2.0):
        for b in (0.0, 1.0, 8.0, 32.0):
            cfg = dict(base, gamma=g, beta=b)
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key); cfgs.append(cfg)
    assert len(cfgs) == 28, len(cfgs)
    return cfgs


def solver(fa, seeds=SEEDS10):
    out = []
    done = {}
    for cfg in configs28():
        key = (cfg["gamma"], cfg["rank"], cfg["depth"], cfg["horizon"])   # the constraint arm ignores beta
        if key in done:
            out.append(dict(config=cfg, **done[key])); continue
        fa.step = sr.make_step(cfg["gamma"])
        tot = dict(steps=0, infeasible=0, expanded=0, unresolved=0, max_res_feasible=0.0, max_res_infeasible_excess=0.0)
        per = []
        for seed in seeds:
            model = fa.fit_model(seed, cfg["rank"], cfg["depth"])
            init, noise, groups = fa.make_cohort(seed, horizon=cfg["horizon"])
            raw = rollout(fa, model, fa.MPCPolicy(*model.dynamics(False), model.decoder, model.decoder_offset,
                                                  beta=cfg["beta"]), init, noise, groups)
            old = rollout(fa, model, sr.CohortMPC(fa, model, 0.0, "constraint"), init, noise, groups)
            pol = CertifiedCohortMPC(fa, model)
            new = rollout(fa, model, pol, init, noise, groups)
            for k in ("steps", "infeasible", "expanded", "unresolved"):
                tot[k] += pol.stats[k]
            for k in ("max_res_feasible", "max_res_infeasible_excess"):
                tot[k] = max(tot[k], pol.stats[k])
            per.append(dict(raw=raw, old=old, new=new))
        rec = dict(stats=tot,
                   infeasible_share=tot["infeasible"] / tot["steps"],
                   certified_abs_gap_vs_raw=summ([abs(p["new"]["gap"]) - abs(p["raw"]["gap"]) for p in per]),
                   certified_k0_vs_raw=summ([p["new"]["k0"] - p["raw"]["k0"] for p in per]),
                   certified_k1_vs_raw=summ([p["new"]["k1"] - p["raw"]["k1"] for p in per]),
                   max_abs_change_vs_old=float(max(max(abs(p["new"][q] - p["old"][q]) for q in ("gap", "k0", "k1")) for p in per)))
        done[key] = rec
        out.append(dict(config=cfg, **rec))
        print(f"[solver] {cfg}: steps {tot['steps']} infeasible {tot['infeasible']} expanded {tot['expanded']} "
              f"unresolved {tot['unresolved']} max res (feasible) {tot['max_res_feasible']:.1e} "
              f"max change vs reported solver {rec['max_abs_change_vs_old']:.2e}", flush=True)
    fa.step = sr.make_step(1.0)
    return out


# ---------------------------------------------------------------------------
# E3  structural checks for Eedi, OULAD and the group-aware model
# ---------------------------------------------------------------------------
def structural(fa):
    import assistments_pipeline as ap
    import rank_matched_controls as rm
    out = {"eedi": [], "oulad": [], "group_aware": []}

    def rows(df, seed, split, strata):
        rng, Xtr, Gtr, Str, Xte, Gte, Ste = rm.prepare(df, seed, split, strata)
        fit = ap.fit_edmd(Xtr)
        G, _ = ap.metric(fit.G, ap.RIDGE_RATIO)
        c, _ = ap.contrast(fit, Xtr, Gtr)
        P, R, cand = ap.build_projector(fit.K, G, c)
        A_P, a_P = ap.propagator(fit, P)
        Rv = np.eye(len(P)) - P.T
        Gi = np.linalg.inv(G)
        return dict(duality_err=float(np.abs(fit.K - fit.A.T).max()),
                    idempotency_err=float(np.abs(P @ P - P).max()),
                    metric_duality_err=float(np.abs(Gi @ P.T @ G - P).max()),
                    eps_q=float(np.sqrt(max((P @ c) @ G @ (P @ c), 0))),
                    r1=ap.wnorm(P @ fit.K @ R, G), r2=ap.wnorm(R @ fit.K @ P, G),
                    leak_a=float(np.linalg.norm(Rv @ a_P)), candidate_rank=int(cand),
                    numerical_rank_G=int(fit.num_rank))
    df = ap.keep_full_trajectories(ap.load_features(f"{DATA}/Eedi_NeurIPS2020/eedi_features_time.csv"))
    out["eedi"] = [rows(df, 20260914 + 200 + j, "school", "given") for j in range(5)]
    base = ap.keep_full_trajectories(ap.load_features(f"{DATA}/OULAD/oulad_features_weekly.csv"))
    for att in ("imd", "gender", "disability"):
        df = base.copy(); df["S"] = df[f"S_{att}"]; df["G"] = df["S"].astype(int)
        out["oulad"] += [dict(attribute=att, **rows(df, 20260914 + 300 + j, "student", "given")) for j in range(5)]
    # group-aware model: least-squares optimality, and the constraints of the Pareto-fair controller
    for gamma in (0.5, 1.0, 2.0):
        fa.step = sr.make_step(gamma)
        for seed in SEEDS10:
            model = fa.fit_model(seed, 4, 3)
            states, controls, groups = fa.training_data(seed)
            Z = model.encode(states[:, :-1].reshape(-1, 4)); Zn = model.encode(states[:, 1:].reshape(-1, 4))
            Gc = np.repeat(groups, states.shape[1] - 1).astype(float)
            X = np.column_stack([Z, controls.reshape(-1), np.ones(len(Z)), Gc])
            A, B, a0, aG = pf.fit_group_aware(fa, model, seed)
            C = np.column_stack([A, B, a0, aG]).T
            normal = float(np.abs(X.T @ (Zn - X @ C)).max() / np.abs(X.T @ Zn).max())
            init, noise, g = fa.make_cohort(seed, horizon=T)
            pol = pf.GroupAwareMPC(fa, model, 8.0, A, B, a0, aG, True, cH=1, first_budget=True, budget_factor=2.0)
            rollout(fa, model, pol, init, noise, g)
            out["group_aware"].append(dict(gamma=gamma, seed=seed, normal_equation_rel_residual=normal,
                                           max_floor_violation=float(pol.floor_violation),
                                           max_budget_violation=float(pol.budget_violation),
                                           fair_step_share=pol.fair_steps / max(pol.steps, 1)))
    fa.step = sr.make_step(1.0)
    for k, v in out.items():
        keys = [q for q in v[0] if isinstance(v[0][q], float)]
        print(f"[structural] {k}: " + ", ".join(f"{q} max {max(r[q] for r in v):.2e}" for q in keys), flush=True)
    return out


# ---------------------------------------------------------------------------
# E4  the 28 settings with 50 replicates and Holm correction
# ---------------------------------------------------------------------------
def replicates(fa, n_seeds=50):
    seeds = list(range(n_seeds))
    rows = []
    for cfg in configs28():
        fa.step = sr.make_step(cfg["gamma"])
        per = []
        for seed in seeds:
            model = fa.fit_model(seed, cfg["rank"], cfg["depth"])
            init, noise, groups = fa.make_cohort(seed, horizon=cfg["horizon"])
            mk = lambda proj: fa.MPCPolicy(*model.dynamics(proj), model.decoder, model.decoder_offset, beta=cfg["beta"])
            per.append(dict(raw=rollout(fa, model, mk(False), init, noise, groups),
                            proj=rollout(fa, model, mk(True), init, noise, groups),
                            con=rollout(fa, model, CertifiedCohortMPC(fa, model), init, noise, groups)))
        d_proj = [abs(p["proj"]["gap"]) - abs(p["raw"]["gap"]) for p in per]
        d_con = [abs(p["con"]["gap"]) - abs(p["raw"]["gap"]) for p in per]
        worst_con = [min(p["con"]["k0"] - p["raw"]["k0"], p["con"]["k1"] - p["raw"]["k1"]) for p in per]
        lower_grp = 0 if np.mean([p["con"]["k0"] - p["raw"]["k0"] for p in per]) < np.mean([p["con"]["k1"] - p["raw"]["k1"] for p in per]) else 1
        d_low = [p["con"][f"k{lower_grp}"] - p["raw"][f"k{lower_grp}"] for p in per]
        rows.append(dict(config=cfg, proj_abs_gap=summ(d_proj), con_abs_gap=summ(d_con),
                         con_worst_group=summ(worst_con), con_lower_group=lower_grp, con_lower_group_k=summ(d_low),
                         n_widen_proj=int(np.sum(np.array(d_proj) > 0))))
        r = rows[-1]
        print(f"[replicates] {cfg}: proj d|gap| {r['proj_abs_gap']['mean']:+.4f} p={r['proj_abs_gap']['p']:.1e} | "
              f"con d|gap| {r['con_abs_gap']['mean']:+.4f} lower group k {r['con_lower_group_k']['mean']:+.4f}", flush=True)
    for key in ("proj_abs_gap", "con_abs_gap", "con_lower_group_k"):
        adj = holm([r[key]["p"] for r in rows])
        for r, a in zip(rows, adj):
            r[key]["p_holm"] = float(a)
    ok = lambda r: r["config"]["gamma"] != 0.0
    summary = dict(
        n_settings=len(rows),
        proj_widens_holm=int(sum(r["proj_abs_gap"]["p_holm"] < .05 and r["proj_abs_gap"]["mean"] > 0 for r in rows)),
        proj_narrows_holm=int(sum(r["proj_abs_gap"]["p_holm"] < .05 and r["proj_abs_gap"]["mean"] < 0 for r in rows)),
        con_narrows_holm=int(sum(r["con_abs_gap"]["p_holm"] < .05 and r["con_abs_gap"]["mean"] < 0 for r in rows if ok(r))),
        con_levels_down_holm=int(sum(r["con_lower_group_k"]["p_holm"] < .05 and r["con_lower_group_k"]["mean"] < 0 for r in rows if ok(r))),
        n_with_group_effect=int(sum(ok(r) for r in rows)))
    fa.step = sr.make_step(1.0)
    print("[replicates] summary", summary, flush=True)
    return dict(n_seeds=n_seeds, summary=summary, rows=rows)


# ---------------------------------------------------------------------------
# E5  OULAD: held-out presentations, and a time split
# ---------------------------------------------------------------------------
def oulad(seed=20260914):
    import assistments_pipeline as ap
    base = ap.keep_full_trajectories(ap.load_features(f"{DATA}/OULAD/oulad_features_weekly.csv"))
    out = {}
    orig_split = ap.split_schools

    def time_split(df, rng):
        users = df.drop_duplicates("user_id").set_index("user_id")["school_id"]
        tr_all = users.index[users.str.contains("2013")].values
        te = users.index[users.str.contains("2014")].values
        tr_all = rng.permutation(tr_all)
        n = int(0.8 * len(tr_all))
        return tr_all[:n], tr_all[n:], te
    for att in ("imd", "disability", "gender"):
        df = base.copy(); df["S"] = df[f"S_{att}"]; df["G"] = df["S"].astype(int)
        res = {}
        ap.split_schools = orig_split
        runs = [ap.run(df, f"{att}_presentation_disjoint_{j}", seed + 400 + j, "school", "given",
                       ridge_sweep=False, outdir=os.path.join(OUT, "oulad_runs", att)) for j in range(5)]
        res["presentation_disjoint"] = ap.summarise(runs)
        ap.split_schools = time_split
        run = ap.run(df, f"{att}_time_2013_to_2014", seed + 500, "school", "given", ridge_sweep=False,
                     outdir=os.path.join(OUT, "oulad_runs", att))
        res["time_split"] = dict(observed=run["replay"]["terminal_gap_observed"],
                                 observed_ci=run["replay"]["terminal_gap_observed_school_ci"],
                                 raw=run["replay"]["models"]["edmd_raw"]["terminal_gap_pred"],
                                 groupcal=run["replay"]["models"]["edmd_groupcal"]["terminal_gap_pred"],
                                 proj=run["replay"]["models"]["edmd_proj"]["terminal_gap_pred"],
                                 mse_h1_raw=run["replay"]["models"]["edmd_raw"]["mse"]["1"]["k_t"],
                                 mse_h1_proj=run["replay"]["models"]["edmd_proj"]["mse"]["1"]["k_t"],
                                 proj_gap_error_ci=run["replay"]["models"]["edmd_proj"]["gap_error_school_ci"],
                                 n_train=run["n_train"], n_test=run["n_test"])
        ap.split_schools = orig_split
        out[att] = res
        s = res["presentation_disjoint"]
        print(f"[oulad] {att}: presentation-disjoint observed {s['observed_gap']['mean']:+.4f} raw {s['replay_gap_raw']['mean']:+.4f} "
              f"proj {s['replay_gap_proj']['mean']:+.4f} groupcal {s['replay_gap_groupcal']['mean']:+.4f} "
              f"mse h1 ratio {s['mse_k_h1_edmd_proj']['mean'] / s['mse_k_h1_edmd_raw']['mean']:.1f} | time split: "
              f"obs {res['time_split']['observed']:+.4f} proj {res['time_split']['proj']:+.4f} "
              f"ratio {res['time_split']['mse_h1_proj'] / res['time_split']['mse_h1_raw']:.1f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# E6  scenarios not used to design the method
# ---------------------------------------------------------------------------
from scipy.special import expit


def make_step_scenario(gamma=1.0, ctrl_g1=1.0, decay=0.85, noise_scale=1.0, saturating=False, start_gap=0.0):
    def step(states, controls, groups, noise):
        k, e, p, d = np.moveaxis(np.asarray(states), -1, 0)
        nk, ne, nd = np.moveaxis(np.asarray(noise) * noise_scale, -1, 0)
        eff = np.where(groups == 1, ctrl_g1, 1.0) * controls
        keff = eff * (1 - k) * 2 if saturating else eff
        k1 = np.clip(decay * k + .15 * expit(3 * e - 1.5) + .20 * keff - .06 * gamma * groups + nk, 0, 1)
        e1 = np.clip(.80 * e + .10 * np.tanh(p) + .15 * eff + ne, 0, 1)
        return np.stack([k1, e1, expit(6 * (k1 - .5)), .90 * d - .05 * gamma * groups + .05 * eff + nd], axis=-1)
    return step


SCENARIOS = {
    "weaker_response_group1": dict(ctrl_g1=0.5),
    "slower_learning_more_noise": dict(decay=0.92, noise_scale=2.0),
    "diminishing_returns": dict(saturating=True),
    "gap_at_start_no_drift": dict(gamma=0.0, start_gap=0.10),
}


def scenarios(fa, seeds=SEEDS10, beta=8.0):
    orig_cohort = fa.make_cohort
    out = {}
    for name, kw in SCENARIOS.items():
        fa.step = make_step_scenario(**kw)
        sg = kw.get("start_gap", 0.0)

        def cohort(seed, students=150, horizon=25, *, role="evaluation", index=0):
            init, noise, groups = orig_cohort(seed, students, horizon, role=role, index=index)
            init = init.copy()
            init[groups == 1, 0] = np.clip(init[groups == 1, 0] - sg, 0, 1)
            return init, noise, groups
        fa.make_cohort = cohort
        per = []
        for seed in seeds:
            model, (A, B, a0, aG) = ga_models(fa, seed)
            init, noise, groups = fa.make_cohort(seed, horizon=T)
            mk = lambda proj: fa.MPCPolicy(*model.dynamics(proj), model.decoder, model.decoder_offset, beta=beta)
            ga = lambda **k: pf.GroupAwareMPC(fa, model, beta, A, B, a0, aG, **k)
            r = dict(raw=rollout(fa, model, mk(False), init, noise, groups),
                     proj=rollout(fa, model, mk(True), init, noise, groups),
                     con=rollout(fa, model, CertifiedCohortMPC(fa, model), init, noise, groups),
                     ga_ref=rollout(fa, model, ga(fair=False), init, noise, groups))
            fair_pol = ga(fair=True, cH=1, first_budget=True, budget_factor=2.0)
            r["fair"] = rollout(fa, model, fair_pol, init, noise, groups)
            behind = 1 if r["ga_ref"]["gap"] < 0 else 0
            run_k = lambda kap: rollout(fa, model, TopUp(ga(fair=False), kap, behind), init, noise, groups)
            r["topup_u1"] = run_k(bisect(lambda kap: run_k(kap)["u1"], 0.0, 1.0, r["fair"]["u1"]))
            r["fair_floor_violation"] = float(fair_pol.floor_violation)
            per.append(r)
        res = {}
        for arm, ref in (("proj", "raw"), ("con", "raw"), ("fair", "ga_ref"), ("topup_u1", "ga_ref")):
            res[f"{arm}_vs_{ref}"] = dict(
                abs_gap=summ([abs(p[arm]["gap"]) - abs(p[ref]["gap"]) for p in per]),
                k0=summ([p[arm]["k0"] - p[ref]["k0"] for p in per]), k1=summ([p[arm]["k1"] - p[ref]["k1"] for p in per]),
                u1_ratio=summ([p[arm]["u1"] / p[ref]["u1"] for p in per]))
        res["fair_minus_topup"] = dict(abs_gap=summ([abs(p["fair"]["gap"]) - abs(p["topup_u1"]["gap"]) for p in per]),
                                       k0=summ([p["fair"]["k0"] - p["topup_u1"]["k0"] for p in per]),
                                       k1=summ([p["fair"]["k1"] - p["topup_u1"]["k1"] for p in per]))
        res["raw_abs_gap"] = summ([abs(p["raw"]["gap"]) for p in per])
        res["max_fair_floor_violation"] = float(max(p["fair_floor_violation"] for p in per))
        out[name] = dict(settings=kw, **res)
        f = lambda e: f"{e['mean']:+.3f} [{e['ci'][0]:+.3f},{e['ci'][1]:+.3f}]"
        print(f"[scenarios] {name}: raw |gap| {res['raw_abs_gap']['mean']:.3f} | proj d|gap| {f(res['proj_vs_raw']['abs_gap'])} "
              f"| con d|gap| {f(res['con_vs_raw']['abs_gap'])} k0 {f(res['con_vs_raw']['k0'])} k1 {f(res['con_vs_raw']['k1'])} "
              f"| fair d|gap| {f(res['fair_vs_ga_ref']['abs_gap'])} k0 {f(res['fair_vs_ga_ref']['k0'])} k1 {f(res['fair_vs_ga_ref']['k1'])} "
              f"| topup d|gap| {f(res['topup_u1_vs_ga_ref']['abs_gap'])}", flush=True)
    fa.make_cohort = orig_cohort
    fa.step = sr.make_step(1.0)
    return out


if __name__ == "__main__":
    part = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    fa = sr.load_sim(SIM)
    t0 = time.time()
    res = {"budget": lambda: budget(fa), "solver": lambda: solver(fa), "structural": lambda: structural(fa),
           "replicates": lambda: replicates(fa), "oulad": lambda: oulad(), "scenarios": lambda: scenarios(fa)}[part]()
    json.dump(res, open(os.path.join(OUT, f"{part}.json"), "w"), indent=1, default=float)
    print(f"[{part}] done in {time.time() - t0:.0f}s", flush=True)
