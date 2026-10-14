"""
Tables of the revision of 13 October 2026 (review 6), from the aggregate outputs:
  tab_controlled.tex        controlled study of the price and its estimate     (out_reviewer/price_reliability_synthetic.json)
  tab_controlled_full.tex   the same, every cell, both sample sizes and learners (appendix)
  tab_next_item.tex         next-item prediction with the selected DKT           (out_rank_matched/next_item_tuned_*.json)
  tab_next_item_rel.tex     reliability of the announced price on DKT, per split (appendix)
  tab_next_item_grid.tex    the validation grid and the selected configurations   (appendix)
  tab_next_item_long.tex    Eedi, longer histories                               (out_rank_matched/next_item_tuned_eedi_long*.json)
  tab_oracle.tex            controllers against the true-model reference         (out_reviewer/oracle_reference.json)

    python make_tables_review6.py --dest ../Revised_manuscript [--only controlled,next_item,oracle]
"""
import argparse
import glob
import json
import os

import numpy as np

CELL_NAMES = {"useless": "Useless", "partial": "Partly aligned", "aligned": "Aligned"}


def fx(v, d=3, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return f"\\({s}\\)" if sign else s


# --------------------------------------------------------------------------- controlled study
def controlled(dest):
    D = json.load(open("out_reviewer/price_reliability_synthetic.json"))["cells"]
    order = ([f"{al}_a{a}" for al in ("useless", "partial", "aligned") for a in (0.2, 0.7, 1.5)]
             + ["shift_utility_lost", "shift_utility_gained"])

    def label(c):
        if c.startswith("shift_utility_lost"):
            return "Shift: utility lost", "0.7"
        if c.startswith("shift_utility_gained"):
            return "Shift: utility gained", "0.7"
        al, a = c.split("_a")
        return CELL_NAMES[al], a
    main = [r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
            r" & & & \multicolumn{2}{c}{True price} & \multicolumn{3}{c}{Least squares: announced price} & \multicolumn{3}{c}{Share interpretable}\\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
            r"Group signal & \(a\) & \(R^2_{\grp}\) & train & test & \(n=2{,}000\) & \(n=20{,}000\) & coverage & LS, \(2{,}000\) & LS, \(20{,}000\) & GB, \(20{,}000\)\\", r"\midrule"]
    for c in order:
        o2, o, g = D[f"{c}|2000|ols"], D[f"{c}|20000|ols"], D[f"{c}|20000|gbm"]
        name, a = label(c)
        main.append(f"{name} & {a} & {o['true_r2_train']:.3f} & {o['true_price_train']:.3f} & {o['true_price_test']:.3f} & "
                    f"{o2['announced_median']:.3f} & {o['announced_median']:.3f} & {(format(o['coverage_train_truth'], '.2f') if o['true_price_train'] > 1e-9 else '--')} & "
                    f"{o2['share_interpretable']:.2f} & {o['share_interpretable']:.2f} & {g['share_interpretable']:.2f}\\\\")
        if c in ("useless_a1.5", "partial_a1.5", "aligned_a1.5"):
            main.append(r"\addlinespace")
    main += [r"\bottomrule", r"\end{tabular}"]
    full = [r"\begin{tabular}{@{}lrlrrrrrrrrr@{}}", r"\toprule",
            r"Cell & \(n\) & Learner & True & Announced (q10--q90) & Cover. & CV & \(\widehat\tau\) & Measured & Frontier & Interp. & R1/R2/R3/R4\\",
            r"\midrule"]
    for c in order:
        for n in (2000, 20000):
            for l in ("ols", "gbm"):
                o = D[f"{c}|{n}|{l}"]
                name, a = label(c)
                ck = "/".join(f"{o['checks_pass'][k]:.2f}" for k in ("R1", "R2", "R3", "R4"))
                full.append(f"{name}, \\(a={a}\\) & {n} & {'LS' if l == 'ols' else 'GB'} & {o['true_price_train']:.3f} & "
                            f"{o['announced_median']:.3f} ({o['announced_q10_q90'][0]:.3f}--{o['announced_q10_q90'][1]:.3f}) & "
                            f"{o['coverage_train_truth']:.2f} & {min(o['mean_cv'], 9.99):.2f} & {fx(o['tau_median'], 2, True)} & "
                            f"{o['measured_mean']:.3f} & {o['frontier_mean']:.3f} & {o['share_interpretable']:.2f} & {ck}\\\\")
        full.append(r"\addlinespace")
    full = full[:-1] + [r"\bottomrule", r"\end{tabular}"]
    w(dest, "tab_controlled.tex", main)
    w(dest, "tab_controlled_full.tex", full)


# --------------------------------------------------------------------------- next item
ARMS = [("edmd_raw", "EDMD, original"), ("edmd_projected", "EDMD, projected"), ("dkt", "DKT, selected on validation"),
        ("leace_hidden", "DKT, LEACE on the hidden state"), ("leace_by_skill", "DKT, LEACE within each next skill"),
        ("parity_training", "DKT, mean-parity fine-tuning"), ("output_removal", "DKT, output correction (Prop.~\\ref{prop:built}), three-stage")]


def load_runs(ds):
    runs = {}
    for f in sorted(glob.glob(f"out_rank_matched/next_item_tuned_{ds}*.json")):
        if "_long" in f:
            continue
        for k, r in enumerate(json.load(open(f))["splits"]):
            runs[r.get("split", k)] = r
    return [runs[k] for k in sorted(runs)]


def next_item(dest):
    R = {ds: load_runs(ds) for ds in ("assistments", "eedi")}
    V3 = {ds: load_v3(ds) for ds in ("assistments", "eedi")}
    for ds in R:
        if V3[ds]:
            for r, r3 in zip(R[ds], V3[ds]):
                r["scores"]["output_removal"] = r3["scores"]["output_v3"]
                r["paired_vs_dkt"]["output_removal"] = r3["paired_vs_dkt"]["output_v3"]
    head = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
            r" & \multicolumn{4}{c}{ASSISTments (school)} & \multicolumn{4}{c}{Eedi (PremiumPupil)}\\",
            r"\cmidrule(lr){2-5}\cmidrule(l){6-9}",
            r"Model & AUC & \(\Delta\)AUC & \(\Delta\)Brier & \(\widehat\tau\) & AUC & \(\Delta\)AUC & \(\Delta\)Brier & \(\widehat\tau\)\\", r"\midrule"]
    lines = []
    for arm, name in ARMS:
        cells = [name]
        for ds in ("assistments", "eedi"):
            runs = R[ds]
            if not runs:
                cells += ["--"] * 4
                continue
            auc = np.mean([r["scores"][arm]["auc"] for r in runs])
            if arm == "dkt":
                cells += [f"{auc:.3f}", "ref.", "ref.", "1"]
                continue
            p = [r["paired_vs_dkt"][arm] for r in runs]
            da = np.mean([q["auc"]["diff"] for q in p])
            below = sum(q["auc"]["ci"][1] < 0 for q in p)
            above = sum(q["auc"]["ci"][0] > 0 for q in p)
            tag = f"{below}/{len(p)}\\(\\downarrow\\)" if below else (f"{above}/{len(p)}\\(\\uparrow\\)" if above else f"0/{len(p)}")
            db = np.mean([q["brier"]["diff"] for q in p])
            tau = np.median([q["tau"]["diff"] for q in p])
            cells += [f"{auc:.3f}", f"\\({da:+.3f}\\) {tag}", f"\\({db:+.4f}\\)", f"\\({tau:+.2f}\\)"]
        lines.append(" & ".join(cells) + "\\\\")
        if arm in ("edmd_projected",):
            lines.append(r"\addlinespace")
    w(dest, "tab_next_item.tex", head + lines + [r"\bottomrule", r"\end{tabular}"])
    # reliability per split
    rel = [r"\begin{tabular}{@{}llrrrrrrrrl@{}}", r"\toprule",
           r"Data & Split & Price [refit 95\%] & \(R^2_{\grp}\) & \(\widehat\tau\) & Measured & Frontier & Meas.\,\(-\)\,front. [95\%] & Slope & ECE & Interpretable (failed)\\",
           r"\midrule"]
    for ds, nm in (("assistments", "ASSISTments"), ("eedi", "Eedi")):
        for k, r in enumerate(R[ds]):
            x = r["reliability"]
            checks = dict(x["checks"])
            checks["tau_ok"] = abs(x["tau"]) <= 0.2
            ok = all(checks.values())
            failed = ",".join(n for n, key in (("R1", "contrast_excludes_zero"), ("R2", "q_slope_ok"), ("R2", "q_ece_ok"), ("R3", "tau_ok"), ("R4", "price_stable"))
                              if not checks[key])
            failed = ",".join(dict.fromkeys(failed.split(","))) if failed else ""
            rel.append(f"{nm if k == 0 else ''} & {r.get('split', k) + 1} & {x['announced']['price']:.4f} [{x['price_ci'][0]:.4f}, {x['price_ci'][1]:.4f}] & "
                       f"{x['announced']['r2']:.3f} & \\({x['tau']:+.2f}\\) & {x['measured']:.4f} & {x['frontier']:.4f} & "
                       f"[\\({x['measured_minus_frontier_ci'][0]:+.4f}\\), \\({x['measured_minus_frontier_ci'][1]:+.4f}\\)] & "
                       f"{x['q_slope']:.2f} & {x['q_ece']:.3f} & {'yes' if ok else 'no (' + failed + ')'}\\\\")
        rel.append(r"\addlinespace")
    w(dest, "tab_next_item_rel.tex", rel[:-1] + [r"\bottomrule", r"\end{tabular}"])
    grid = [r"\begin{tabular}{@{}llrrrrl@{}}", r"\toprule",
            r"Data & Split & Hidden & Learning rate & Dropout & Epochs & Validation log-loss (best; range over grid)\\", r"\midrule"]
    for ds, nm in (("assistments", "ASSISTments"), ("eedi", "Eedi")):
        for k, r in enumerate(R[ds]):
            s, g = r["selected"], r["grid"]
            best = min(x["val_logloss"] for x in g)
            ep = next(x["epochs"] for x in g if x["val_logloss"] == best)
            grid.append(f"{nm if k == 0 else ''} & {r.get('split', k) + 1} & {s['hidden']} & {s['lr']:g} & {s['dropout']:g} & {ep} & "
                        f"{best:.4f} ({min(x['val_logloss'] for x in g):.4f}--{max(x['val_logloss'] for x in g):.4f})\\\\")
    w(dest, "tab_next_item_grid.tex", grid + [r"\bottomrule", r"\end{tabular}"])
    summary = {}
    for ds in R:
        runs = R[ds]
        if not runs:
            continue
        summary[ds] = dict(n=len(runs), dkt_auc=float(np.mean([r["scores"]["dkt"]["auc"] for r in runs])),
                           edmd_auc=float(np.mean([r["scores"]["edmd_raw"]["auc"] for r in runs])),
                           dkt_minus_edmd=[float(r["scores"]["dkt"]["auc"] - r["scores"]["edmd_raw"]["auc"]) for r in runs],
                           obs_gap=float(np.mean([r["scores"]["dkt"]["obs_gap"] for r in runs])),
                           pred_gap_dkt=float(np.mean([r["scores"]["dkt"]["pred_gap"] for r in runs])),
                           parity_lambda=[r["parity"]["lam"] for r in runs], parity_val_tau=[r["parity"]["val_tau"] for r in runs],
                           selected=[r["selected"] for r in runs],
                           price=[r["reliability"]["announced"]["price"] for r in runs], measured=[r["reliability"]["measured"] for r in runs],
                           frontier=[r["reliability"]["frontier"] for r in runs], tau=[r["reliability"]["tau"] for r in runs],
                           interpretable=[bool(r["reliability"]["interpretable"] and abs(r["reliability"]["tau"]) <= 0.2) for r in runs])
    print(json.dumps(summary, indent=1))


def next_item_long(dest):
    files = sorted(glob.glob("out_rank_matched/next_item_tuned_eedi_long*.json"))
    runs = [r for f in files for r in json.load(open(f))["splits"]]
    if not runs:
        return
    out = [r"\begin{tabular}{@{}lrrrrrl@{}}", r"\toprule",
           r"History & DKT AUC & Removal \(\Delta\)AUC & Removal \(\Delta\)Brier & Price & Measured & Interpretable\\", r"\midrule"]
    for L in ("25", "50", "100"):
        xs = [r["results"][L] for r in runs]
        out.append(f"{L} items & {np.mean([x['scores']['dkt']['auc'] for x in xs]):.3f} & "
                   f"\\({np.mean([x['paired_vs_dkt']['output_removal']['auc']['diff'] for x in xs]):+.3f}\\) & "
                   f"\\({np.mean([x['paired_vs_dkt']['output_removal']['brier']['diff'] for x in xs]):+.4f}\\) & "
                   f"{np.mean([x['reliability']['announced']['price'] for x in xs]):.4f} & {np.mean([x['reliability']['measured'] for x in xs]):.4f} & "
                   f"{sum(bool(x['reliability']['interpretable'] and abs(x['reliability']['tau']) <= 0.2) for x in xs)}/{len(xs)}\\\\")
    w(dest, "tab_next_item_long.tex", out + [r"\bottomrule", r"\end{tabular}"])
    print("long:", [[(L, round(r["results"][L]["scores"]["dkt"]["auc"], 3)) for L in ("25", "50", "100")] for r in runs])


# --------------------------------------------------------------------------- oracle
ORACLE_ARMS = [("oracle", "Known equations, with the group"), ("oracle_blind", "Known equations, group replaced by its mean"),
               ("group_aware", "Learned, group-aware"), ("edmd_raw", "Learned, original"),
               ("edmd_projected", "Learned, group direction projected out"), ("edmd_random", "Learned, random rank-4 removal")]


def oracle(dest):
    S = json.load(open("out_reviewer/oracle_reference.json"))["summary"]
    out = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
           r" & \multicolumn{5}{c}{Approximately equal budget (\(\mathcal U_1\) of the reference)} & \multicolumn{2}{c}{\(\rho=1\)}\\",
           r"\cmidrule(lr){2-6}\cmidrule(l){7-8}",
           r"Controller & \(\overline k_T\) & \(\overline k_{T,0}\) & \(\overline k_{T,1}\) & \(|\delta^k_T|\) & \(\rho\) & \(\overline k_T\) & \(\mathcal U_1\)\\", r"\midrule"]
    for obj, title in (("knowledge", "Objective: mean knowledge"), ("paper", r"Objective of the paper (\(\beta_{\mathrm{ind}}=8\))")):
        key = f"{obj}|1.0"
        row = S[key]
        out.append(f"\\multicolumn{{8}}{{@{{}}l}}{{{title}, \\(\\gamma=1\\); budget \\(\\mathcal U_1={row['budget_u1']['mean']:.3f}\\)}}\\\\")
        for arm, name in ORACLE_ARMS:
            m, r1 = row["matched"][arm], row["rho1"][arm]
            out.append(f"\\quad {name} & {m['k']['mean']:.3f} & {m['k0']['mean']:.3f} & {m['k1']['mean']:.3f} & {m['abs_kgap']['mean']:.3f} & "
                       f"{row['rho'][arm]['mean']:.2f} & {r1['k']['mean']:.3f} & {r1['u1']['mean']:.3f}\\\\")
        out.append(r"\addlinespace")
    w(dest, "tab_oracle.tex", out[:-1] + [r"\bottomrule", r"\end{tabular}"])
    names = [("model_error_with_group", "Model error, with group"), ("model_error_without_group", "Model error, without group"),
             ("erasure", "Erasure"), ("compression", "Random compression"), ("group_information", "Group information")]
    dif = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
           r"Objective & \(\gamma\) & Mode & " + " & ".join(n for _, n in names) + r"\\", r"\midrule"]
    for obj in ("knowledge", "paper"):
        for g in ("0.5", "1.0", "2.0"):
            for mode, lab in (("matched", "equal \\(\\mathcal U_1\\)"), ("rho1", "\\(\\rho=1\\)")):
                d = S[f"{obj}|{g}"][mode]["differences"]
                cell = lambda n: f"\\({d[n]['k']['mean']:+.3f}\\) [\\({d[n]['k']['ci'][0]:+.3f}\\), \\({d[n]['k']['ci'][1]:+.3f}\\)]"
                dif.append(f"{obj if (g == '0.5' and mode == 'matched') else ''} & {g if mode == 'matched' else ''} & {lab} & " + " & ".join(cell(n) for n, _ in names) + r"\\")
        dif.append(r"\addlinespace")
    w(dest, "tab_oracle_diff.tex", dif[:-1] + [r"\bottomrule", r"\end{tabular}"])


def w(dest, name, lines):
    open(os.path.join(dest, name), "w").write("\n".join(lines) + "\n")
    print("wrote", name)


def main():
    a_ = argparse.ArgumentParser()
    a_.add_argument("--dest", default="../Revised_manuscript")
    a_.add_argument("--only", default="controlled,next_item,next_item_long,oracle")
    a = a_.parse_args()
    for k in a.only.split(","):
        globals()[k](a.dest)



# --------------------------------------------------------------------------- second review round (v2)
def controlled_v2(dest):
    D = json.load(open("out_reviewer/price_reliability_synthetic_v2.json"))["cells"]
    order = ([f"{al}_a{a}" for al in ("useless", "partial", "aligned") for a in (0.2, 0.7, 1.5)]
             + ["shift_utility_lost", "shift_utility_gained", "calibrated_misspecified", "calibrated_misspecified_correct_model"])

    def label(c):
        if c.startswith("shift_utility_lost"):
            return "Shift: utility lost", "0.7"
        if c.startswith("shift_utility_gained"):
            return "Shift: utility gained", "0.7"
        if c == "calibrated_misspecified":
            return "Calibrated, misspecified $q$", "--"
        if c == "calibrated_misspecified_correct_model":
            return "Same, correct $q$", "--"
        al, a = c.split("_a")
        return CELL_NAMES[al], a
    pct = lambda v: "--" if v is None else f"{v:.2f}"
    main = [r"\begin{tabular}{@{}llrrrrrrrrrr@{}}", r"\toprule",
            r" & & & \multicolumn{2}{c}{True cost} & \multicolumn{2}{c}{Announced} & \multicolumn{3}{c}{Share of replicates} & \multicolumn{2}{c}{Among accepted}\\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(l){11-12}",
            r"Group signal & \(a\) & \(R^2_{\grp}\) & min. & built & v1 & v2 & acc., \(2{,}000\) & acc., \(20{,}000\) & conf., \(20{,}000\) & coverage & rel. error\\",
            r"\midrule"]
    for c in order:
        o2, o, g = D[f"{c}|2000|ols"], D[f"{c}|20000|ols"], D[f"{c}|20000|gbm"]
        name, a = label(c)
        main.append(f"{name} & {a} & {o['r2']:.3f} & {o['truth_min']:.3f} & {o['truth_built']:.3f} & {o['v1']['median']:.3f} & {o['v2']['median']:.3f} & "
                    f"{o2['v2']['share_accepted']:.2f} & {o['v2']['share_accepted']:.2f} & {o['v2'].get('share_confirmed', float('nan')):.2f} & "
                    f"{pct(o['v2']['coverage_built_accepted'])} & {pct(o['v2']['rel_error_min_accepted'] if o['truth_min'] == o['truth_built'] else None)}\\\\")
        if c in ("useless_a1.5", "partial_a1.5", "aligned_a1.5", "shift_utility_gained"):
            main.append(r"\addlinespace")
    w(dest, "tab_controlled.tex", main + [r"\bottomrule", r"\end{tabular}"])
    full = [r"\begin{tabular}{@{}lrlrrrrrrrrrr@{}}", r"\toprule",
            r"Cell & \(n\) & Learner & min & built & v1 (acc.) & v2 (acc.) & v2 cov.\ min (acc.) & v2 cov.\ built (acc.) & v2 rel.\ err.\ acc./rej. & D1/D2/D3 & \(\widehat\tau\) & cross\\",
            r"\midrule"]
    for c in order:
        for n in (2000, 20000):
            for l in ("ols", "gbm"):
                o = D[f"{c}|{n}|{l}"]
                name, a = label(c)
                v = o["v2"]
                ck = "/".join(f"{v['checks_pass'][k]:.2f}" for k in ("D1", "D2", "D3"))
                full.append(f"{name}{', ' + chr(36) + 'a=' + a + chr(36) if a != '--' else ''} & {n} & {'LS' if l == 'ols' else 'GB'} & {o['truth_min']:.3f} & {o['truth_built']:.3f} & "
                            f"{o['v1']['median']:.3f} ({o['v1']['share_accepted']:.2f}) & {v['median']:.3f} ({v['share_accepted']:.2f}) & "
                            f"{pct(v['coverage_min'])} ({pct(v['coverage_min_accepted'])}) & {pct(v['coverage_built'])} ({pct(v['coverage_built_accepted'])}) & "
                            f"{pct(v['rel_error_min_accepted'])}/{pct(v['rel_error_min_rejected'])} & {ck} & \\({v['tau_median']:+.2f}\\) & \\({v['cross_mean']:+.3f}\\)\\\\")
        full.append(r"\addlinespace")
    w(dest, "tab_controlled_full.tex", full[:-1] + [r"\bottomrule", r"\end{tabular}"])


def load_v2(ds, long=False):
    runs = {}
    for f in sorted(glob.glob(f"out_rank_matched/next_item_v2_{ds}{'_long' if long else ''}*.json")):
        if not long and "_long" in f:
            continue
        for r in json.load(open(f))["splits"]:
            runs[r["split"]] = r
    return [runs[k] for k in sorted(runs)]


def next_item_v2(dest):
    R = {ds: load_v2(ds) for ds in ("assistments", "eedi")}
    t = [r"\begin{tabular}{@{}llrrrrrrrrrl@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{First version (training units)} & \multicolumn{2}{c}{Platt recalibrated} & \multicolumn{5}{c}{Three-stage audit (Proposition~\ref{prop:built})}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(l){8-12}",
         r"Data & Split & Price & \(\widehat\tau\) & slope & \(\widehat\tau\) & slope & Price [95\%] & \(\widehat\tau\) & Measured (\(E d^2\), cross) & In interval & Accepted (failed)\\",
         r"\midrule"]
    for ds, nm in (("assistments", "ASSISTments"), ("eedi", "Eedi")):
        for k, r in enumerate(R[ds]):
            a, b, v = r["v1"], r["v1_platt"], r["v2"]
            fail = ",".join(n for n, key in (("D1", "D1_gap_detectable"), ("D2", "D2_score_informative"), ("D3", "D3_price_stable")) if not v["checks"][key])
            m = v["measured"]
            t.append(f"{nm if k == 0 else ''} & {r['split'] + 1} & {a['price']:.4f} & \\({a['tau']:+.2f}\\) & {a['q_slope']:.2f} & \\({b['tau']:+.2f}\\) & {b['q_slope']:.2f} & "
                     f"{v['price']:.4f} [{v['price_ci'][0]:.4f}, {v['price_ci'][1]:.4f}] & \\({v['tau']:+.2f}\\) & {m['total']:.4f} ({m['d2']:.4f}, \\({m['cross']:+.4f}\\)) & "
                     f"{'yes' if v['measured_in_price_ci'] else 'no'} & {'yes' if v['accepted'] else 'no (' + fail + ')'}\\\\")
        t.append(r"\addlinespace")
    w(dest, "tab_next_item_rel.tex", t[:-1] + [r"\bottomrule", r"\end{tabular}"])
    names = [("leace_hidden", "LEACE on the hidden state", "frozen"), ("leace_by_skill", "LEACE within each next skill", "frozen"),
             ("parity_training", "Mean-parity fine-tuning", "retrained")]
    c = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
         r"Method & Network & \(\widehat\tau\) & Cost & \(E d^2\) & cross & Output correction, same \(\widehat\tau\)\\", r"\midrule"]
    runs = R["eedi"]
    for key, nm, kind in names:
        x = [r["comparison"][key] for r in runs]
        c.append(f"{nm} & {kind} & \\({np.mean([q['tau'] for q in x]):+.2f}\\) & {np.mean([q['cost']['total'] for q in x]):.4f} & "
                 f"{np.mean([q['cost']['d2'] for q in x]):.4f} & \\({np.mean([q['cost']['cross'] for q in x]):+.4f}\\) & {np.mean([q['output_same_tau']['total'] for q in x]):.4f}\\\\")
    v = [r["v2"] for r in runs]
    c.append(f"Output correction (Prop.~\\ref{{prop:built}}) & frozen & \\({np.mean([q['tau'] for q in v]):+.2f}\\) & {np.mean([q['measured']['total'] for q in v]):.4f} & "
             f"{np.mean([q['measured']['d2'] for q in v]):.4f} & \\({np.mean([q['measured']['cross'] for q in v]):+.4f}\\) & --\\\\")
    w(dest, "tab_next_item_compare.tex", c + [r"\bottomrule", r"\end{tabular}"])
    for ds in R:
        rs = R[ds]
        if rs:
            print(ds, "v2 accepted", sum(r["v2"]["accepted"] for r in rs), "/", len(rs), "| price", [round(r["v2"]["price"], 4) for r in rs],
                  "| measured", [round(r["v2"]["measured"]["total"], 4) for r in rs], "| tau", [round(r["v2"]["tau"], 2) for r in rs],
                  "| in CI", [r["v2"]["measured_in_price_ci"] for r in rs], "| platt slope", [round(r["v1_platt"]["q_slope"], 2) for r in rs],
                  "| platt tau", [round(r["v1_platt"]["tau"], 2) for r in rs])


def next_item_v2_long(dest):
    runs = load_v2("eedi", long=True)
    if not runs:
        return
    out = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
           r"History & Price (three-stage) & \(\widehat\tau\) & Measured & Measured in interval & Accepted\\", r"\midrule"]
    for L in ("25", "50", "100"):
        v = [r["results"][L]["v2"] for r in runs]
        out.append(f"{L} items & {np.mean([q['price'] for q in v]):.4f} & \\({np.mean([q['tau'] for q in v]):+.2f}\\) & {np.mean([q['measured']['total'] for q in v]):.4f} & "
                   f"{sum(q['measured_in_price_ci'] for q in v)}/{len(v)} & {sum(q['accepted'] for q in v)}/{len(v)}\\\\")
    w(dest, "tab_next_item_long_v2.tex", out + [r"\bottomrule", r"\end{tabular}"])



# --------------------------------------------------------------------------- third review round (v3)
def load_v3(ds, long=False):
    runs = {}
    for f in sorted(glob.glob(f"out_rank_matched/next_item_v3_{ds}{'_long' if long else ''}*.json")):
        if not long and "_long" in f:
            continue
        for r in json.load(open(f))["splits"]:
            runs[r["split"]] = r
    return [runs[k] for k in sorted(runs)]


def iv(c, d=4):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def next_item_v3(dest):
    R = {ds: load_v3(ds) for ds in ("assistments", "eedi")}
    R2 = {ds: {r["split"]: r for r in load_v2(ds)} for ds in R}
    # summary of the three versions, split by split
    t = [r"\begin{tabular}{@{}llrrrrrrrl@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{First version (training units)} & \multicolumn{2}{c}{Platt recalibrated} & \multicolumn{3}{c}{Three-stage audit}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(l){8-10}",
         r"Data & Split & Price & \(\widehat\tau\) & slope & \(\widehat\tau\) & slope & Total announced & \(\widehat\tau\) & Accepted / transfers (failed)\\", r"\midrule"]
    for ds, nm in (("assistments", "ASSISTments"), ("eedi", "Eedi")):
        for k, r in enumerate(R[ds]):
            o = R2[ds][r["split"]]
            s = r["skill"]
            fail = ",".join(n for n in ("D1", "D2", "D3") if not s["checks"][n])
            if s["accepted"] and not s["confirmed"]:
                fail = ",".join(n for n in ("T1", "T2") if not s["test_checks"][n])
            acc = ("yes / " + ("yes" if s["confirmed"] else "no")) if s["accepted"] else "no / --"
            t.append(f"{nm if k == 0 else ''} & {r['split'] + 1} & {o['v1']['price']:.4f} & \\({o['v1']['tau']:+.2f}\\) & {o['v1']['q_slope']:.2f} & "
                     f"\\({o['v1_platt']['tau']:+.2f}\\) & {o['v1_platt']['q_slope']:.2f} & {s['announced']['total']:.4f} & \\({s['tau']:+.2f}\\) & {acc}{' (' + fail + ')' if fail else ''}\\\\")
        t.append(r"\addlinespace")
    w(dest, "tab_next_item_rel.tex", t[:-1] + [r"\bottomrule", r"\end{tabular}"])
    # the three intervals, one table per data set; point estimates, then intervals on a second line
    for ds, fname in (("assistments", "tab_next_item_audit_assist.tex"), ("eedi", "tab_next_item_audit_eedi.tex")):
        a = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r" & & \multicolumn{2}{c}{Announced (training + validation)} & \multicolumn{2}{c}{Measured (test)} & \multicolumn{2}{c}{Measured \(-\) announced}\\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
             r"Split & & Quadratic \(\widehat\pi_s\) & Total & \(\E d^2\) & Total & Quadratic & Total\\", r"\midrule"]
        for k, r in enumerate(R[ds]):
            s = r["skill"]
            dq = s["measured"]["d2"] - s["announced"]["quad"]
            dt = s["measured"]["total"] - s["announced"]["total"]
            a.append(f"{r['split'] + 1} & estimate & {s['announced']['quad']:.4f} & {s['announced']['total']:.4f} & {s['measured']['d2']:.4f} & "
                     f"{s['measured']['total']:.4f} & \\({dq:+.4f}\\) & \\({dt:+.4f}\\)\\\\")
            a.append(f" & 95\\% & {iv(s['quad_ci'])} & {iv(s['total_ci'])} & {iv(s['measured_quad_ci'])} & {iv(s['measured_total_ci'])} & "
                     f"{iv(s['diff_quad_ci'])} & {iv(s['diff_total_ci'])}\\\\")
            a.append(r"\addlinespace")
        w(dest, fname, a[:-1] + [r"\bottomrule", r"\end{tabular}"])
    # comparison at matched gap (Eedi)
    runs = R["eedi"]
    names = [("leace_hidden", "LEACE on the hidden state", "frozen"), ("leace_by_skill", "LEACE within each next skill", "frozen"),
             ("parity_training", "Mean-parity fine-tuning", "retrained")]
    c = [r"\begin{tabular}{@{}llrrrrrrr@{}}", r"\toprule",
         r" & & \multicolumn{3}{c}{Method} & \multicolumn{2}{c}{Output corr., \(\widehat\tau\) matched on test} & \multicolumn{2}{c}{Output corr., intensity set on validation}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
         r"Method & Network & \(\widehat\tau\) & Cost & cross & Cost & Method \(-\) output: splits \(>0\) & \(\widehat\tau\) & Cost\\", r"\midrule"]
    for key, nm, kind in names:
        x = [r["comparison"][key] for r in runs]
        pos = sum(q["diff_ci"][0] > 0 for q in x)
        neg = sum(q["diff_ci"][1] < 0 for q in x)
        c.append(f"{nm} & {kind} & \\({np.mean([q['tau_test'] for q in x]):+.2f}\\) & {np.mean([q['cost']['total'] for q in x]):.4f} & "
                 f"\\({np.mean([q['cost']['cross'] for q in x]):+.4f}\\) & {np.mean([q['output_same_tau']['total'] for q in x]):.4f} & "
                 f"{pos}/{len(x)}{' (' + str(neg) + ' \\(<0\\))' if neg else ''} & \\({np.mean([q['output_val_tau']['tau_test'] for q in x]):+.2f}\\) & {np.mean([q['output_val_tau']['total'] for q in x]):.4f}\\\\")
    v = [r["skill"] for r in runs]
    c.append(f"Output correction (Prop.~\\ref{{prop:built}}) & frozen & \\({np.mean([q['tau'] for q in v]):+.2f}\\) & {np.mean([q['measured']['total'] for q in v]):.4f} & "
             f"\\({np.mean([q['measured']['cross'] for q in v]):+.4f}\\) & -- & -- & -- & --\\\\")
    w(dest, "tab_next_item_compare.tex", c + [r"\bottomrule", r"\end{tabular}"])
    for ds in R:
        rs = R[ds]
        if not rs:
            continue
        S = [r["skill"] for r in rs]
        print(ds, "acc", sum(s["accepted"] for s in S), "conf", sum(s["confirmed"] for s in S), "| quad", [round(s["announced"]["quad"], 4) for s in S],
              "| total", [round(s["announced"]["total"], 4) for s in S], "| meas", [round(s["measured"]["total"], 4) for s in S],
              "| d2", [round(s["measured"]["d2"], 4) for s in S], "| tau", [round(s["tau"], 2) for s in S],
              "| total in CI", [s["total_in_ci"] for s in S], "quad in CI", [s["quad_in_ci"] for s in S],
              "| diff total CI has 0", [s["diff_total_ci"][0] <= 0 <= s["diff_total_ci"][1] for s in S],
              "| hidden-only total", [round(r["hidden_only"]["announced"]["total"], 4) for r in rs], "meas", [round(r["hidden_only"]["measured"]["total"], 4) for r in rs],
              "| clip", [round(r["clip_share"], 4) for r in rs])
        print("   output auc", [round(r["scores"]["output_v3"]["auc"], 3) for r in rs], "dAUC", [round(r["paired_vs_dkt"]["output_v3"]["auc"]["diff"], 3) for r in rs],
              "dAUC CI", [[round(x, 3) for x in r["paired_vs_dkt"]["output_v3"]["auc"]["ci"]] for r in rs])


def next_item_v3_long(dest):
    runs = load_v3("eedi", long=True)
    if not runs:
        return
    out = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
           r"History & Announced total & \(\widehat\tau\) & Measured total & In announced interval & Accepted & Gap reduction transfers\\", r"\midrule"]
    for L in ("25", "50", "100"):
        v = [r["results"][L]["skill"] for r in runs]
        out.append(f"{L} items & {np.mean([q['announced']['total'] for q in v]):.4f} & \\({np.mean([q['tau'] for q in v]):+.2f}\\) & {np.mean([q['measured']['total'] for q in v]):.4f} & "
                   f"{sum(q['total_in_ci'] for q in v)}/{len(v)} & {sum(q['accepted'] for q in v)}/{len(v)} & {sum(q['confirmed'] for q in v)}/{len(v)}\\\\")
    w(dest, "tab_next_item_long_v2.tex", out + [r"\bottomrule", r"\end{tabular}"])


def controlled_v3(dest):
    D = json.load(open("out_reviewer/price_reliability_synthetic_v3.json"))["cells"]
    D1 = json.load(open("out_reviewer/price_reliability_synthetic_v2.json"))["cells"]
    order = ([f"{al}_a{a}" for al in ("useless", "partial", "aligned") for a in (0.2, 0.7, 1.5)]
             + ["shift_utility_lost", "shift_utility_gained", "calibrated_misspecified", "calibrated_misspecified_correct_model"])

    def label(c):
        if c.startswith("shift_utility_lost"):
            return "Shift: utility lost", "0.7"
        if c.startswith("shift_utility_gained"):
            return "Shift: utility gained", "0.7"
        if c == "calibrated_misspecified":
            return "Calibrated, misspecified \\(s\\)", "--"
        if c == "calibrated_misspecified_correct_model":
            return "Same, correct \\(s\\)", "--"
        al, a = c.split("_a")
        return CELL_NAMES[al], a
    pct = lambda v: "--" if v is None else f"{v:.2f}"
    main = [r"\begin{tabular}{@{}llrrrrrrrrrrr@{}}", r"\toprule",
            r" & & & \multicolumn{2}{c}{Population cost} & \multicolumn{2}{c}{Median announced} & \multicolumn{3}{c}{Share of replicates} & \multicolumn{3}{c}{Coverage among accepted}\\",
            r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}",
            r"Group signal & \(a\) & \(R^2_{\grp}\) & min. & built & v1 & quad. & acc., \(2{,}000\) & acc., \(20{,}000\) & transf., \(20{,}000\) & realized quad. & realized total & built\\",
            r"\midrule"]
    for c in order:
        o2, o, o1 = D[f"{c}|2000|ols"], D[f"{c}|20000|ols"], D1[f"{c}|20000|ols"]
        name, a = label(c)
        main.append(f"{name} & {a} & {o1['r2']:.3f} & {o['truth_min']:.3f} & {o['truth_built']:.3f} & {o1['v1']['median']:.3f} & {o['ann_quad_median']:.3f} & "
                    f"{o2['share_accepted']:.2f} & {o['share_accepted']:.2f} & {o['share_confirmed']:.2f} & "
                    f"{pct(o['cov_full_quad_realised_val_accepted'])} & {pct(o['cov_full_total_realised_val_accepted'])} & {pct(o['cov_full_quad_built_accepted'])}\\\\")
        if c in ("useless_a1.5", "partial_a1.5", "aligned_a1.5", "shift_utility_gained"):
            main.append(r"\addlinespace")
    w(dest, "tab_controlled.tex", main + [r"\bottomrule", r"\end{tabular}"])
    for l, fname in (("ols", "tab_controlled_full_ls.tex"), ("gbm", "tab_controlled_full_gb.tex")):
        full = [r"\begin{tabular}{@{}lrrrrrrrrrrr@{}}", r"\toprule",
                r" & & \multicolumn{2}{c}{Announced (median)} & \multicolumn{2}{c}{Realized (median)} & & & \multicolumn{3}{c}{Coverage of realized total} & \\",
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){9-11}",
                r"Cell & \(n\) & quad. & total & quad. & total & acc. & transf. & audit & audit, acc. & cond. & cov.\ built, acc.\\",
                r"\midrule"]
        for c in order:
            for n in (2000, 20000):
                o = D[f"{c}|{n}|{l}"]
                name, a = label(c)
                full.append(f"{name}{', ' + chr(92) + '(a=' + a + chr(92) + ')' if a != '--' else ''} & {n} & {o['ann_quad_median']:.3f} & {o['ann_total_median']:.3f} & "
                            f"{o['realised_val_quad_median']:.3f} & {o['realised_val_total_median']:.3f} & {o['share_accepted']:.2f} & {o['share_confirmed']:.2f} & "
                            f"{o['cov_full_total_realised_val']:.2f} & {pct(o['cov_full_total_realised_val_accepted'])} & {o['cov_cond_total_realised_val']:.2f} & "
                            f"{pct(o['cov_full_quad_built_accepted'])}\\\\")
            full.append(r"\addlinespace")
        w(dest, fname, full[:-1] + [r"\bottomrule", r"\end{tabular}"])
    try:
        S = json.load(open("out_reviewer/price_reliability_synthetic_v3_stability.json"))["cells"]
    except FileNotFoundError:
        return
    st = [r"\begin{tabular}{@{}llrrrrrrrr@{}}", r"\toprule",
          r" & & \multicolumn{4}{c}{Coverage of realized total} & \multicolumn{4}{c}{Largest end-point change vs.\ \(B=400\) (share of width)}\\",
          r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
          r"Cell & Learner & \(B=40\) & 60 & 200 & 400 & 40 & 60 & 200 & 400\\", r"\midrule"]
    for key, o in S.items():
        c, n, l = key.split("|")
        name, a = label(c)
        s = o["stability"]
        st.append(f"{name}{', ' + chr(92) + '(a=' + a + chr(92) + ')' if a != '--' else ''} & {'LS' if l == 'ols' else 'GB'} & "
                  + " & ".join(f"{s[f'{b}_total_cov_realised_val']:.2f}" for b in (40, 60, 200, 400)) + " & "
                  + " & ".join(f"{s[f'{b}_total_endpoint_rel_dev_median']:.2f}" for b in (40, 60, 200, 400)) + "\\\\")
    w(dest, "tab_controlled_stability.tex", st + [r"\bottomrule", r"\end{tabular}"])


if __name__ == "__main__":
    main()
