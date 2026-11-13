"""Table 2 of the manuscript (tab_certificate.tex) from outputs/12_certificate_v3/results_<regime>.json.

    python code/scripts/make_tab_certificate.py

Ranges over the fitted models of each regime and gamma (models whose reference has no model gap are left out and
counted).  Risk = P(intervene and fail) over calibrations and test cohorts, split into a loss of a group mean and a
narrowing below rho; MC: 95% interval of the mean coverage from resampling the cohorts of both pools.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "12_certificate_v3"


def rng_(v, fmt="{:.2f}"):
    v = [x for x in v if x is not None]
    if not v:
        return "--"
    lo, hi = min(v), max(v)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}--{fmt.format(hi)}"


def main():
    cov, dec = [], []
    for regime in ("standard", "heterogeneous", "hard"):
        res = json.load(open(OUT / f"results_{regime}.json"))
        for i, g in enumerate(("0.5", "1.0", "2.0")):
            runs = [r for r in res[g] if not r.get("skipped")]
            skipped = len(res[g]) - len(runs)
            c = [r["full"]["conformal"] for r in runs]
            l = [r["full"]["ltt"] for r in runs]
            d = [r["full"]["conformal_disjoint"] for r in runs]
            name = regime if i == 0 else ""
            gtxt = {"0.5": "0.5", "1.0": "1", "2.0": "2"}[g] + (f"\\(^{{{len(runs)}}}\\)" if skipped else "")
            cov.append(" & ".join([name, gtxt, rng_([x["cover_mean"] for x in c]),
                                   f"[{min(x['cover_mean_ci'][0] for x in c):.2f}, {max(x['cover_mean_ci'][1] for x in c):.2f}]",
                                   rng_([x["share_sets_below_target"] for x in c]), rng_([x["cover_min"] for x in d]),
                                   rng_([r["naive"]["success"] if r["naive"] else None for r in runs])]) + "\\\\")
            dec.append(" & ".join([name, gtxt,
                                   rng_([x["intervention_rate"] for x in c]), rng_([x["risk"] for x in c], "{:.3f}"),
                                   rng_([x["risk_loss"] for x in c], "{:.3f}"), rng_([x["success_given_intervention"] for x in c]),
                                   rng_([x["extra_given_intervention"] for x in c]),
                                   rng_([x["intervention_rate"] for x in l]), rng_([x["risk"] for x in l], "{:.3f}"),
                                   rng_([x["extra_given_intervention"] for x in l])]) + "\\\\")
        cov.append("\\addlinespace"); dec.append("\\addlinespace")
    cov, dec = cov[:-1], dec[:-1]
    tex = r"""\begin{table}[t]
	\centering\small
	\caption{The certificate of Theorem~\ref{thm:conformal} with knowledge as the target: coverage of the score (five fitted models for each \(\gamma\), ranges over models; a superscript gives the number of models when some have an efficient reference with no model gap; \(\alpha=0.1\), \(m=100\)). Marginal: share of test cohorts with \(s\leq\hat q\), averaged over 200 calibrations; MC: 95\% Monte Carlo interval from resampling the cohorts of both pools; sets \(<0.9\): share of calibrations whose coverage is below 0.9; min, disjoint: smallest coverage over 20 disjoint calibrations; no margin: success of the plan with \(\delta=0\) applied without calibration.}
	\label{tab:certificate}
	\begin{tabular}{@{}lrlllll@{}}
		\toprule
		Regime & \(\gamma\) & marginal & MC & sets \(<0.9\) & min, disjoint & no margin\\
		\midrule
""" + "\n".join("\t\t" + x for x in cov) + r"""
		\bottomrule
	\end{tabular}
\end{table}

\begin{table}[t]
	\centering\small
	\caption{The decisions of the certificate and of Learn then Test (LTT; fixed-sequence binomial tests, \(\eta=0.1\)), same settings as Table~\ref{tab:certificate}. Intervene: share of calibrations that change the plan (the others keep the reference). Risk: probability, over calibrations and test cohorts, that the plan is changed \emph{and} fails, the event bounded by the theorem; loss: the part of it where a group mean goes down. Success \(\mid\) int.: success among the calibrations that change the plan, which the theorem does \emph{not} guarantee at \(1-\alpha\). Extra budget: mean extra budget of the plan used, as a share of the reference budget, when the plan is changed.}
	\label{tab:certificate-decisions}
	\footnotesize\setlength{\tabcolsep}{3.5pt}
	\begin{tabular}{@{}lrllllllll@{}}
		\toprule
		& & \multicolumn{5}{c}{Certificate} & \multicolumn{3}{c}{Learn then Test}\\
		\cmidrule(lr){3-7}\cmidrule(l){8-10}
		Regime & \(\gamma\) & intervene & risk & loss & success \(\mid\) int. & extra budget & intervene & risk & extra budget\\
		\midrule
""" + "\n".join("\t\t" + x for x in dec) + r"""
		\bottomrule
	\end{tabular}
\end{table}
"""
    for d in ("manuscript_TMLR", "manuscript"):            # the second folder exists only in the working project
        if d == "manuscript_TMLR" or (ROOT / d).is_dir():
            (ROOT / d).mkdir(parents=True, exist_ok=True)
            (ROOT / d / "tab_certificate.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
