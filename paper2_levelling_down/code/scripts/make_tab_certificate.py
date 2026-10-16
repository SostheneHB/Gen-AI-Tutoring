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
    lines = []
    for regime in ("standard", "heterogeneous", "hard"):
        res = json.load(open(OUT / f"results_{regime}.json"))
        for i, g in enumerate(("0.5", "1.0", "2.0")):
            runs = [r for r in res[g] if not r.get("skipped")]
            skipped = len(res[g]) - len(runs)
            c = [r["full"]["conformal"] for r in runs]
            l = [r["full"]["ltt"] for r in runs]
            d = [r["full"]["conformal_disjoint"] for r in runs]
            ci_lo = min(x["cover_mean_ci"][0] for x in c)
            ci_hi = max(x["cover_mean_ci"][1] for x in c)
            name = regime if i == 0 else ""
            gtxt = {"0.5": "0.5", "1.0": "1", "2.0": "2"}[g] + (f"\\(^{{{len(runs)}}}\\)" if skipped else "")
            lines.append(" & ".join([
                name, gtxt,
                rng_([x["cover_mean"] for x in c]), f"[{ci_lo:.2f}, {ci_hi:.2f}]", rng_([x["share_sets_below_target"] for x in c]),
                rng_([x["cover_min"] for x in d]),
                rng_([x["intervention_rate"] for x in c]), rng_([x["risk"] for x in c], "{:.3f}"),
                rng_([x["risk_loss"] for x in c], "{:.3f}"), rng_([x["risk_shortfall"] for x in c], "{:.3f}"), rng_([x["success_given_intervention"] for x in c]),
                rng_([x["intervention_rate"] for x in l]), rng_([x["risk"] for x in l], "{:.3f}"),
                rng_([r["naive"]["success"] if r["naive"] else None for r in runs]),
            ]) + "\\\\")
        lines.append("\\addlinespace")
    lines = lines[:-1]
    tex = r"""\begin{table}[t]
	\centering\footnotesize
	\caption{The certificate of Theorem~\ref{thm:conformal} with knowledge as the target (five fitted models for each \(\gamma\), ranges over models; a superscript gives the number of models when some have an efficient reference with no model gap; \(\alpha=0.1\), \(m=100\)). Coverage: share of test cohorts with \(s\leq\hat q\), averaged over 200 calibrations (marginal), with the 95\% Monte Carlo interval from resampling the cohorts of both pools (MC); share of these calibrations whose coverage is below 0.9; smallest coverage over 20 disjoint calibrations. Intervene: share of calibrations that change the plan (the others keep the reference). Risk: probability, over calibrations and test cohorts, that the plan is changed \emph{and} fails, which is the event bounded by the theorem; split into a loss of a group mean and a narrowing below \(\rho\) without loss; overall success is \(1-\)risk. Success \(\mid\) int.: success among the calibrations that change the plan, which the theorem does \emph{not} guarantee at \(1-\alpha\). LTT: Learn then Test (fixed-sequence binomial tests, \(\eta=0.1\)). No margin: success of the plan with \(\delta=0\) applied without calibration.}
	\label{tab:certificate}
	\resizebox{\textwidth}{!}{%
	\begin{tabular}{@{}lrllllllllllll@{}}
		\toprule
		& & \multicolumn{4}{c}{Coverage of the score} & \multicolumn{5}{c}{Conformal} & \multicolumn{2}{c}{Learn then Test} & No margin\\
		\cmidrule(lr){3-6}\cmidrule(lr){7-11}\cmidrule(lr){12-13}\cmidrule(l){14-14}
		Regime & \(\gamma\) & marginal & MC & sets \(<0.9\) & min, disjoint & intervene & risk & loss & shortfall & success \(\mid\) int. & intervene & risk & success\\
		\midrule
""" + "\n".join("\t\t" + x for x in lines) + r"""
		\bottomrule
	\end{tabular}}
\end{table}
"""
    (ROOT / "manuscript" / "tab_certificate.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
