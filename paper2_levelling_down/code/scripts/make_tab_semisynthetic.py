"""Table of the semi-synthetic test (tab_semisynthetic.tex) from outputs/22_semisynthetic/results_knowledge_k*.json.

    python code/scripts/make_tab_semisynthetic.py
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rng_(v, fmt="{:.2f}"):
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else f"{lo}--{hi}"


def main():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "outputs" / "22_semisynthetic" / "results_knowledge_k*.json"))):
        rows += json.load(open(f))["rows"]
    json.dump(dict(rows=rows), open(ROOT / "outputs" / "22_semisynthetic" / "results_knowledge.json", "w"), indent=1)
    lines = []
    for k in sorted({r["kappa"] for r in rows}):
        rs = [r for r in rows if r["kappa"] == k and not r.get("skipped")]
        p = lambda i, key: [r["price"][i][key] for r in rs]
        c = [r["certificate"] for r in rs]
        lines.append(" & ".join([
            f"{k:g}", rng_([abs(r["model_gap"]) for r in rs], "{:.3f}"), rng_([abs(r["true_gap"]) for r in rs], "{:.3f}"),
            f"{sum(all(r['assumptions'].values()) for r in rs)}/{len(rs)}",
            rng_(p(1, "extra_budget"), "{:.1f}"), rng_(p(1, "promised"), "{:.3f}"), rng_(p(1, "true_narrowing"), "{:.3f}"),
            rng_(p(2, "promised"), "{:.3f}"), rng_(p(2, "true_narrowing"), "{:.3f}"),
            rng_([x["naive_success"] for x in c]), rng_([x["intervene"] for x in c]), rng_([x["risk"] for x in c], "{:.3f}"),
            rng_([x["cover_mean"] for x in c]), rng_([x["ltt_risk"] for x in c], "{:.3f}")]) + "\\\\")
    tex = r"""\begin{table}[t]
	\centering\footnotesize
	\caption{The method on a semi-synthetic system calibrated on Eedi (Appendix~\ref{app:semisim}; the strength \(\kappa\) of the effect of help is assumed). Five fitted models for each \(\kappa\), ranges over models; reference: the efficient plan with the smallest gap. Gaps: final knowledge gap of the reference in the model and on the system (20 cohorts). Half and full: the cheapest fair plans for half of and all of what the model can close (\(\min(\rho_{\max},|\delta^{\mathrm{ref}}|)\)); extra budget as a multiple of the reference budget; ``true'' is the narrowing on the system. In every model and plan both group means are kept (smallest change \(-2\times10^{-16}\)) and no student ends with less knowledge. Certificate: target \(\rho\) = half of what the model can close, margins \(0,0.04,\dots,0.60\), \(\alpha=0.1\), \(m=100\), 200 calibrations; no margin: success of the plan with \(\delta=0\); risk: probability that the plan is changed and fails; LTT: Learn then Test.}
	\label{tab:semisynthetic}
	\resizebox{\textwidth}{!}{%
	\begin{tabular}{@{}rlllllllllllll@{}}
		\toprule
		& \multicolumn{3}{c}{Model and system} & \multicolumn{3}{c}{Half of the closable gap} & \multicolumn{2}{c}{Full} & \multicolumn{4}{c}{Certificate} & LTT\\
		\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-13}\cmidrule(l){14-14}
		\(\kappa\) & model gap & true gap & (A1)--(A5) & extra budget & promised & true & promised & true & no margin & intervene & risk & coverage & risk\\
		\midrule
""" + "\n".join("\t\t" + x for x in lines) + r"""
		\bottomrule
	\end{tabular}}
\end{table}
"""
    for d in ("manuscript_TMLR",):
        (ROOT / d / "tab_semisynthetic.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
