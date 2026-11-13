"""Table of experiment 24 (tab_mixed.tex) from outputs/24_mixed_family/results_knowledge.json.

    python code/scripts/make_tab_mixed.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"max-score": r"Theorem~\ref{thm:conformal} (maximum over the family)", "per-plan": r"per plan, union bound (Remark~\ref{rem:perplan})",
         "LTT-FS": "Learn then Test, fixed sequence", "LTT-Bonf": "Learn then Test, Bonferroni"}


def rng_(v, fmt="{:.2f}"):
    v = [0.0 if (x is not None and abs(x) < 1e-9) else x for x in v if x is not None]
    if not v:
        return "--"
    lo, hi = fmt.format(min(v)), fmt.format(max(v))
    return lo if lo == hi else f"{lo}--{hi}"


def main():
    R = json.load(open(ROOT / "outputs" / "24_mixed_family" / "results_knowledge.json"))["results"]
    lines = []
    for k, name in NAMES.items():
        v = [r[k] for r in R]
        lines.append(" & ".join([name, rng_([x["intervene"] for x in v]), rng_([x["risk"] for x in v], "{:.3f}"),
                                 rng_([x["chose_move"] for x in v]), rng_([x["safe_rejected"] for x in v], "{:.1f}"),
                                 rng_([x["extra_over_cheapest_safe"] for x in v], "{:.3f}")]) + r"\\")
    safe = [sum(p["truly_safe"] for p in r["plans"]) for r in R]
    tex = r"""\begin{table}[t]
	\centering\small
	\caption{A family with safe and harmful plans (uniform reference, heterogeneous responses; three fitted models for each \(\gamma\), ranges over the nine models; \(\alpha=\eta=0.1\), \(m=100\), 200 calibrations). Each family has four plans that only add help to the group behind (harmless: they never lower a group mean; successful, that is, also reaching the target in at least 90\% of the test cohorts, an estimated share, in """ + f"{min(safe)} to {max(safe)}" + r""" of them per model) and four cheaper plans that move teaching between steps (they save 35 to 71\% of the reference budget, and lower the knowledge of the group ahead in every test cohort). The model predicts that all eight are safe. Each procedure uses the cheapest plan it accepts. Harmful: share of interventions that use a harmful plan; successful plans rejected: mean number of successful plans that the procedure does not accept; extra cost: extra budget of the plan used minus that of the cheapest successful plan (share of the reference budget). Protocol: Appendix~\ref{app:setup}.}
	\label{tab:mixed}
	\footnotesize
	\begin{tabular}{@{}lccccc@{}}
		\toprule
		Procedure & intervene & risk & harmful & successful plans rejected & extra cost\\
		\midrule
""" + "\n".join("\t\t" + x for x in lines) + r"""
		\bottomrule
	\end{tabular}
\end{table}
"""
    (ROOT / "manuscript_TMLR").mkdir(parents=True, exist_ok=True)
    (ROOT / "manuscript_TMLR" / "tab_mixed.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
