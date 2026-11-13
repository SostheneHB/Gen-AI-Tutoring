
"""
Generate the semi-synthetic experiment table.

Input:
    outputs/22_semisynthetic/results_knowledge.json

Output:
    manuscript_TMLR/tab_semisynthetic.tex

Run:
    python code/scripts/make_tab_semisynthetic.py

The input JSON contains results for:
    kappa = 0.25, 0.5, 1.0
    Five random seeds per kappa.
"""

import json
from pathlib import Path


# ============================================================
# 1. Project directory
# ============================================================

ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. Format numerical ranges
# ============================================================

def rng_(values, fmt="{:.2f}"):
    """Format the minimum and maximum of numerical values."""
    if not values:
        raise ValueError("Cannot compute a range from empty values.")

    lo = fmt.format(min(values))
    hi = fmt.format(max(values))

    return lo if lo == hi else f"{lo}--{hi}"


# ============================================================
# 3. Main function
# ============================================================

def main():

    # --------------------------------------------------------
    # A. Load the combined experimental results
    # --------------------------------------------------------

    results_dir = ROOT / "outputs" / "22_semisynthetic"
    input_file = results_dir / "results_knowledge.json"

    if not input_file.is_file():
        raise FileNotFoundError(
            f"Missing experimental results:\n{input_file}\n\n"
            "Run the semi-synthetic experiments first:\n"
            "P2_OUTCOME=knowledge python "
            "code/scripts/run_semisynthetic.py "
            "--kappas 0.25 0.5 1.0"
        )

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data["rows"]

    if not rows:
        raise ValueError(
            "The semi-synthetic results contain no rows."
        )

    expected_kappas = {0.25, 0.5, 1.0}
    actual_kappas = {float(r["kappa"]) for r in rows}

    if actual_kappas != expected_kappas:
        raise ValueError(
            "Unexpected kappa values.\n"
            f"Expected: {sorted(expected_kappas)}\n"
            f"Found: {sorted(actual_kappas)}"
        )

    print(f"Loaded {len(rows)} semi-synthetic results.")

    # --------------------------------------------------------
    # B. Generate numerical rows for the LaTeX table
    # --------------------------------------------------------

    price_lines, cert_lines = [], []

    for k in sorted(actual_kappas):

        rs = [
            r for r in rows
            if float(r["kappa"]) == k
            and not r.get("skipped")
        ]

        if not rs:
            raise ValueError(
                f"No valid results for kappa={k}."
            )

        def price_values(index, key):
            return [
                r["price"][index][key]
                for r in rs
            ]

        certificates = [
            r["certificate"]
            for r in rs
        ]

        assumption_count = sum(
            all(r["assumptions"].values())
            for r in rs
        )

        values = [
            f"{k:g}",

            rng_(
                [abs(r["model_gap"]) for r in rs],
                "{:.3f}"
            ),

            rng_(
                [abs(r["true_gap"]) for r in rs],
                "{:.3f}"
            ),

            f"{assumption_count}/{len(rs)}",

            rng_(
                price_values(1, "extra_budget"),
                "{:.1f}"
            ),

            rng_(
                price_values(1, "promised"),
                "{:.3f}"
            ),

            rng_(
                price_values(1, "true_narrowing"),
                "{:.3f}"
            ),

            rng_(
                price_values(2, "promised"),
                "{:.3f}"
            ),

            rng_(
                price_values(2, "true_narrowing"),
                "{:.3f}"
            ),

            rng_([
                x["naive_success"]
                for x in certificates
            ]),

            rng_([
                x["intervene"]
                for x in certificates
            ]),

            rng_(
                [x["risk"] for x in certificates],
                "{:.3f}"
            ),

            rng_([
                x["cover_mean"]
                for x in certificates
            ]),

            rng_(
                [x["ltt_risk"] for x in certificates],
                "{:.3f}"
            ),
        ]

        price_lines.append(" & ".join(values[:9]) + r" \\")
        cert_lines.append(" & ".join([values[0]] + values[9:]) + r" \\")

    # --------------------------------------------------------
    # C. Generate LaTeX table
    # --------------------------------------------------------

    tex = r"""\begin{table}[t]
\centering\small
\caption{The method on a semi-synthetic system calibrated on Eedi (Appendix~\ref{app:semisim}; the strength \(\kappa\) of the effect of help is assumed): the model and the price. Five fitted models for each \(\kappa\), ranges over models; reference: the efficient plan with the smallest gap. Gaps: final knowledge gap of the reference in the model and on the system (20 cohorts). Half and full: the cheapest fair plans for half of and all of what the model can close (\(\min(\rho_{\max},|\delta^{\mathrm{ref}}|)\)); extra budget as a multiple of the reference budget \(B_{\mathrm{ref}}\), on top of it (an extra budget of \(1.8B_{\mathrm{ref}}\) is a total of \(2.8B_{\mathrm{ref}}\)); the budget is a normalized quantity of help; ``true'' is the narrowing on the system. In every model and plan both group means are kept (smallest change \(-2\times10^{-16}\)) and no student ends with less knowledge.}
\label{tab:semisynthetic}
\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}rllllllll@{}}
\toprule
& \multicolumn{3}{c}{Model and system} & \multicolumn{3}{c}{Half of the closable gap} & \multicolumn{2}{c}{Full}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}
\(\kappa\) & model gap & true gap & (A1)--(A5) & extra budget & promised & true & promised & true\\
\midrule
""" + "\n".join("\t\t" + line for line in price_lines) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[t]
\centering\small
\caption{The certificate on the semi-synthetic system of Table~\ref{tab:semisynthetic}: target \(\rho\) = half of what the model can close, margins \(0,0.04,\dots,0.60\), \(\alpha=0.1\), \(m=100\), 200 calibrations. No margin: success of the plan with \(\delta=0\); intervene: share of calibrations that change the plan; risk: probability that the plan is changed and fails; coverage: mean coverage of the score; LTT: risk of Learn then Test.}
\label{tab:semisynthetic-cert}
\begin{tabular}{@{}rlllll@{}}
\toprule
\(\kappa\) & no margin & intervene & risk & coverage & LTT risk\\
\midrule
""" + "\n".join("\t\t" + line for line in cert_lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""

    # --------------------------------------------------------
    # D. Save the LaTeX table
    # --------------------------------------------------------

    output_file = (
        ROOT
        / "manuscript_TMLR"
        / "tab_semisynthetic.tex"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file.write_text(
        tex,
        encoding="utf-8"
    )

    print(f"LaTeX table saved to: {output_file}")
    print("\nGenerated LaTeX table:\n")
    print(tex)


# ============================================================
# 4. Entry point
# ============================================================

if __name__ == "__main__":
    main()
