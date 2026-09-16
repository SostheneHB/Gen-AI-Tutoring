"""Reproduce the matched ridge sweep and restore its portable figure/table sources."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, ".")
import fa_edmd_mpc_v3_corrected as edmd


def run():
    records = []
    for seed in range(10):
        initial, noise, groups = edmd.make_cohort(seed)
        reference = edmd.fit_model(seed)
        raw, _, _ = edmd.rollout(reference, initial, noise, groups,
                                edmd.make_policy(reference, False), False)
        raw_contrast = float(edmd.contrast(raw[:, -1, edmd.D_IDX], groups))
        for exponent in range(-6, 0):
            ratio = 10.0 ** exponent
            model = edmd.fit_model(seed, ridge_relative=ratio)
            np.testing.assert_array_equal(model.propagation, reference.propagation)
            np.testing.assert_array_equal(model.decoder, reference.decoder)
            projected, _, _ = edmd.rollout(model, initial, noise, groups,
                                          edmd.make_policy(model, True), True)
            projected_contrast = float(edmd.contrast(projected[:, -1, edmd.D_IDX], groups))
            operator = model.propagation.T
            compressed = model.projector @ operator @ model.projector
            operator_norm = edmd.weighted_norm(operator, model.metric, 2)
            compressed_norm = edmd.weighted_norm(compressed, model.metric, 2)
            records.append({
                "seed": seed, "ridge_relative": ratio,
                "applied_rank": model.diagnostics["applied_rank"],
                "candidate_rank": model.diagnostics["numeric_candidate_rank"],
                "r1": model.diagnostics["r1_centered_block"],
                "r2": model.diagnostics["r2_centered_block"],
                "operator_norm": operator_norm, "compressed_norm": compressed_norm,
                "c25": sum(operator_norm ** (24-power) * compressed_norm ** power
                           for power in range(25)),
                "c25_envelope": 25 * operator_norm ** 24,
                "raw_contrast": raw_contrast, "projected_contrast": projected_contrast,
                "paired_effect": projected_contrast - raw_contrast,
                "absolute_gap_change": abs(projected_contrast) - abs(raw_contrast),
            })
        print(f"Restored replicate {seed}", flush=True)
    summaries = []
    for exponent in range(-6, 0):
        rows = [row for row in records if row["ridge_relative"] == 10.0 ** exponent]
        summary = {"ridge_relative": 10.0 ** exponent}
        for name in ("r1", "r2", "raw_contrast", "projected_contrast", "paired_effect"):
            summary[name] = edmd.summarize_seed_values([row[name] for row in rows])
        summaries.append(summary)
    sources = ["fa_edmd_mpc_v3_corrected.py", "scripts/reviewer_sensitivity.py"]
    return {"protocol": {"replicates": list(range(10)), "rank_cap": 4, "depth": 3,
                         "tolerance": 1e-8, "horizon": 25, "students": 150,
                         "random_root": edmd.RANDOM_ROOT,
                         "pairing": "Same training data and evaluation cohort at every ridge within replicate",
                         "scope": "Reproduction of reviewer sensitivity results; not independent new evidence"},
            "versions": edmd.provenance(),
            "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                              for name in sources}, "records": records, "summaries": summaries}


def coordinates(rows, name):
    return " ".join(f"({row['ridge_relative']:.12g},{row[name]:.16g})" for row in rows)


def render(data):
    records, summaries = data["records"], data["summaries"]
    assert len(records) == 60
    assert all(row["applied_rank"] == 4 and row["candidate_rank"] == 7 for row in records)
    assert all(row["projected_contrast"] < row["raw_contrast"] < 0 for row in records)
    reference = next(row for row in records if row["seed"] == 0 and row["ridge_relative"] == .001)
    np.testing.assert_allclose([reference["c25"], reference["c25_envelope"]],
                               [12.951824, 48.060377], atol=5e-7, rtol=0)
    archive = json.loads(Path("corrected-results/2026-09-14-final/result.json").read_text())
    np.testing.assert_allclose(reference["paired_effect"],
        archive["experiment"]["paired_projection_effects"]["cohort_contrast"], atol=1e-11, rtol=0)
    figure = [r"\begin{tikzpicture}[font=\small]",
              r"\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.7cm,vertical sep=2.1cm},",
              r"width=7.0cm,height=5.5cm,xmode=log,xmin=1e-6,xmax=1e-1,",
              r"xtick={1e-6,1e-5,1e-4,1e-3,1e-2,1e-1},xticklabel style={font=\scriptsize},",
              r"scaled y ticks=false,yticklabel style={/pgf/number format/fixed},",
              r"xlabel={$\eta/\lambda_{\max}(G)$},axis lines=left,grid=major,grid style={black!10},",
              r"legend style={draw=none,font=\scriptsize},legend cell align=left]"]
    figure.append(r"\nextgroupplot[title={(a) Selected and candidate ranks},ylabel={Dimension},ymin=3,ymax=8,ytick={4,5,6,7},legend style={at={(0.5,0.5)},anchor=center}]")
    for name, color, label in [("applied_rank", "blue", "Applied rank"), ("candidate_rank", "orange", "Candidate rank")]:
        rows = [row for row in records if row["seed"] == 0]
        figure.append(r"\addplot[" + color + r",thick,mark=*] coordinates {" + coordinates(rows, name) + "};")
        figure.append(r"\addlegendentry{" + label + "}")
    panels = [
        (r"\nextgroupplot[title={(b) Cross-block residuals},ylabel={Weighted Frobenius norm},legend pos=north west]",
         [("r1", "blue", r"$r_1$"), ("r2", "orange", r"$r_2$")]),
        (r"\nextgroupplot[title={(c) Terminal physical contrasts},ylabel={$\delta_{25}$},legend style={at={(0.55,0.7)},anchor=center}]",
         [("raw_contrast", "blue", "Raw MPC"), ("projected_contrast", "orange", "Projected MPC")]),
        (r"\nextgroupplot[title={(d) Paired projection effects},ylabel={$\tau_\delta$},ymax=0.01,legend style={at={(0.5,0.8)},anchor=center}]",
         [("paired_effect", "blue", "Mean paired effect")])]
    for header, series in panels:
        figure.append(header)
        for name, color, label in series:
            for seed in range(10):
                rows = [row for row in records if row["seed"] == seed]
                figure.append(r"\addplot[" + color + r"!25,thin,forget plot] coordinates {" + coordinates(rows, name) + "};")
            means = [{"ridge_relative": row["ridge_relative"], name: row[name]["mean"]} for row in summaries]
            figure.append(r"\addplot[" + color + r",thick,mark=*] coordinates {" + coordinates(means, name) + "};")
            figure.append(r"\addlegendentry{" + label + "}")
    figure.extend([r"\addplot[black,dashed,forget plot] coordinates {(1e-6,0) (1e-1,0)};",
                   r"\end{groupplot}", r"\end{tikzpicture}"])
    Path("reviewer_metric_figure.tex").write_text("\n".join(figure) + "\n")
    table = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Ridge ratio & Mean $r_1$ & Mean $r_2$ & Mean $\delta^{\mathrm{proj}}$ & Mean $\tau_\delta$ & 95\% interval \\", r"\midrule"]
    for index, row in enumerate(summaries):
        lower, upper = row["paired_effect"]["t95_interval"]
        values = " & ".join(f"{row[name]['mean']:.6f}" for name in ("r1", "r2", "projected_contrast", "paired_effect"))
        table.append(f"$10^{{{index-6}}}$ & {values} & $[{lower:.6f},{upper:.6f}]$ " + r"\\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    Path("reviewer_metric_table.tex").write_text("\n".join(table) + "\n")


def main():
    output = Path("reviewer-results/2026-09-16/metric-sensitivity.json")
    if output.exists():
        data = json.loads(output.read_text())
    else:
        data = run()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
    render(data)


if __name__ == "__main__":
    main()