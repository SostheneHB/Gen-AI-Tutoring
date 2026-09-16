from pathlib import Path
import json

import numpy as np
from scipy.stats import t as student_t


def start_figure(columns=2, rows=1, height="6.1cm"):
    return [
        r"\begin{tikzpicture}[font=\small]",
        r"\definecolor{journalgray}{HTML}{555555}",
        r"\definecolor{journalorange}{HTML}{D55E00}",
        r"\definecolor{journalblue}{HTML}{0072B2}",
        r"\definecolor{journalgreen}{HTML}{009E73}",
        r"\begin{groupplot}[",
        f"group style={{group size={columns} by {rows},horizontal sep=1.65cm,vertical sep=2.3cm}},",
        f"width=7.4cm,height={height},",
        r"axis lines=left,grid=major,grid style={black!12},tick align=outside,scaled ticks=false,",
        r"legend style={draw=none,fill=white,font=\scriptsize},legend cell align=left]",
    ]


def add_curve(figure, horizontal, vertical, style, legend=None):
    coordinates = np.column_stack([horizontal, vertical])
    if not np.isfinite(coordinates).all():
        raise ValueError("Figure data must be finite.")
    figure.append(r"\addplot[" + style + "] coordinates {")
    figure.extend(f"({horizontal_value:.17g},{vertical_value:.17g})"
                  for horizontal_value, vertical_value in coordinates)
    figure.append("};")
    if legend is not None:
        figure.append(r"\addlegendentry{" + legend + "}")


def finish_figure(path, figure):
    figure.extend([r"\end{groupplot}", r"\end{tikzpicture}"])
    Path(path).write_text("\n".join(figure) + "\n")


def paired_effects(summary):
    figure = start_figure()
    seeds = np.array([record["seed"] for record in summary["per_seed"]])
    labels = [("cohort_contrast", r"(a) Effect on disparity contrast", r"$\tau_\delta$"),
              ("mean_knowledge", r"(b) Effect on mean knowledge", r"$\tau_k$")]
    for key, title, ylabel in labels:
        differences = np.array([record["methods"]["mpc_projected"][key]
                                - record["methods"]["mpc_raw"][key]
                                for record in summary["per_seed"]])
        mean = differences.mean()
        halfwidth = student_t.ppf(.975, len(differences) - 1) * differences.std(ddof=1) / np.sqrt(len(differences))
        np.testing.assert_allclose([mean - halfwidth, mean + halfwidth],
                                   summary["paired_projection_effects"][key]["t95_interval"],
                                   rtol=0, atol=1e-12)
        tick_options = (r"ytick={-0.145,-0.140,-0.135,-0.130},ymin=-0.145,ymax=-0.128,"
                        r"yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3},"
                        if key == "cohort_contrast" else "")
        figure.append(r"\nextgroupplot[" + tick_options + "title={" + title + "},ylabel={" + ylabel + "},"
                      r"xlabel={Replicate / summary},xmin=-0.5,xmax=11.8,"
                      r"xtick={0,1,2,3,4,5,6,7,8,9,11},xticklabels={0,1,2,3,4,5,6,7,8,9,Mean}]")
        add_curve(figure, seeds, differences, "only marks,mark=*,journalorange,mark options={draw=journalorange,fill=journalorange},mark size=2pt")
        figure.append(r"\addplot[only marks,mark=diamond*,journalblue,mark options={draw=journalblue,fill=journalblue},mark size=3pt,"
                      r"error bars/.cd,y dir=both,y explicit] coordinates {")
        figure.append(f"(11,{mean:.17g}) +- (0,{halfwidth:.17g})" + "};")
    finish_figure("journal_paired_effects.tex", figure)
    table = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Projected minus raw MPC & Mean effect & Sample SD & 95\% interval for mean \\",
             r"\midrule"]
    for key, label in [("cohort_contrast", r"Disparity contrast $\delta$"),
                       ("mean_disparity", r"Pooled disparity $\overline D_T$"),
                       ("mean_knowledge", r"Pooled knowledge $\overline k_T$"),
                       ("knowledge_group0", "Knowledge, group 0"),
                       ("knowledge_group1", "Knowledge, group 1")]:
        stats = summary["paired_projection_effects"][key]
        differences = np.array([record["methods"]["mpc_projected"][key]
                                - record["methods"]["mpc_raw"][key]
                                for record in summary["per_seed"]])
        halfwidth = student_t.ppf(.975, len(differences) - 1) * differences.std(ddof=1) / np.sqrt(len(differences))
        np.testing.assert_allclose([differences.mean(), differences.std(ddof=1)],
                                   [stats["mean"], stats["sample_sd"]], rtol=0, atol=1e-12)
        np.testing.assert_allclose([differences.mean() - halfwidth, differences.mean() + halfwidth],
                                   stats["t95_interval"], rtol=0, atol=1e-12)
        lower, upper = stats["t95_interval"]
        table.append(f"{label} & ${stats['mean']:+.6f}$ & ${stats['sample_sd']:.6f}$ & "
                     f"$[{lower:+.6f},{upper:+.6f}]$" + r" \\")
    table.extend([r"\bottomrule", r"\end{tabular}"])
    Path("journal_paired_summary.tex").write_text("\n".join(table) + "\n")


def conformal_diagnostics(conformal):
    pairs = np.array(conformal["test_true_predicted"])
    scores = np.array(conformal["test_scores"])
    calibration = np.array(conformal["calibration_scores"])
    intervals = np.array(conformal["test_intervals"])
    covered = np.array(conformal["test_covered"], dtype=bool)
    radius = conformal["quantile"]
    np.testing.assert_allclose(scores, np.abs(pairs[:, 0] - pairs[:, 1]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(intervals, np.column_stack([pairs[:, 1] - radius, pairs[:, 1] + radius]),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.sort(calibration)[conformal["order"] - 1], radius, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(covered, (pairs[:, 0] >= intervals[:, 0]) & (pairs[:, 0] <= intervals[:, 1]))
    np.testing.assert_allclose(covered.mean(), conformal["coverage"], rtol=0, atol=1e-12)
    figure = start_figure()
    figure.append(r"\nextgroupplot[title={(a) Absolute prediction errors},xlabel={Score},ylabel={Empirical CDF},"
                  r"ymin=0,ymax=1.03,legend pos=north west]")
    for values, style, label in [(calibration, "journalblue", "Calibration"),
                                 (scores, "journalorange,dashed", "Test")]:
        ordered = np.sort(values)
        horizontal = np.concatenate(([ordered[0]], ordered))
        vertical = np.arange(len(ordered) + 1) / len(ordered)
        add_curve(figure, horizontal, vertical, "const plot,mark=none,line width=1pt," + style, label)
    add_curve(figure, [radius, radius], [0, 1], "mark=none,journalgray,densely dotted", r"Radius $\widehat q$")
    figure.append(r"\nextgroupplot[title={(b) All held-out cohorts},xlabel={Cohort index (not time)},"
                  r"ylabel={Disparity contrast},xmin=-2,xmax=101,ymin=-0.52,ymax=0.52,legend pos=north east]")
    indices = np.arange(len(pairs))
    figure.append(r"\addplot[only marks,mark=none,journalblue!25,forget plot,"
                  r"error bars/.cd,y dir=both,y explicit,error bar style={line width=0.25pt}] coordinates {")
    for index, prediction in zip(indices, pairs[:, 1]):
        figure.append(f"({index},{prediction:.17g}) +- (0,{radius:.17g})")
    figure.append("};")
    add_curve(figure, indices, pairs[:, 1], "mark=none,journalblue,line width=1pt", "Predicted")
    add_curve(figure, indices[covered], pairs[covered, 0], "only marks,mark=*,mark size=1.1pt,journalgray", "Realized: covered")
    add_curve(figure, indices[~covered], pairs[~covered, 0], "only marks,mark=x,mark size=3pt,journalorange,thick", "Realized: missed")
    finish_figure("journal_conformal_diagnostics.tex", figure)


def latent_diagnostics(run):
    figure = start_figure(rows=2, height="5.7cm")
    history = run["history"]
    epochs = [record["epoch"] for record in history]
    figure.append(r"\nextgroupplot[title={(a) Recorded training losses},xlabel={Epoch},ylabel={Loss},"
                  r"legend pos=north west]")
    for key, style, label in [("total", "journalgray", "Total"),
                              ("latent", "journalblue,dashed", "Latent"),
                              ("pred", "journalorange,densely dotted", "Prediction")]:
        add_curve(figure, epochs, [record["losses"][key] for record in history],
                  "mark=none,line width=1pt," + style, label)
    for start, letter in [(0, "b"), (4, "c")]:
        raw = sorted([record for record in run["test"] if record["start"] == start
                      and record["method"] == "unprojected"], key=lambda record: record["horizon"])
        projected = sorted([record for record in run["test"] if record["start"] == start
                            and record["method"] == "projected"], key=lambda record: record["horizon"])
        np.testing.assert_allclose([record["real_ability_gap"] for record in raw],
                                   [record["real_ability_gap"] for record in projected], rtol=0, atol=1e-12)
        horizons = [record["horizon"] for record in raw]
        figure.append(r"\nextgroupplot[title={(" + letter + f") Ability contrast, start {start}" + "},"
                      r"xlabel={Forecast horizon},ylabel={Group 1 minus group 0},ymin=-0.2,ymax=2.5,"
                      r"xtick distance=1,legend pos=north west]")
        for records, key, style, label in [(raw, "real_ability_gap", "journalgray,densely dotted", "Realized"),
                                           (raw, "predicted_ability_gap", "journalblue", "Unprojected"),
                                           (projected, "predicted_ability_gap", "journalorange,dashed", "Projected")]:
            add_curve(figure, horizons, [record[key] for record in records],
                      "mark=none,line width=1pt," + style, label if start == 0 else None)
    figure.append(r"\nextgroupplot[title={(d) Group ability errors, start 4},xlabel={Forecast horizon},"
                  r"ylabel={Ability MSE},xtick distance=1,legend pos=north west]")
    for method, color in [("unprojected", "journalblue"),
                          ("projected", "journalorange")]:
        records = sorted([record for record in run["test"] if record["start"] == 4
                          and record["method"] == method], key=lambda record: record["horizon"])
        for group, linestyle in [(0, "densely dotted"), (1, "solid")]:
            add_curve(figure, [record["horizon"] for record in records],
                      [record[f"ability_mse_group{group}"] for record in records],
                      f"mark=none,line width=1pt,{color},{linestyle}")
    finish_figure("journal_latent_diagnostics.tex", figure)


def main():
    bundle = Path("corrected-results/2026-09-14-final")
    paired_effects(json.loads((bundle / "edmd-multiseed-summary.json").read_text()))
    conformal_diagnostics(json.loads((bundle / "result.json").read_text())["conformal"])
    latent_diagnostics(json.loads(Path("reruns/2026-09-12/dka/results.json").read_text()))


if __name__ == "__main__":
    main()