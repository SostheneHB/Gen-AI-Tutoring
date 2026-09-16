from pathlib import Path
import json

import numpy as np


def main():
    bundle = Path("corrected-results/2026-09-14-final")
    results = json.loads((bundle / "result.json").read_text())
    with np.load(bundle / "trajectories.npz") as archive:
        groups = archive["groups"]
        trajectories = {name: archive[name] for name in
                        ("zero_raw", "zero_projected", "mpc_projected", "mpc_raw")}
    np.testing.assert_array_equal(trajectories["zero_raw"], trajectories["zero_projected"])
    for name, states in trajectories.items():
        contrast = states[groups == 1, -1, 3].mean() - states[groups == 0, -1, 3].mean()
        np.testing.assert_allclose(contrast, results["experiment"]["methods"][name]["cohort_contrast"],
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(states[:, -1, 0].mean(),
                                   results["experiment"]["methods"][name]["mean_knowledge"],
                                   rtol=0, atol=1e-12)
    figure = [
        r"\begin{tikzpicture}[font=\small]",
        r"\definecolor{journalgray}{HTML}{555555}",
        r"\definecolor{journalorange}{HTML}{D55E00}",
        r"\definecolor{journalblue}{HTML}{0072B2}",
        r"\begin{groupplot}[",
        r"group style={group size=2 by 1,horizontal sep=1.6cm},",
        r"width=7.4cm,height=6.1cm,",
        r"xlabel={Turn},xmin=0,xmax=25,xtick={0,5,10,15,20,25},",
        r"axis lines=left,grid=major,grid style={black!12},",
        r"tick align=outside,scaled ticks=false,",
        r"legend style={draw=none,fill=white,font=\footnotesize},",
        r"legend cell align=left]",
    ]
    panels = [
        r"\nextgroupplot[title={(a) Realized group contrast},ylabel={$\delta_t$ (group 1 minus group 0)},legend pos=north east]",
        r"\nextgroupplot[title={(b) Mean knowledge},ylabel={$\overline{k}_t$}]",
    ]
    styles = [
        ("zero_raw", "No control", "journalgray,densely dotted"),
        ("mpc_projected", "Projected MPC", "journalorange,dashed"),
        ("mpc_raw", "Raw MPC", "journalblue,solid"),
    ]
    for panel_index, options in enumerate(panels):
        figure.append(options)
        for name, label, style in styles:
            states = trajectories[name]
            values = (states[groups == 1, :, 3].mean(axis=0)
                      - states[groups == 0, :, 3].mean(axis=0)
                      if panel_index == 0 else states[:, :, 0].mean(axis=0))
            figure.append(r"\addplot+[mark=none,line width=1pt," + style + "] coordinates {")
            figure.extend(f"({turn},{value:.17g})" for turn, value in enumerate(values))
            figure.append("};")
            if panel_index == 0:
                figure.append(r"\addlegendentry{" + label + "}")
    figure.extend([r"\end{groupplot}", r"\end{tikzpicture}"])
    Path("journal_edmd_results.tex").write_text("\n".join(figure) + "\n")


if __name__ == "__main__":
    main()