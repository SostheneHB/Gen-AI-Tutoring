"""
Section 7.4 from the 50-replicate regime study (reviewer_experiments.py replicates_full):
table tab:regimes (selected settings), figure fig_synthetic_regimes.png (group strength) and the
counts quoted in the text (Holm correction over the 28 settings, 5% level).

    python make_tables_replicates_full.py --dest ../Revised_manuscript
"""
import argparse
import json
import os

import numpy as np

import make_figures_v2 as mf

ARMS = (("proj", "Projected MPC", mf.ORANGE), ("pen", "Gap-penalty MPC", mf.MAGENTA), ("con", "Gap-constrained MPC", mf.AQUA))
BASE = dict(rank=4, depth=3, horizon=25)


def key(r):
    c = r["config"]
    return (c["gamma"], c["rank"], c["depth"], c["horizon"], c["beta"])


def find(rows, gamma=1.0, beta=8.0, **kw):
    c = dict(BASE, **kw)
    for r in rows:
        if key(r) == (gamma, c["rank"], c["depth"], c["horizon"], beta):
            return r
    raise KeyError((gamma, beta, kw))


sig = lambda e: e["p_holm"] < 0.05
up = lambda e: sig(e) and e["mean"] > 0
down = lambda e: sig(e) and e["mean"] < 0


def counts(rows):
    eff = [r for r in rows if r["config"]["gamma"] != 0.0]
    c = dict(n=len(rows), n_group_effect=len(eff))
    for a in ("proj", "pen", "con"):
        c[a] = dict(widens=sum(up(r[a]["abs_gap"]) for r in rows), narrows=sum(down(r[a]["abs_gap"]) for r in rows),
                    k_down=sum(down(r[a]["k"]) for r in rows), k_up=sum(up(r[a]["k"]) for r in rows),
                    narrows_eff=sum(down(r[a]["abs_gap"]) for r in eff),
                    one_group_down_eff=sum(down(r[a]["k0"]) or down(r[a]["k1"]) for r in eff),
                    worst_group_range_eff=[min(min(r[a]["k0"]["mean"], r[a]["k1"]["mean"]) for r in eff),
                                           max(min(r[a]["k0"]["mean"], r[a]["k1"]["mean"]) for r in eff)],
                    abs_gap_range_eff=[min(r[a]["abs_gap"]["mean"] for r in eff), max(r[a]["abs_gap"]["mean"] for r in eff)])
    return c


def fmt(e, star=True):
    s = f"{e['mean']:+.3f}"
    if abs(e["mean"]) < 1.5e-3:
        s = f"{e['mean']:+.4f}" if sig(e) else "0.000"
    return f"\\({s}{'^*' if star and sig(e) else ''}\\)"


def table(rows):
    sel = [(r"\(\gamma=0\), \(\beta_{\mathrm{ind}}=8\)", find(rows, 0.0)),
           (r"\(\gamma=0.5\), \(\beta_{\mathrm{ind}}=8\)", find(rows, 0.5)),
           (r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=8\) (main setting)", find(rows, 1.0)),
           (r"\(\gamma=2\), \(\beta_{\mathrm{ind}}=8\)", find(rows, 2.0)),
           (r"\(\gamma=-1\), \(\beta_{\mathrm{ind}}=8\)", find(rows, -1.0)),
           (r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=0\)", find(rows, 1.0, 0.0)),
           (r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=32\)", find(rows, 1.0, 32.0)),
           (r"\(\gamma=1\), \(\beta_{\mathrm{ind}}=8\), \(T=50\)", find(rows, 1.0, horizon=50))]
    out = []
    for name, r in sel:
        cells = [f"\\({r['raw_gap']['mean']:+.3f}\\)" if abs(r['raw_gap']['mean']) >= 5e-4 else "\\(0.000\\)"]
        for a, _, _ in ARMS:
            cells += [fmt(r[a]["abs_gap"]), fmt(r[a]["k"])]
        out.append(name + " & " + " & ".join(cells) + "\\\\")
    return "\n".join(out) + "\n"


def figure(rows, dest):
    plt = mf.plt
    sweep = sorted([r for r in rows if (r["config"]["rank"], r["config"]["depth"], r["config"]["horizon"], r["config"]["beta"]) == (4, 3, 25, 8.0)],
                   key=lambda r: r["config"]["gamma"])
    g = np.array([r["config"]["gamma"] for r in sweep])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for q, ax, lab in (("abs_gap", axs[0], "Change in $|\\delta_T|$ vs raw MPC"), ("k", axs[1], "Change in $\\overline{k}_T$ vs raw MPC")):
        for off, (a, name, c) in zip((-0.04, 0, 0.04), ARMS):
            m = np.array([r[a][q]["mean"] for r in sweep])
            lo = np.array([r[a][q]["ci"][0] for r in sweep]); hi = np.array([r[a][q]["ci"][1] for r in sweep])
            ax.errorbar(g + off, m, yerr=[m - lo, hi - m], fmt="-o", color=c, ms=5, mec="white", capsize=2, lw=1.5, label=name)
        ax.axhline(0, color=mf.INK2, lw=0.8)
        ax.set_xlabel("Group strength $\\gamma$ in the simulator")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=7.5)
    mf.panel(axs[0], "(a) Final disparity gap (positive = wider)")
    mf.panel(axs[1], "(b) Mean final knowledge")
    fig.tight_layout()
    fig.savefig(f"{dest}/fig_synthetic_regimes.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out_reviewer/replicates_full.json")
    ap.add_argument("--dest", default="../Revised_manuscript")
    a = ap.parse_args()
    rows = json.load(open(a.src))["rows"]
    assert len(rows) == 28
    open(os.path.join(a.dest, "tab_regimes50_main.tex"), "w").write(table(rows))
    figure(rows, a.dest)
    c = counts(rows)
    json.dump(c, open("out_reviewer/replicates_full_counts.json", "w"), indent=1)
    print(json.dumps(c, indent=1))
    m = find(rows, 1.0)
    for q in ("abs_gap", "k", "k0", "k1"):
        print("main", q, {x: (round(m[x][q]["mean"], 3), [round(v, 3) for v in m[x][q]["ci"]]) for x in ("proj", "pen", "con")})
    print("beta 1/8/32 proj:", [round(find(rows, 1.0, b)["proj"]["abs_gap"]["mean"], 3) for b in (1.0, 8.0, 32.0)])
    print("rank 1..8 proj:", [round(find(rows, 1.0, rank=k)["proj"]["abs_gap"]["mean"], 3) for k in (1, 2, 4, 6, 8)])
    print("depth 1..5 proj:", [round(find(rows, 1.0, depth=k)["proj"]["abs_gap"]["mean"], 3) for k in (1, 2, 3, 5)])
    print("T 5..50 proj:", [round(find(rows, 1.0, horizon=k)["proj"]["abs_gap"]["mean"], 3) for k in (5, 10, 25, 50)])
    print("gamma -1 proj:", round(find(rows, -1.0)["proj"]["abs_gap"]["mean"], 3))
    for r in rows:
        print(key(r), " ".join(f"{x}:{r[x]['abs_gap']['mean']:+.3f}{'*' if sig(r[x]['abs_gap']) else ''}/{r[x]['k']['mean']:+.3f}{'*' if sig(r[x]['k']) else ''}" for x in ("proj", "pen", "con")),
              "con k0/k1", f"{r['con']['k0']['mean']:+.3f}{'*' if down(r['con']['k0']) else ''}", f"{r['con']['k1']['mean']:+.3f}{'*' if down(r['con']['k1']) else ''}",
              "pen k0/k1", f"{r['pen']['k0']['mean']:+.3f}{'*' if down(r['pen']['k0']) else ''}", f"{r['pen']['k1']['mean']:+.3f}{'*' if down(r['pen']['k1']) else ''}")


if __name__ == "__main__":
    main()
