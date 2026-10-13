"""Figure of the modularity side result (appendix): a two-community
graph and the spectrum of the random-walk matrix P against the coefficient
operator of the full-indicator dictionary.

    python make_fig_modularity.py --dest ../Revised_manuscript

The coefficient operator is not copied from P. It is computed (i) from the
population moments of the indicator dictionary under the stationary law,
K = G^{-1} C_+ with G = diag(pi) and C_+ = diag(pi) P, and (ii) by EDMD from one
simulated stationary walk of N steps, K_hat = G_hat^+ C_hat_+.  (i) must equal P
to machine precision; (ii) must be close to P but not equal.
(This replaces an earlier notebook version whose panel compared P with itself.)
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def make_graph(n1=6, n2=6, p_in=0.85, p_out=0.15, seed=42):
    for attempt in range(50):
        g = nx.Graph(nx.stochastic_block_model([n1, n2], [[p_in, p_out], [p_out, p_in]], seed=seed + attempt))
        if all(d >= 1 for _, d in g.degree()) and nx.is_connected(g):
            break
    rng = np.random.default_rng(seed + 1)
    for u, v in g.edges():
        g[u][v]["weight"] = float(rng.uniform(0.5, 1.5))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="../Revised_manuscript")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    g = make_graph(seed=a.seed)
    A = nx.to_numpy_array(g, weight="weight")
    n = len(A)
    k = A.sum(1)
    P = A / k[:, None]
    pi = k / k.sum()
    comm = np.array([0] * (n // 2) + [1] * (n - n // 2))
    grp = np.arange(n) % 2

    # (i) population moments of the indicator dictionary
    G = np.diag(pi)
    Cp = np.diag(pi) @ P
    K_pop = np.linalg.solve(G, Cp)
    # (ii) EDMD from a simulated stationary walk
    rng = np.random.default_rng(a.seed)
    x = np.empty(a.steps + 1, dtype=int)
    x[0] = rng.choice(n, p=pi)
    for t in range(a.steps):
        x[t + 1] = rng.choice(n, p=P[x[t]])
    Z, Zn = np.eye(n)[x[:-1]], np.eye(n)[x[1:]]
    K_hat = np.linalg.pinv(Z.T @ Z / a.steps) @ (Z.T @ Zn / a.steps)

    ev = lambda M: np.sort(np.real(np.linalg.eigvals(M)))[::-1]
    eP, ePop, eHat = ev(P), ev(K_pop), ev(K_hat)
    print(f"max |eig(P) - eig(K_pop)| = {np.abs(eP - ePop).max():.1e}   (must be ~1e-15)")
    print(f"max |eig(P) - eig(K_hat)| = {np.abs(eP - eHat).max():.3f}  (EDMD, {a.steps} steps; must be small, not 0)")
    print(f"max |K_pop - P| = {np.abs(K_pop - P).max():.1e};  max |K_hat - P| = {np.abs(K_hat - P).max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pos = nx.spring_layout(g, seed=a.seed)
    for node in g.nodes():
        nx.draw_networkx_nodes(g, pos, nodelist=[node], node_color=["#3b82f6" if comm[node] == 0 else "#ef4444"],
                               node_shape="o" if grp[node] == 0 else "s", node_size=100 + 30 * k[node],
                               edgecolors="black", ax=axes[0])
    nx.draw_networkx_edges(g, pos, alpha=0.4, ax=axes[0])
    axes[0].set_title("(a) Example graph with two communities")
    axes[0].axis("off")
    idx = np.arange(n)
    axes[1].plot(idx, eP, "o-", label=r"$\lambda(P)$", ms=8, lw=1.5)
    axes[1].plot(idx, ePop, "x--", label=r"$\lambda(K)$, population moments", ms=9, lw=1.2)
    axes[1].plot(idx, eHat, "^:", label=fr"$\lambda(\widehat K)$, EDMD from a walk of {a.steps:,} steps", ms=6, lw=1.0, alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_title("(b) Spectrum of $P$ and of the indicator-dictionary operator")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{a.dest}/fig_modularity_correspondence.png", dpi=150)
    print("saved", f"{a.dest}/fig_modularity_correspondence.png")


if __name__ == "__main__":
    main()
