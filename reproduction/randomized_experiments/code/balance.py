"""Sensitivity 3 of the plan: balance of the six baseline covariates between arms.

For each experiment, a logistic regression of the assigned arm on the standardized covariates, with a
likelihood-ratio test against the intercept-only model.  Under randomization the p-values are uniform.
Writes outputs/balance.json.  Uses no outcome.
"""
import json
import os

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression

import analysis as A


def main():
    m = A.load(False)
    pvals = {}
    for e, d in m.groupby("target_sequence"):
        X = d[A.COV].copy()
        X = X.fillna(X.median())
        X = ((X - X.mean()) / X.std().replace(0, 1)).to_numpy()
        a = d.A.to_numpy()
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X, a)
        p = np.clip(lr.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        ll1 = np.sum(a * np.log(p) + (1 - a) * np.log(1 - p))
        p0 = a.mean()
        ll0 = np.sum(a * np.log(p0) + (1 - a) * np.log(1 - p0))
        pvals[e] = float(stats.chi2.sf(2 * (ll1 - ll0), X.shape[1]))
    v = np.array(list(pvals.values()))
    out = dict(n_experiments=len(v), share_below_0_05=float((v < .05).mean()), share_below_0_10=float((v < .10).mean()),
               ks_uniform_p=float(stats.kstest(v, "uniform").pvalue), min_p=float(v.min()),
               holm_min_p=float(min(1.0, v.min() * len(v))), pvalues=pvals)
    json.dump(out, open(os.path.join(A.OUT, "balance.json"), "w"), indent=1)
    print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in out.items() if k != "pvalues"})


if __name__ == "__main__":
    main()
