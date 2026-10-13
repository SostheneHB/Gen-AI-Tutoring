"""
Table tab:general-price: nonlinear erasure on held-out students, compared with the frontier of
Proposition prop:general-price at the share of the gap actually left (achieved tau).

For each split: tau = (held-out gap of f_0) / (held-out gap of f*); nonlinear frontier = (1 - tau)^2 * price
(price estimated on the training students, out of fold); linear frontier = (1 - tau)^2 * linear price,
the cost of a linear erasure that leaves the same share of the gap.  Mean over five splits, one step ahead.

    python make_table_general_price.py   (reads out_rank_matched/general_price.json)
"""
import json
import os

import numpy as np

import rank_matched_controls as rm

NAMES = dict(assistments="ASSISTments (school)", eedi="Eedi (PremiumPupil)", oulad_imd="OULAD (deprivation)",
             oulad_disability="OULAD (disability)", oulad_gender="OULAD (gender)")


def rows(res, h_index=0):
    out = {}
    for k in NAMES:
        L = [r["linear"][0] for r in res[k]]
        N = [r["nonlinear"][h_index] for r in res[k]]
        tau = np.array([x["remaining_gap_test"] / x["original_gap_test"] for x in N])
        lin = np.array([x["formula"] for x in L])
        nl = np.array([x["formula"] for x in N])
        meas = np.array([x["measured_test"] for x in N])
        out[k] = dict(lin_price=lin.mean(), lin_r2=np.mean([x["r2_group"] for x in L]),
                      nl_price=nl.mean(), nl_r2=np.mean([x["r2_group"] for x in N]),
                      tau=tau.mean(), tau_min=tau.min(), tau_max=tau.max(),
                      measured=meas.mean(), nl_frontier=((1 - tau) ** 2 * nl).mean(),
                      lin_frontier=((1 - tau) ** 2 * lin).mean(), ratio=(meas / ((1 - tau) ** 2 * nl)).mean(),
                      min_abs_heldout_gap=float(np.min(np.abs([x["original_gap_test"] for x in N]))))
    return out


def main():
    res = json.load(open(os.path.join(rm.OUT, "general_price.json")))
    out = rows(res)
    json.dump(out, open(os.path.join(rm.OUT, "general_price_table.json"), "w"), indent=1)
    for k, r in out.items():
        print(f"{NAMES[k]} & {r['lin_price']:.4f} ({r['lin_r2']:.3f}) & {r['nl_price']:.4f} ({r['nl_r2']:.3f}) & "
              f"{r['tau']:+.2f} & {r['measured']:.4f} & {r['nl_frontier']:.4f} & {r['lin_frontier']:.4f}\\\\"
              f"   % ratio {r['ratio']:.2f}, tau range [{r['tau_min']:+.2f},{r['tau_max']:+.2f}], "
              f"min |held-out gap| {r['min_abs_heldout_gap']:.4f}")


if __name__ == "__main__":
    main()
