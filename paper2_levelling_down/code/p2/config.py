"""Which outcome is the fairness outcome D of the simulator experiments.

    P2_OUTCOME=knowledge (default)   D = final knowledge k_T: narrow the knowledge gap without lowering the
                                     mean knowledge of either group (main experiments);
    P2_OUTCOME=disparity             D = the synthetic disparity state d_T, as in the first version (extension).

OUT_IDX is the index of D in the simulator state (k, e, p, d); TAG is appended to output file names.
"""
import os

OUTCOME = os.environ.get("P2_OUTCOME", "knowledge")
if OUTCOME not in ("knowledge", "disparity"):
    raise ValueError("P2_OUTCOME must be 'knowledge' or 'disparity'")
OUT_IDX = 0 if OUTCOME == "knowledge" else 3
TAG = "" if OUTCOME == "disparity" else "_knowledge"

# Reference budget: mean help per student and step.  With knowledge as the outcome we use 0.1: at 0.3 most simulated
# students reach the ceiling k = 1 (94-100% at gamma <= 1), the knowledge gap vanishes and the linear model is far off.
REF_HELP = 0.1 if OUTCOME == "knowledge" else 0.3

# Which efficient plan is the reference (it is not unique in the group-aware model, where both groups have the same
# h^k): maximise total knowledge, then minimise |gap| ("min_gap"; review of 16 October 2026, third round).  The
# extension with the synthetic disparity keeps the plan returned by the solver, as in its first version.
REF_RULE = os.environ.get("P2_REF_RULE", "min_gap" if OUTCOME == "knowledge" else "solver")
