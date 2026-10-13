# Analysis plan — Do group-aware assignment policies help the students behind? Evidence from randomized experiments in ASSISTments

Version 1, 12 October 2026. Written **before** any outcome was analysed (the audit in `code/audit.py`
uses assignment and baseline covariates only). Any change after outcomes are seen will be listed in a
"Deviations" section with its reason.

## 1. Why this study

The companion simulation study shows two things about tutoring policies: removing group information does not
narrow a gap, and narrowing a gap without lowering any group needs extra help directed to the group behind.
Its real-data studies only test *predictions*. The open question (reviewer point 4) is whether a policy
that uses group information really changes **outcomes** for real students. Randomized experiments, where
the assignment probability is known, allow this to be estimated without a new data collection.

## 2. Data

- Source: ASSISTments randomized experiments released with Gagnon-Bartsch, Sales et al. (JEDM, forthcoming;
  arXiv 2306.06273), OSF project `k8ph9`, under the ASSISTments Data Terms of Use (non-commercial, no
  redistribution, open outcomes, no re-identification). Files used: `experiment_conditions.csv`,
  `experiment_prior_stats_and_targets.csv`. The data stay local and are never shared or pushed.
- Unit: one student working on the experimental skill builder of one experiment.
- Sample (audit, rules R1–R4 in `code/audit.py`): 30 experiments, 30,640 records, 21,557 students,
  1,788 classes; 125 to 2,909 students per experiment (median 927).

## 3. Groups

Prior achievement, measured before randomization: terciles of `student_prior_average_problem_correctness`
on the whole retained sample (cut points 0.595 and 0.740). **Low** = bottom tercile, **high** = top tercile.
The middle tercile is kept in all overall values. Missing values (0.4%) form a separate category,
included in overall values and excluded from group comparisons. These are achievement groups, not
demographic groups; no attribute is inferred from names.

## 4. Outcomes

- Primary: `assignment_completed` (0/1), the outcome analysed by the data authors.
- Secondary: `problems_completed` among students who completed (effort needed to reach mastery).
- Gap: completion of the high group minus completion of the low group.

## 5. Policies compared, inside each experiment

A policy assigns each student to the control or the treatment condition of the experiment.

| Policy | Rule | Uses group |
|---|---|---|
| P0 control for all | everybody gets control | no |
| P1 treatment for all | everybody gets treatment | no |
| P2 best arm | the arm with the higher estimated overall completion | no |
| P3 best arm per group | in each achievement group, the arm with the higher estimated completion | yes |
| P4 personalised | arm with the higher predicted completion for the student (T-learner, gradient boosting on the baseline covariates) | no |
| P5 gap-aware with floors | like P4, but a student of the low group gets the arm that is better for the low group, subject to: the estimated completion of each group is not below its value under P2 (floor) | yes |

P2 is the reference ("blind" decision). P5 is the observational counterpart of the floor-constrained
controller of the simulation study, for a single decision.

## 6. Estimation

- Cross-fitting: in each experiment, classes are split into 5 folds (one fold per class when an experiment
  has fewer than 5 classes; added before any outcome analysis, after the pipeline test on simulated outcomes). Policies (and the predictions they
  use) are fitted on 4 folds and evaluated on the held-out fold; every student is evaluated once.
- Value of a policy for a group: doubly robust estimator with the known assignment probability 1/2 and an
  outcome model fitted on the other folds; the inverse-probability estimator is reported as a check.
- Pooling across experiments: student-weighted values over all experiments, with a joint cluster bootstrap
  (classes resampled within each experiment). Changed before any outcome analysis from inverse-variance
  (DerSimonian–Laird) pooling, because the pipeline test showed that experiments where two policies make
  identical choices have zero variance and would dominate such a pool. Per-experiment estimates are
  reported in an appendix.
- Uncertainty: cluster bootstrap over classes within experiments (2,000 draws). Experiments with very few
  classes give unreliable class-level intervals; their number is reported, and a sensitivity analysis
  excludes experiments with fewer than 10 classes.

## 7. Questions and decision rules (fixed in advance)

For each policy compared with P2, pooled over experiments:

- Q1. Does it raise the completion of the low group? (difference, 95% interval)
- Q2. Does it lower the completion of the high group by more than the **non-inferiority margin of 2
  percentage points**? (one-sided 95% upper limit of the loss)
- Q3. Does it narrow the gap? (difference in gap, 95% interval)
- Q4. What does it cost overall? (difference in overall completion)

A policy is called **gap-narrowing without levelling down** if Q3 shows a narrower gap (interval below 0)
and Q2 rules out a loss larger than 2 points for **each** group. Per-experiment tests use a Holm
correction across experiments. Null results are reported with the same detail as positive ones.

## 8. Sensitivity analyses

1. The 26 further experiments that pass the size rule but not the 50/50 test, with the observed treated
   share as the assignment probability (valid if the share was constant within the experiment). Most of
   these shares are close to 2/3 or 3/4, which fits conditions merged into one arm (2 of 3, or 3 of 4).
2. Groups by prior completion rate instead of prior correctness.
3. A balance check of the six baseline covariates between arms (standardized differences; the audit
   finds a median of 0.05 and 18% above 0.1, consistent with chance in small arms, to be tested formally).
4. Inverse-probability instead of doubly robust values.

## 9. Expected limits

- One decision per student, not a learning path; completion is an engagement and mastery proxy.
- Many A/B tests in education have small effects; the study may find little room for any policy to
  differ from P2. This would itself be informative.
- Groups are by prior achievement; conclusions do not transfer to demographic groups.
- The experiments were designed by other researchers for other questions; conditions vary between
  experiments (hints, feedback, presentation).

## Approval

Fixed by the authors on 12 October 2026, before any outcome was analysed. Any later change is listed under "Deviations".

## Deviations

None. After approval, the code was extended only to run the sensitivity analyses of Section 8
(option `--sens`, and `code/balance.py`); the primary result is unchanged to the last digit. The code as
approved is kept in `notes/analysis_at_approval.py`.

Clarifications (12 October 2026, written in `code/secondary.py`, which does not change `analysis.py`):

1. Secondary outcome (Section 4). The plan did not say which policies are evaluated on it. They are the
   policies learnt on completion (same folds), and their value on `problems_completed` is estimated among
   completers with the same doubly robust formula. Completion happens after randomization, so this is a
   descriptive comparison, not a causal one.
2. Per-experiment tests (Section 7). Two-sided percentile bootstrap p-values (classes resampled, 2,000
   draws), Holm correction across the 30 experiments, separately for each policy (P3, P4, P5 vs P2) and
   each quantity (low, high, gap). In two small experiments (PSA2KQB, 4 classes; PSAJY7C, 8 classes),
   8 to 11 of the 2,000 draws had no student of one group; these draws are dropped and counted
   (`draws_dropped`).
3. Reading of the decision rule (Section 7). The 2-point non-inferiority margin is checked for each group
   with the one-sided 5% lower limit of the difference with P2.

Deviation 1 (13 October 2026, after outcomes were seen; `code/global_bootstrap.py`). An external reader noted
that the same students and classes appear in several experiments. 547 of the 1,788 classes (58% of the
records) and 5,848 students do. Resampling classes separately inside each experiment ignores this
dependence. The corrected bootstrap gives each class one weight in all experiments. Point estimates do not
change. The primary intervals change by at most 0.1 point. In sensitivity 1 (56 experiments), the P5 gap
interval becomes [-1.9, +0.0] instead of [-1.8, -0.04], so no analysis now gives a significant narrowing.
The corrected intervals are the ones to report.

Post-hoc additions (13 October 2026, after the outcomes were seen; descriptive only, no test, no change to any
pre-specified analysis):

4. Spread of the estimated treatment effects by group and the largest gap change that any allocation of the two
   conditions by group could give (`code/effect_spread.py`, from `outputs/policy_real.json` only). Asked for by a
   referee of the companion simulation study, to show whether any allocation could have moved the gap.
5. Decision of the authors (13 October 2026): this study is reported as a section of the main paper.

6. Dependence between experiments (13 October 2026, after a referee report; `code/dependence_check.py`): 27% of the
   students and 31% of the classes appear in more than one experiment. The pooled differences were recomputed with
   classes, and with students, resampled jointly across experiments. The intervals hardly change (P5 gap: [-1.6, +0.6]
   and [-1.6, +0.7] against [-1.6, +0.7]). Sensitivity only; the primary analysis is unchanged.
