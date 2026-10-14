# Hiding a Gap Does Not Close It: The Price of Removing the Mean Group Contrast from Learned Models of Learning Dynamics — reproduction package (release v1.3)

Every number, table and figure of the revised manuscript is produced by the
scripts in `scripts/` and `randomized_experiments/code/`. The numerical outputs are not included here: some are derived from licensed data (Eedi, CC BY-NC-ND; ASSISTments) that may not be redistributed. Every output is recomputed by the commands below; the files named in the tables below are the files the scripts write.
All seeds are fixed (root seed 20260914 unless stated).

## Environment

```
conda env create -f environment.yml
conda activate opcomp-fairness-r2
```

All runs are CPU-only. The neural ablation is run with one thread per
configuration; with several threads per process PyTorch changes the
floating-point reduction order and the same seed gives different final losses.

## Data

ASSISTments 2012–2013 "with affect predictions" release,
`2012-2013-data-with-predictions-4-final.csv` (6,123,270 rows), available from
the ASSISTments data portal after accepting its terms of use. It is not
redistributed here.

| File | SHA-256 |
|---|---|
| `2012-2013-data-with-predictions-4-final.csv` | `1d06aee9e649c5ba9db49052fe9a86e68d0c3ccb4709a7487289d7721f5464db` |
| `assistments_features_time.csv` (built, timestamp order) | `bbd96758e74e7eb1a71c7f8668d63d81324d5c7e3f1ccf2c2961a1edf4229cce` |
| `assistments_features_pid.csv` (built, problem-id order) | `839a1549e1a4fcaae6034631c8848b6e481a7a10e60ac0a1db332201cda0e062` |

`build_features.py --order problem_id` reproduces, value for value, the
historical upstream file `assistments_features_with_pid.csv` used in the
previous version (all 2,711,753 rows; extra columns `correct`, `skill_id`,
`school_id`, `problem_log_id`, `start_time` are appended).

## Commands (from `scripts/`)

```
# 1. ASSISTments features (about 2 min each)
python build_features.py --raw <path>/2012-2013-data-with-predictions-4-final.csv --order time       --out assistments_features_time.csv
python build_features.py --raw <path>/2012-2013-data-with-predictions-4-final.csv --order problem_id --out assistments_features_pid.csv

# 2. ASSISTments pipeline: primary (timestamp) and sensitivity (problem id) (about 7 min each)
python assistments_pipeline.py --features assistments_features_time.csv --out ../outputs/assistments_time
python assistments_pipeline.py --features assistments_features_pid.csv  --out ../outputs/assistments_pid

# 3. Synthetic paired experiments, ridge sweep and conformal check (published bundle)
python rebuild_manuscript_results.py

# 4. Synthetic regime study (about 40 min)
python synthetic_regimes.py --sim . --out ../outputs/synthetic

# 5. Neural ablation with the common frozen diagnostic (about 2 min on 8 cores)
python run_ablation_frozen.py --workers 8 --out ../outputs/ablation_frozen_results.json

# 6. Side results: modularity identity and D_K rate
python verify_modularity.py --cases 1334 --out ../outputs/modularity_check.json
python dk_rate_fixed_metric.py            # writes ../out/dk_rate_fixed_metric.json

# 7. Tables and calibration figure of the third revision (paired differences, calibration, all regime settings, budget check)
python make_tables_R3.py --real ../outputs/assistments_time --synthetic ../outputs/synthetic --dest ../figures

# 8. Fourth revision: sample flow, structural test, beta_gap grid and T = 50 diagnostics, tables
python sample_flow.py --features assistments_features_time.csv --out ../outputs/sample_flow.csv
python structural_diagnostics.py --sim . --features assistments_features_time.csv --out ../outputs/out_structural.json   # exits with an error if an identity fails
python synthetic_extras.py --sim . --out ../outputs/synthetic_extras.json          # about 30 min
python make_tables_R4.py --synthetic ../outputs/synthetic --extras ../outputs/synthetic_extras.json \
    --structural ../outputs/out_structural.json --flow ../outputs/sample_flow.csv --dest ../figures

# 9. Pareto-fair MPC: budget frontier (about 20 min per gamma)
for g in 0.5 1 2; do python fair_frontier.py --sim . --gamma $g --out ../outputs/out_frontier_g$g.json; done
python make_frontier.py --inputs ../outputs/out_frontier_g0.5.json ../outputs/out_frontier_g1.json ../outputs/out_frontier_g2.json --dest ../figures

# 9b. Signed knowledge-gap change (Delta delta^k_T = Delta k_{T,1} - Delta k_{T,0}) added to the group-wise and frontier
#     tables, from the displayed rounded group means (no new interval); run after steps 8 and 9
python add_knowledge_gap_column.py ../figures

# 9c. Modularity figure (operator from population moments and from a 20,000-step walk)
python make_fig_modularity.py --dest ../figures

# 10. Eedi study (real sensitive attribute: PremiumPupil). Download the data from
#     https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip (CC BY-NC-ND 4.0; do not redistribute)
python build_features_eedi.py --data <Eedi folder>/data --out eedi_features_time.csv
python run_eedi.py --features eedi_features_time.csv --out ../outputs/eedi/out_eedi                       # about 40 min
python run_eedi.py --features eedi_features_time.csv --out ../outputs/eedi/out_eedi_strict --strict <Eedi folder>/data
python make_tables_eedi.py --main ../outputs/eedi/out_eedi --strict ../outputs/eedi/out_eedi_strict --dest ../figures

# 11. OULAD study (declared attributes: deprivation, disability, gender). Download from
#     https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset (CC BY 4.0)
python build_features_oulad.py --data <OULAD folder> --out oulad_features_weekly.csv
python run_oulad.py --features oulad_features_weekly.csv --out ../outputs/oulad          # about 10 min
python make_tables_oulad.py --oulad ../outputs/oulad --dest ../figures
python make_figures_eedi.py --eedi ../outputs/oulad/imd --prefix imd_student_split --stem oulad \
    --groups "Less deprived areas,Most deprived areas (IMD 0-30%)" --unit course-presentation \
    --splitname "student splits" --gaplabel "Gap in mean score (most deprived - others)" --ylim=-0.08,0.02 --dest ../figures
python make_figures_eedi.py --eedi ../outputs/eedi/out_eedi --dest ../figures              # Eedi figures

# 12. Figures
python make_figures_v2.py --real ../outputs/assistments_time --synthetic ../outputs/synthetic \
    --ablation ../outputs/ablation_frozen_results.json --dest ../figures
```

`assistments_pipeline_v1.py` is the previous pipeline, kept only to document
the change; it is not used for any number of the revised manuscript.

## Privacy and licences

No data file, and no per-student array, is included. The pipeline writes per-student prediction arrays
(`arrays.npz`) next to each `results.json` when it is run; they are regenerated locally and must not be shared,
because they are derived from licensed data (ASSISTments terms of use; Eedi CC BY-NC-ND 4.0).

## Where each reported number comes from

(The `outputs/...` files below are written by the scripts; they are not shipped in this repository.)

Table and figure numbers in the first part of this table refer to revision 4; later moves to the appendix changed
the numbering, so the rows below identify each item by its LaTeX label.

| Manuscript item | File | Key |
|---|---|---|
| Table 3 (splits, schools per stratum, p1, threshold) | `outputs/assistments_time/school_disjoint_j/results.json` | `n_train`, `n_val`, `n_test`, `schools`, `p1_train`, `p1_test`, `stratum_threshold_train_only` |
| Table 8, Fig. 9 (observed and replayed gaps, school-cluster CIs) | same | `replay.terminal_gap_observed[_school_ci]`, `replay.models.<m>.terminal_gap_pred`, `gap_error_school_ci` |
| Table 8 (MSE ratio, free-running effect) | same | `replay.models.<m>.mse`, `free_run.tau_delta[_school_ci]` |
| Split-level means and t4 intervals | `outputs/assistments_time/summary.json` | `school_disjoint.<quantity>` |
| Table 9, Fig. 10 (MSE by horizon, soft projection, recalibration) | `school_disjoint_j/results.json` | `replay.models.<m>.mse.<h>`, `replay.paired_vs_raw` |
| Ridge sensitivity on ASSISTments | same | `ridge_sweep` |
| TRS and magnitude diagnostics, Fig. 12 | same (per-student arrays are regenerated locally, not shipped) | `trs.*`, `free_run.spearman_*` |
| Table 12 (secondary and sensitivity analyses) | `outputs/assistments_{time,pid}/summary.json`, `student_split_full_strata/results.json` | `student_split`, `subsample`, `school_disjoint` |
| Ordering statistics (9.6 % of pairs, 39.5 % of rows) | computed from the two feature files | see `docs/ordering_check.py` |
| Synthetic paired effects, ridge sweep, conformal | `rebuild_manuscript_results.py` output bundle | as in the previous version |
| Regime study, comparators, Table 7, Fig. 6 | `outputs/synthetic/regimes.json` | `one_factor.<factor>[i]`, `gamma_x_beta[i]` |
| Neural ablation, Table S2, Fig. S2 (Supplement S1) | `outputs/ablation_frozen_results.json` | `aggregate`, `runs[i].frozen` |  (appendix removed in revision 7; script kept)
| Modularity check (1,334 cases) | `outputs/modularity_check.json` | `max_*` |  (appendix removed in revision 7; script kept)
| D_K rate study | `out/dk_rate_fixed_metric.json` | as in the previous version |  (appendix removed in revision 7; script kept)
| Paired differences with school CIs (Table 10) | `outputs/assistments_time/school_disjoint_j/results.json` | `replay.paired_vs_raw`, `replay.models.<m>.gap_error_school_ci` |
| Calibration table and reliability figure (Table 11, Fig. 11) | same | `calibration.<model>.<group>` |
| Other-student-label variant (Table 12) | `outputs/assistments_time/school_disjoint_otherlabel_j/results.json`, `summary.json` | `school_disjoint_otherlabel` |
| Constrained-controller feasibility and budget (Tables 16, 13) | `outputs/synthetic/regimes.json` | `<arm>.infeasible_fraction`, `<arm>.mean_abs_control`, `budget_check` |
| Sample flow 29,006 → 18,366 → 16,668 → 16,661 students (Table 2) | `outputs/sample_flow.csv` | one row per stage |
| Structural diagnostics for all fits (Table 18, Appendix E) | `outputs/out_structural.json` | `synthetic_all_fits`, `assistments:<family>`, `rows` |
| Effects on each group (Table 17) | `outputs/synthetic/regimes.json` | `effect_<arm>_vs_raw.{k_g0,k_g1,d_g0,d_g1}` |
| Log-scale beta_gap grid, multipliers (Table 14) | `outputs/synthetic_extras.json` | `beta_gap_grid[i]` |
| T = 50 diagnostics and 50-step training sensitivity (Fig. 5) | `outputs/synthetic_extras.json` | `horizon50_train25`, `horizon50_train50` |
| Pareto-fair MPC, budget frontier (Section 6.6, Fig. 7, Table 15, Proposition 3) | `outputs/out_frontier_g{0.5,1,2}.json` | `arms.<plain_rho|fair_c>.<quantity>` |
| Eedi study, Table 8 (PremiumPupil gap, class-disjoint and strict splits) | `outputs/eedi/out_eedi*/class_disjoint*/results.json`, `summary.json` | `replay.*`, `calibration.*`, `free_run.*` |
| OULAD study, Table 10, Figures 11-12 (deprivation, disability, gender) | `outputs/oulad/<attribute>/<attribute>_student_split_j/results.json`, `summary.json` | `replay.*`, `calibration.*`, `free_run.*` |

| `tab:rank-matched`, Figure `fig:rank-matched` A (rank-matched controls) | `outputs/rank_matched/<dataset>.json` | `across_splits.<arm>.<metric>`, `splits[j].percentile_of_group_in_draws` |
| Figure `fig:rank-matched` C-D (simulator policies) | `outputs/rank_matched/simulator.json` | `configs[i].effects_vs_raw.<arm>` |
| Proposition `prop:erasure-price`, Figure `fig:rank-matched` B | `outputs/rank_matched/erasure_price.json` | `<dataset>[j].taus.<tau>.{formula,excess_mse_train,excess_mse_test}`, `rho_yhat_ghat` |
| `tab:equal-budget` (equal-budget baselines) | `outputs/reviewer/budget.json` | `<gamma>.<arm>.{abs_gap,k0,k1,u1_ratio,u2_ratio}`, `<gamma>.fair_minus_<arm>` |
| Solver certification (Appendix `app:solver`, Table `tab:regimes-full` column Infeas.) | `outputs/reviewer/solver.json` | `[i].stats`, `[i].infeasible_share`, `[i].max_abs_change_vs_old` |
| Structural checks for Eedi, OULAD, group-aware model (Appendix `app:structural`) | `outputs/reviewer/structural.json` | `eedi`, `oulad`, `group_aware` |
| OULAD held-out presentations and time split (Section `sec:results-oulad`) | `outputs/reviewer/oulad.json` | `<attribute>.presentation_disjoint`, `<attribute>.time_split` |
| `tab:scenarios` (scenarios not used to design the method) | `outputs/reviewer/scenarios.json` | `<scenario>.<arm>_vs_<ref>` |
| `tab:replicates50` (50 replicates, Holm) | `outputs/reviewer/replicates.json` | `summary`, `rows[i].{proj_abs_gap,con_abs_gap,con_lower_group_k}.p_holm` |


## Revision 5: rank-matched controls and answers to the external review

All commands from `scripts/`; outputs are aggregates only (no per-student arrays).

```
# Rank-matched controls (Section "Is the loss caused by the group, or by compression?")
python rank_matched_controls.py --part A --dataset assistments      # also eedi, oulad_imd, oulad_gender, oulad_disability
python rank_matched_controls.py --part B --sim .                    # simulator policies
python erasure_price_check.py                                       # Proposition (price of erasure), formula vs measurement
python make_fig_rank_matched.py                                     # fig_rank_matched*.png

# External review
python reviewer_experiments.py budget       # equal-budget baselines for the Pareto-fair MPC (Table tab:equal-budget)
python reviewer_experiments.py solver       # certification of the gap-constrained solver with Eq. (gap-box-range)
python reviewer_experiments.py structural   # level-1 checks for Eedi, OULAD and the group-aware model
python reviewer_experiments.py oulad        # OULAD with held-out presentations and a 2013->2014 time split
python reviewer_experiments.py scenarios    # scenarios not used to design the method (Table tab:scenarios)
python reviewer_experiments.py replicates   # 28 settings, 50 replicates, Holm correction (Table tab:replicates50)
python make_tables_reviewer.py              # the three tables
```
`reviewer_experiments.py` and `rank_matched_controls.py` read the simulator from the path given in `SIM` / `--sim`
and the Eedi and OULAD feature files from `DATA`; edit these two paths for your machine.

## Paths

The scripts of revision 5 read three environment variables: `SIM_DIR` (folder with `fa_edmd_mpc_v3_corrected.py`, default `.`), `DATA_DIR` (folder with the Eedi and OULAD subfolders, default `data`) and `DATA_ROOT` (parent of a `Datasets/` folder, default `.`). Set them before running, for example `export SIM_DIR=. DATA_DIR=../data DATA_ROOT=..`.

## Revision 6: general price of erasure and check on randomized experiments

```
# 13. General (nonlinear) price of erasure and horizon profile (Proposition prop:general-price,
#     Table tab:general-price); same data as the rank-matched controls (about 15 min)
DATA_ROOT=/path/to/data python general_price.py        # -> out_rank_matched/general_price.json

# 14. Offline check on 30 randomized ASSISTments experiments (Discussion, "A first check with real
#     decisions"). Download experiment_conditions.csv and experiment_prior_stats_and_targets.csv from
#     OSF project k8ph9 (ASSISTments terms of use; do not redistribute) into randomized_experiments/data/
cd ../randomized_experiments/code
python audit.py                       # sample flow, groups (no outcomes used)
python analysis.py --sim null         # pipeline test on simulated outcomes
python analysis.py --sim effect
python analysis.py --real             # primary analysis
for s in propensity groups_completion ipw min10classes; do python analysis.py --real --sens $s; done
python balance.py
python secondary.py                   # per-experiment Holm tests, secondary outcome
```

`randomized_experiments/notes/analysis_plan.md` is the analysis plan (not registered; its original text was not
preserved); its "Deviations" section lists every later change. `notes/analysis_at_approval.py` is the code
as it was when the plan was fixed. The outputs hold only values pooled over students.

| Manuscript item | File | Key |
|---|---|---|
| Proposition `prop:general-price`, Table `tab:general-price`, horizon values | `outputs/rank_matched/general_price.json` | `<dataset>[j].linear[i].{formula,measured_train,measured_test}`, `<dataset>[j].nonlinear[i].{formula,measured_test,original_gap_test,remaining_gap_test}` |
| Randomized experiments: completion, gap, policy differences | `randomized_experiments/outputs/policy_real.json` | `values`, `pooled_vs_P2` |
| Randomized experiments: sensitivity analyses | `randomized_experiments/outputs/policy_real_<analysis>.json` | `pooled_vs_P2` |
| Randomized experiments: per-experiment tests, secondary outcome | `randomized_experiments/outputs/per_experiment_holm.json`, `secondary_problems.json` | `summary`, `vs_P2` |

## Revision 7: 50-replicate regime study for every arm, nonlinear check at the achieved tau

```
# 15. Regime study with 50 replicates for the raw, projected, gap-penalty and gap-constrained MPC
#     (about 40 min on one core), then Table tab:regimes, Figure fig:regimes and the counts of Section 7.4
SIM_DIR=/path/to/sim python reviewer_experiments.py replicates_full     # -> out_reviewer/replicates_full.json
python make_tables_replicates_full.py --dest ../figures                   # -> fig_synthetic_regimes.png, tab_regimes50_main.tex
# 16. Table tab:general-price (nonlinear erasure at the achieved tau), from general_price.json
python make_table_general_price.py
# 17. Budget-frontier figure and tables with the controller's new name (floor-constrained MPC)
python make_frontier.py --inputs out_frontier_g0.5.json out_frontier_g1.json out_frontier_g2.json --dest ../figures
```

| Manuscript item | File | Key |
|---|---|---|
| Section 7.4 counts, Table `tab:regimes`, Figure `fig:regimes` | `outputs/reviewer/replicates_full.json`, `replicates_full_counts.json` | `rows[i].<proj|pen|con>.<abs_gap|k|k0|k1>.{mean,ci,p_holm}`, `rows[i].raw_gap` |
| Table `tab:general-price` | `outputs/rank_matched/general_price_table.json` | `<dataset>.{lin_price,nl_price,tau,measured,nl_frontier,lin_frontier,ratio}` |

The controller called "Pareto-fair MPC" in earlier revisions is now called "floor-constrained MPC"; the module
`pareto_fair.py` keeps its old name.

## Revision 8: knowledge as primary simulator outcome, dynamics-only correction, next-item target with DKT

```
# 18. The three remaining simulator checks with 50 replicates (regime table with control use, budget check,
#     penalty grid), certified solver, with the knowledge gap; parallel (about 30 min on 6 cores)
SIM_DIR=/path/to/sim python e50_runs.py --workers 6 --out out_synthetic50
python make_tables_e50.py --src out_synthetic50 --dest ../figures
# 19. Dynamics-only correction on ASSISTments, Eedi and OULAD (about 5 min)
DATA_ROOT=/path/to/data python dynamics_only.py           # -> out_rank_matched/dynamics_only.json
# 20. Next-item target, EDMD readouts and deep knowledge tracing, CPU only (about 15 min per data set)
DATA_ROOT=/path/to/data python next_item.py --dataset assistments
DATA_ROOT=/path/to/data python next_item.py --dataset eedi
# 21. Randomized experiments: post-hoc spread of treatment effects (listed in the plan's Deviations)
cd ../randomized_experiments/code && python effect_spread.py
```

| Manuscript item | File | Key |
|---|---|---|
| Table `tab:regimes`, Section 7.4 counts on knowledge and both gaps | `outputs/synthetic50/regimes.json` | `one_factor.all[i].effect_<arm>_vs_raw.<knowledge|abs_kgap|abs_gap|k_g0|k_g1>.{mean,ci,p_holm}` |
| Tables `tab:regimes-full`, `tab:budget`, `tab:groupwise`, `tab:beta-gap` | `outputs/synthetic50/regimes.json`, `extras.json` | as in revision 4, with 50 replicates |
| Table `tab:dynamics-only` | `outputs/rank_matched/dynamics_only.json` | `summary.<dataset>.<arm>.{mse_k_h1,free_run_gap}`, `ga_same_state_diff_k`, `state_share_of_ga_gap` |
| Table `tab:next-item`, Proposition 5 on DKT | `outputs/rank_matched/next_item_<dataset>.json` | `summary.<arm>.{auc,worst_auc,brier,pred_gap,obs_gap}`, `summary.dkt_prop5` |
| Section `sec:results-rct`, spread of effects and allocation bound | `randomized_experiments/outputs/effect_spread.json` | `effect_low_percentiles_10_25_50_75_90`, `best_allocation_gap_change_weighted` |

Partial credit (0.01% of the ASSISTments answers) counts as incorrect in the next-item target. No data and no
per-student values are included; every output above is pooled over students.

## Randomized-experiment data

The 30 randomized experiments come from OSF project `k8ph9` (Sales, Prihar, Gagnon-Bartsch and Heffernan,
arXiv:2306.06273), under the ASSISTments terms of use: non-commercial use, no redistribution, no attempt to
re-identify students. They are not included here. Place `experiment_conditions.csv` and
`experiment_prior_stats_and_targets.csv` in `randomized_experiments/data/`. `randomized_experiments/notes/`
holds the analysis plan (not registered; its original text was not preserved), with the list of later changes,
and the analysis code as it was when its SHA-256 hash was recorded, before any outcome file was written.

## Release v1.2: margins applied (post hoc) and clipping check for the DKT erasure

```
# Margin rule of Table tab:margins, applied after the experiments to every held-out split
DATA_ROOT=/path/to/data python apply_margins.py           # -> out_reviewer/margins.json
# Next-item study, now also recording the share of erased DKT predictions outside [0, 1] and the cost of
# the unclipped erasure (same seeds; every other number is unchanged)
DATA_ROOT=/path/to/data python next_item.py --dataset eedi
```

(The output files named below are written by the scripts; they are not shipped in this repository.)

## Release v1.3 — horizons, DKT, controllers aimed at knowledge: horizons on common starting points, same-information DKT check, controllers aimed at knowledge

```
# 24. Price over the horizon: direct best linear predictor for each h and the iterated EDMD forecast, on a common
#     set of six starting points; reports the cross term 2 E[(f - y) d] of the iterated forecast
DATA_ROOT=/path/to/data python horizon_check.py                 # -> out_rank_matched/horizon_check.json
# 25. Next-item study; the DKT check now uses the same information (hidden state and next skill) for the
#     prediction and the group model, with class-bootstrap intervals and the calibration of the group model
DATA_ROOT=/path/to/data python next_item.py --dataset eedi
# 26. Controllers aimed at the disparity gap or at the knowledge gap, with and without floors, and a top-up at the
#     same budget; same group-aware model for every arm, 50 replicates, parallel (about 1 h on 5 cores)
SIM_DIR=/path/to/sim python knowledge_controller.py --workers 5  # -> out_reviewer/knowledge_controller.json
```

| Manuscript item | File | Key |
|---|---|---|
| Section 7.5, horizon paragraph | `outputs/rank_matched/horizon_check.json` | `summary.<dataset>[i].{direct,iterated}` |
| Section 7.7, DKT check | `outputs/rank_matched/next_item_eedi.json` | `splits[j].dkt_prop1_same_info.{frontier,measured,cross,ci,q_ece,q_calibration_slope}` |
| Table `tab:knowledge-controller` | `outputs/reviewer/knowledge_controller.json` | `summary.<gamma>.<arm>.{k0,k1,kgap,abs_kgap,abs_dgap,u1_ratio,u2_ratio}` |

## Release v1.3 — environment and further checks

**Environment.** The environment file of earlier revisions pinned Python 3.13.5 with numpy 1.26.4, a combination that
conda-forge cannot install. It now pins Python 3.12 with the same package versions. In a new environment built from
this file, `horizon_check.py` reproduced 450 numbers to within 1.3e-12 and `knowledge_controller.py` (2 replicates)
966 numbers to within 2.5e-8, against the original environment.

```
# 27. Horizons on common starting points, now with the forecast error of the raw and projected models
DATA_ROOT=/path/to/data python horizon_check.py
# 28. Next-item study with paired cluster-bootstrap intervals of AUC and Brier differences, and the accuracy of
#     the same-information DKT removal
DATA_ROOT=/path/to/data python next_item.py --dataset eedi
DATA_ROOT=/path/to/data python next_item.py --dataset assistments
# 29. ASSISTments: shuffled labels between schools (50 draws per split)
python school_permutation.py                                    # -> out_rank_matched/school_permutation.json
# 30. Eedi: pupils with and without a recorded attribute; OULAD: timing and missing data
DATA_ROOT=/path/to/data python eedi_inclusion.py
DATA_ROOT=/path/to/data python oulad_timing.py
# 31. Group-wise and frontier tables with the change of the knowledge gap
python make_tables_e50.py --src out_synthetic50 --dest ../figures
python make_frontier.py --inputs out_frontier_g0.5.json out_frontier_g1.json out_frontier_g2.json --dest ../figures
# 32. Randomized experiments: dependence between experiments (post-hoc sensitivity, listed in the plan)
cd ../randomized_experiments/code && python dependence_check.py
```

| Manuscript item | File | Key |
|---|---|---|
| Section 7.5, error ratios on common starting points | `outputs/rank_matched/horizon_check.json` | `summary.<dataset>[i].{mse_raw_test,mse_proj_test,mse_ratio_proj_raw}` |
| Section 7.7, paired AUC and Brier intervals | `outputs/rank_matched/next_item_<dataset>.json` | `summary.paired_vs_edmd_raw.<arm>.{auc,brier}` |
| Section 7.5, school-level shuffle | `outputs/rank_matched/school_permutation.json` | `summary`, `splits[j]` |
| Eedi selected sample; OULAD timing | `outputs/reviewer/eedi_inclusion.json`, `oulad_timing.json` | all keys |
| Appendix on dependence between randomized experiments | `randomized_experiments/outputs/dependence_check.json` | `overlap`, `classes_joint`, `students_joint` |

## Release v1.4 — part 1: reliability of the estimated price, controlled study, selected DKT, reference controller

The manuscript was reorganized: the main text now has the price and its estimation (Section 4), a controlled study
(Section 5), prediction on real data (Section 6), decisions in simulation (Section 7) and the audit protocol
(Section 8). The randomized experiments moved to an appendix.

```
# 33. Controlled study of the price and of its estimate: known truth, three alignments of the group signal, three
#     group separations, two shifts between training and test, two sample sizes, two learners (about 1 h on 2 cores)
python price_reliability_synthetic.py --workers 2          # -> out_reviewer/price_reliability_synthetic.json
# 34. Next-item prediction with a DKT selected on validation schools (grid of 12 configurations, early stopping),
#     four ways of removing the mean contrast, refit bootstrap of the announced price and the interpretability
#     rule; CPU only (about 25 min per split on ASSISTments and 45 min per split on Eedi with 3 threads)
DATA_ROOT=/path/to/data python next_item_tuned.py --dataset assistments
DATA_ROOT=/path/to/data python next_item_tuned.py --dataset eedi              # or several processes with --split-ids
DATA_ROOT=/path/to/data python next_item_tuned.py --dataset eedi --long --splits 2   # longer histories
# 35. Controllers against a true-model reference (same objective, information and budget; control weight chosen
#     on validation cohorts), 20 replicates, three group strengths, two objectives (about 1 h on 3 cores)
SIM_DIR=/path/to/sim python oracle_reference.py --workers 3                   # -> out_reviewer/oracle_reference.json
# 36. Data checks: OULAD score-availability delay (0, 1, 2, 4 weeks) and Eedi reweighting by the probability of a
#     recorded attribute
DATA_ROOT=/path/to/data python oulad_delay.py                                 # -> out_reviewer/oulad_delay.json
DATA_ROOT=/path/to/data python eedi_reweight.py                               # -> out_reviewer/eedi_reweight.json
# 37. Structural test on every fit of the 50-replicate study (650 simulator fits); exits with an error on failure
python structural_diagnostics.py --sim . --features assistments_features_time.csv --seeds 50 --out ../outputs/out_structural_50.json
# 38. Randomized experiments: assignment shares per experiment (no outcome read)
cd ../randomized_experiments/code && python randomization_table.py
# 39. Tables of this revision
python make_tables_review6.py --dest ../figures
```

### Map: every table and figure, its script and its output

| Label in the manuscript | Script (command above) | Output read |
|---|---|---|
| `tab:real-summary` | `assistments_pipeline.py` (2), `run_eedi.py` (10), `run_oulad.py` (11), `horizon_check.py` (27), `oulad_delay.py` (36) | `outputs/assistments_time/summary.json`, `outputs/eedi/out_eedi/summary.json`, `outputs/oulad/summary.json`, `outputs/rank_matched/horizon_check.json`, `outputs/reviewer/oulad_delay.json` |
| `tab:rank-matched`, `fig:rank-matched` (A, B) | `rank_matched_controls.py --part A`, `erasure_price_check.py`, `make_fig_rank_matched.py` (rev. 5) | `outputs/rank_matched/<dataset>.json`, `erasure_price.json` |
| `tab:erasure-price` | `erasure_price_check.py` (rev. 5) | `outputs/rank_matched/erasure_price.json` |
| `tab:general-price` | `general_price.py` (13), `make_table_general_price.py` (16) | `outputs/rank_matched/general_price_table.json` |
| `tab:controlled`, `tab:controlled-full` | `price_reliability_synthetic.py` (33), `make_tables_review6.py` (39) | `outputs/reviewer/price_reliability_synthetic.json` |
| `tab:next-item`, `tab:next-item-rel`, `tab:next-item-grid` | `next_item_tuned.py` (34), `make_tables_review6.py` (39) | `outputs/rank_matched/next_item_tuned_<dataset>*.json` |
| `tab:next-item-long` | `next_item_tuned.py --long` (34) | `outputs/rank_matched/next_item_tuned_eedi_long*.json` |
| `tab:next-item-fixed` | `next_item.py` (28) | `outputs/rank_matched/next_item_<dataset>.json` |
| `tab:dynamics-only` | `dynamics_only.py` (19) | `outputs/rank_matched/dynamics_only.json` |
| `tab:regimes`, `fig:regimes`, `tab:replicates50` | `e50_runs.py`, `make_tables_e50.py` (18), `reviewer_experiments.py replicates_full` (15) | `outputs/synthetic50/regimes.json`, `outputs/reviewer/replicates_full.json` |
| `fig:rank-matched-policy` | `rank_matched_controls.py --part B`, `make_fig_rank_matched.py` (rev. 5) | `outputs/rank_matched/simulator.json` |
| `tab:oracle`, `tab:oracle-diff` | `oracle_reference.py` (35), `make_tables_review6.py` (39) | `outputs/reviewer/oracle_reference.json` |
| `tab:knowledge-controller` | `knowledge_controller.py` (26) | `outputs/reviewer/knowledge_controller.json` |
| `tab:equal-budget`, `tab:scenarios` | `reviewer_experiments.py budget`, `reviewer_experiments.py scenarios` (rev. 5) | `outputs/reviewer/budget.json`, `scenarios.json` |
| `fig:frontier`, `tab:frontier` | `fair_frontier.py`, `make_frontier.py` (9, 31) | `outputs/out_frontier_g*.json` |
| `tab:regimes-full`, `tab:budget`, `tab:groupwise`, `tab:beta-gap` | `make_tables_e50.py` (18, 31) | `outputs/synthetic50/regimes.json`, `extras.json` |
| `fig:horizon50` | `synthetic_extras.py` (8) | `outputs/synthetic_extras.json` |
| `tab:flow` | `sample_flow.py` (8) | `outputs/sample_flow.csv` |
| `tab:structural` | `structural_diagnostics.py --seeds 50` (37), `make_tables_R4.py` (8) | `outputs/out_structural_50.json` |
| `tab:real-primary`, `fig:real-replay`, `tab:baselines`, `fig:real-baselines`, `tab:paired-differences`, `tab:calibration`, `fig:real-calibration`, `tab:real-sensitivity` | `assistments_pipeline.py` (2), `make_tables_R3.py` (7), `make_figures_v2.py` (12) | `outputs/assistments_time/`, `outputs/assistments_pid/` |
| `tab:eedi`, `fig:eedi-replay`, `fig:eedi-baselines` | `run_eedi.py`, `make_tables_eedi.py`, `make_figures_eedi.py` (10) | `outputs/eedi/` |
| `tab:oulad`, `fig:oulad-replay`, `fig:oulad-baselines` | `run_oulad.py`, `make_tables_oulad.py`, `make_figures_eedi.py` (11) | `outputs/oulad/` |
| `tab:oulad-delay` | `oulad_delay.py` (36) | `outputs/reviewer/oulad_delay.json` |
| Eedi reweighting paragraph | `eedi_reweight.py` (36) | `outputs/reviewer/eedi_reweight.json` |
| `tab:margins` decisions | `apply_margins.py` (22) | `outputs/reviewer/margins.json` |
| `fig:edmd`, `fig:paired-effects`, `tab:paired-summary`, `tab:edmd`, `fig:conformal-diagnostics`, `fig:metric-sensitivity`, `tab:metric-sensitivity` | `rebuild_manuscript_results.py` (3) | published bundle |
| `fig:real-trs` | `assistments_pipeline.py` (2), `make_figures_v2.py` (12) | `outputs/assistments_time/` (per-student arrays regenerated locally, not shipped) |
| `tab:rct` | `randomized_experiments/code/global_bootstrap.py` | `randomized_experiments/outputs/policy_real_globalclass.json` |
| `tab:randomization` | `randomized_experiments/code/randomization_table.py` (38) | `randomized_experiments/outputs/randomization_table.json` |
| Appendix on dependence between experiments | `randomized_experiments/code/dependence_check.py` (32) | `randomized_experiments/outputs/dependence_check.json` |
| `fig:validation-workflow` | drawn by hand | -- |

Every output listed above is an aggregate; none contains per-student values.

## Release v1.4 — part 2: the cost of a correction built from a group model, three-stage audit

The estimation procedure was revised: the price is now announced on validation units as the cost of the correction
built from the group model (Proposition `prop:built`), the diagnostic runs on validation units (D1-D3) and the
correction is confirmed on independent test units (T1-T2). The first version is kept for comparison.

```
# 40. Controlled study, second version: validation set, both procedures, calibrated-but-misspecified group model,
#     error and coverage among accepted replicates, test-stage confirmation (about 1 h on 4 cores)
python price_reliability_synthetic_v2.py --workers 4              # -> out_reviewer/price_reliability_synthetic_v2.json
# 41. Next-item: the selected DKT retrained with the same configuration and seed; first version, first version after
#     Platt recalibration, three-stage audit; the four removals compared at the same remaining gap; longer histories
DATA_ROOT=/path/to/data python next_item_v2.py --dataset assistments
DATA_ROOT=/path/to/data python next_item_v2.py --dataset eedi
DATA_ROOT=/path/to/data python next_item_v2.py --dataset eedi --long
# 42. Reference controller with the known equations: finer search and noise integration (about 20 min on 2 cores)
SIM_DIR=/path/to/sim python oracle_sensitivity.py --workers 2      # -> out_reviewer/oracle_sensitivity.json
# 43. Tables of this revision
python make_tables_review6.py --dest ../figures --only controlled_v2,next_item,next_item_v2,next_item_v2_long,oracle
```

| Label in the manuscript | Script (command above) | Output read |
|---|---|---|
| `tab:controlled`, `tab:controlled-full` (replace part 1) | `price_reliability_synthetic_v2.py` (40), `make_tables_review6.py` (43) | `outputs/reviewer/price_reliability_synthetic_v2.json` |
| `tab:next-item-rel` (replaces part 1), `tab:next-item-compare` | `next_item_v2.py` (41), `make_tables_review6.py` (43) | `outputs/rank_matched/next_item_v2_<dataset>.json` |
| `tab:next-item-long-v2` | `next_item_v2.py --long` (41) | `outputs/rank_matched/next_item_v2_eedi_long.json` |
| Sensitivity of the reference controller (Appendix `app:oracle`) | `oracle_sensitivity.py` (42) | `outputs/reviewer/oracle_sensitivity.json` |
| Realized budgets of the reference comparison (Section `sec:results-oracle`) | `oracle_reference.py` (35) | `outputs/reviewer/oracle_reference.json`, key `summary.<objective>|<gamma>.matched.<arm>.u1` |

Outputs are not distributed in this repository; the scripts recompute them. No data and no per-student values are included.

## Release v1.5: explicit targets of the audit, three separate intervals, bootstrap stability

The audit now announces two quantities on validation units: the quadratic part of the cost of the built correction
and its total cost (quadratic part plus the cross term 2 E[(f - y) d] on validation units). It keeps three
intervals apart: announced (training units resampled with the group model refitted, and validation units),
measured (test units only, correction fixed) and their difference (all three resampled). The checks D1-D3 decide
whether the estimate is accepted; T1-T2 check whether the gap reduction transfers to the test units; the agreement
of announced and measured cost is judged by the interval of their difference. The controlled study computes, for
each replicate, the realized cost of the fitted correction on 200,000 independent draws and reports how often each
interval contains it.

```
# 44. Controlled study with explicit targets (about 1 h on 3 cores), and bootstrap stability (B = 40, 60, 200, 400)
python price_reliability_synthetic_v3.py --workers 3                # -> out_reviewer/price_reliability_synthetic_v3.json
python price_reliability_synthetic_v3.py --workers 1 --stability    # -> out_reviewer/price_reliability_synthetic_v3_stability.json
# 45. Next-item audit with explicit targets (200 bootstrap draws), a second group score (hidden state only), and the
#     removals compared at a matched gap (descriptive) and with the intensity set on validation units (usable)
B_V3=200 DATA_ROOT=/path/to/data python next_item_v3.py --dataset eedi           # about 15 min per split
B_V3=200 DATA_ROOT=/path/to/data python next_item_v3.py --dataset assistments
B_V3=200 DATA_ROOT=/path/to/data python next_item_v3.py --dataset eedi --long
# 46. Tables
python make_tables_review6.py --dest ../figures --only controlled_v3,next_item,next_item_v3,next_item_v3_long
```

| Label in the manuscript | Script (command above) | Output read |
|---|---|---|
| `tab:controlled`, `tab:controlled-full-ls`, `tab:controlled-full-gb`, `tab:controlled-stability` | `price_reliability_synthetic_v3.py` (44), `make_tables_review6.py` (46) | `out_reviewer/price_reliability_synthetic_v3*.json` |
| `tab:next-item` (output correction row), `tab:next-item-rel`, `tab:next-item-audit`, `tab:next-item-audit-assist`, `tab:next-item-compare`, `tab:next-item-long-v2` | `next_item_v3.py` (45), `make_tables_review6.py` (46) | `out_rank_matched/next_item_v3_<dataset>*.json` |

**Where the protocol of Table `tab:audit-data` is implemented.**

| Quantity | Units | `next_item_v3.py`, function `analyse` | `price_reliability_synthetic_v3.py`, function `replicate` |
|---|---|---|---|
| Predictor `f`, group score `s` | training | network trained before `analyse`; `nt.fit_q(Htr, ...)` | `fit_f(ztr, ...)`, `fit_q(Q(ztr), ...)` |
| Coefficient, centre, announced quadratic part and total | validation | `announce(fva, sva, gva, yva)` | `ann(fva, qva, gva, yva, ...)` |
| Announced interval, checks D1-D3 | training (score refitted) + validation | loop over `B` draws: `rows_tr` weights, `fit_q`, `rows_va` resample, `announce` | loop `full`: training weights, refits, validation resample |
| Measured cost, its interval, T1-T2 | test, correction fixed | `cost_terms(fte, d, yte)`; loops `mdraw`, `tgap` over `rows_te` | `meas`; `bt` over test rows |
| Measured minus announced | training + validation + test | `diff_total`, `diff_quad`, in the draws of the announced interval with `rows_te` resampled | not used |
| Target of the controlled study | 200,000 new units | not used | `realised(...)` with the fitted `f`, `q`, coefficient and centre |

Outputs are not distributed in this repository; the scripts recompute them. No data and no per-student values are included.
