# Hiding a Gap Does Not Close It: Auditing Group-Direction Removal in Learned Models of Learning Dynamics — reproduction package (revision 5)

Every number, table and figure of the revised manuscript is produced by the
scripts in `scripts/`. The numerical outputs are not included here: some are derived from licensed data (Eedi, CC BY-NC-ND) that may not be redistributed. Every output is recomputed by the commands below.
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
| Neural ablation, Table S2, Fig. S2 (Supplement S1) | `outputs/ablation_frozen_results.json` | `aggregate`, `runs[i].frozen` |
| Modularity check (1,334 cases) | `outputs/modularity_check.json` | `max_*` |
| D_K rate study | `out/dk_rate_fixed_metric.json` | as in the previous version |
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
