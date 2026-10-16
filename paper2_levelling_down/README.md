# Narrowing a gap without levelling down: code of two companion papers

This folder holds the code of two related papers, both under review:

- **[C] Narrowing a Gap without Levelling Down: Price, Welfare and Certified Decisions with Learned Models** (decisions, certificates; simulators only);
- **[B] What Learning Logs Can and Cannot Tell About Narrowing a Gap without Levelling Down** (educational data).

`code/paper1_base/` is the frozen simulator and model code of the companion paper [A] (the rest of this repository).
Only code is published here: **no data, no derived data file, no per-student output**. Every script writes aggregate numbers only.

Install: `pip install -r requirements.txt` (Python 3.11+). Run every script from this folder, for example
`P2_OUTCOME=knowledge python code/scripts/run_price_of_fairness.py`. Tests: `python -m pytest code/tests`.

## Paper [C]: certified decisions (no data set needed)

All results come from simulators. The semi-synthetic simulator has its ten Eedi-fitted parameters built in, so no data set is needed to rerun it. Python 3.11+, packages in `requirements.txt`.
Run from this folder, with `P2_OUTCOME=knowledge` (main results) or `P2_OUTCOME=disparity` (Appendix C). Seeds are fixed;
each script writes aggregate numbers to `outputs/<experiment>/`.

| Result | Script |
|---|---|
| Price figure, Theorem 2 checks | `run_price_of_fairness.py`, `make_fig02_price.py` |
| Welfare proposition checks | `run_welfare.py` |
| Assumption cases, Remark 1 | `run_assumption_cases.py`; assumptions in the main models: `run_main_assumptions.py` |
| Capacity versus direct search, equal budget | `run_capacity_simulator.py` |
| Choice of the efficient reference | `run_reference_sensitivity.py` |
| Sensitivity table (learned model) | `run_model_sensitivity.py` |
| Training controls U(0,1) | `run_training_controls.py` |
| Certificate table | `run_certificate_v3.py --regime standard|heterogeneous|hard`, then `make_tab_certificate.py` |
| Theorem 4 experiment (estimated effects) | `run_certificate_estimated.py --regime heterogeneous|hard` (or one model per process with `--sizes --gammas --seeds --tag`), then `merge_certificate_estimated.py` |
| Individual losses and temporary drops | `run_individual_losses.py` |
| Planning figure (estimated effects) | `make_fig07_estimated_planning.py` |
| Semi-synthetic test (Eedi-calibrated) | `run_semisynthetic.py --kappas 0.25 0.5 1.0` then `make_tab_semisynthetic.py` (parameters built in; to refit them from the Eedi files: `fit_bkt_eedi.py`) |
| Appendix C audit | `audit_certificate_scores.py` |
| Tests | `python -m pytest code/tests` |

`code/paper1_base/` is the simulator and model code of the companion paper [A], unchanged.

## Paper [B]: what learning logs can tell (data downloaded from their providers)

Python 3.11+, packages in `requirements.txt`. Seeds are fixed; each script writes aggregate numbers only.
**No data are distributed.** Download them from their providers and place them as described in `code/p2/paths.py`:
OULAD (CC BY 4.0), Eedi NeurIPS 2020 Education Challenge (CC BY-NC-ND 4.0), ASSISTments 2012-2013 and the ASSISTments
randomized experiments (OSF project k8ph9; ASSISTments terms of use: cite, non-commercial, no redistribution).
The randomized-experiment files go in `code/rct/data/`.

| Result | Script |
|---|---|
| Decomposition figure (simulator) | `run_decomposition_simulator.py` |
| OULAD start and imputation figures | `run_oulad_start_sensitivity.py`, `run_oulad_first_scores.py`, `make_fig05_oulad_start.py`, `make_fig06_oulad_imputation.py` |
| Eedi decomposition and instrumental variables | `run_eedi_iv_diagnostics.py` |
| Randomized experiments (K = 20, 40, 50) | `run_rct_certificate.py --cohorts K` (uses `code/rct/code/analysis.py` for the sample and the groups) |
| Hints as the control | `extract_assistments_hints.py --raw <raw csv>`, then `run_assistments_hints.py` |

`code/paper1_base/` is the model and data-construction code of the companion paper [A], unchanged.
