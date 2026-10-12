# Gen-AI-Tutoring

Code for the paper **"Hiding a Gap Does Not Close It: Auditing Group-Direction Removal in Learned Models of Learning Dynamics"**.

## Reproduction (release v1.0)

Everything needed to recompute the numbers, tables and figures of the paper is in [`reproduction/`](reproduction/):

- `reproduction/scripts/`: all scripts (simulator, EDMD fits, projection, MPC controllers, Pareto-fair MPC,
  ASSISTments / Eedi / OULAD pipelines, structural diagnostics, rank-matched controls, price of erasure,
  and the experiments answering the external review);
- `reproduction/environment.yml`: the locked environment (`conda env create -f environment.yml`);
- `reproduction/README.md`: the commands, in order, with expected run times and data checksums.

**No data are included.** ASSISTments (terms of use), Eedi (CC BY-NC-ND 4.0) and OULAD (CC BY 4.0) must be downloaded
from their providers; outputs derived from them are not redistributed and are recomputed by the scripts.

## Other files

The scripts at the top level and `notebooks/` are earlier exploratory versions, kept for history.
