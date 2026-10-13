# Gen-AI-Tutoring

Code for the paper **"Hiding a Gap Does Not Close It: The Price of Removing the Mean Group Contrast from Learned Models of Learning Dynamics"**.

## Reproduction (release v1.3)

Everything needed to recompute the numbers, tables and figures of the paper is in [`reproduction/`](reproduction/):

- `reproduction/scripts/`: all scripts (simulator, EDMD fits, projection, MPC controllers, floor-constrained MPC,
  ASSISTments / Eedi / OULAD pipelines, structural diagnostics, rank-matched controls, price of erasure,
  the general price of erasure, the dynamics-only correction, the next-item target with deep knowledge
  tracing, the 50-replicate simulator study, and the experiments answering the external reviews);
- `reproduction/randomized_experiments/`: the analysis of 30 randomized ASSISTments experiments, with the
  analysis plan and the analysis code as hashed before any outcome file was written;
- `reproduction/environment.yml`: the locked environment (`conda env create -f environment.yml`);
- `reproduction/README.md`: the commands, in order, with expected run times and data checksums.

**No data are included.** ASSISTments and its randomized experiments (terms of use), Eedi (CC BY-NC-ND 4.0) and OULAD (CC BY 4.0) must be downloaded
from their providers; outputs derived from them are not redistributed and are recomputed by the scripts.

## Archive

Release v1.3 is archived on Zenodo: [doi:10.5281/zenodo.23342342](https://doi.org/10.5281/zenodo.23342342). Earlier release v1.1: [doi:10.5281/zenodo.23329603](https://doi.org/10.5281/zenodo.23329603).

## Other files

The scripts at the top level and `notebooks/` are earlier exploratory versions, kept for history.
