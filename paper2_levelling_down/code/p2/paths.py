"""One place for every path of this project.

    from p2.paths import PATHS, check
    check()            # prints what is available

Data are never copied into this project: data/ holds links to the original
folders (licences: ASSISTments terms of use, Eedi CC BY-NC-ND 4.0, OULAD CC BY 4.0,
FoundationalASSIST CC BY-NC 4.0 + access agreement).
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

PATHS = {
    "root": ROOT,
    "outputs": ROOT / "outputs",          # aggregate results only (no per-student files)
    "figures": ROOT / "figures",
    "manuscript": ROOT / "manuscript",
    "report": ROOT / "report",
    "paper1_code": ROOT / "code" / "paper1_base",
    # simulator of the companion paper [A] (fa_edmd_mpc_v3_corrected.py) is in paper1_base
    "simulator": ROOT / "code" / "paper1_base",
    # real data (links)
    "assistments_features": DATA / "ASSISTments_2012_2013" / "assistments_features_time.csv",
    "eedi_raw": DATA / "Eedi_NeurIPS2020" / "data",
    "eedi_features": DATA / "Eedi_NeurIPS2020" / "eedi_features_time.csv",
    "oulad_raw": DATA / "OULAD",
    "oulad_features": DATA / "OULAD" / "oulad_features_weekly.csv",
    "foundational_assist": DATA / "FoundationalASSIST" / "Data",
}


def use_paper1_code():
    """Make the frozen the companion paper [A] modules importable (assistments_pipeline, pareto_fair, ...)."""
    p = str(PATHS["paper1_code"])
    if p not in sys.path:
        sys.path.insert(0, p)


def check(verbose=True):
    ok = {}
    for k, v in PATHS.items():
        ok[k] = v.exists()
        if verbose:
            print(f"{'ok ' if ok[k] else 'MISSING'}  {k:22s} {v}")
    return ok


if __name__ == "__main__":
    check()
