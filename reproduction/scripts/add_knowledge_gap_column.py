"""Add the signed knowledge-gap change Delta delta^k_T = Delta k_{T,1} - Delta k_{T,0} to the group-wise and
frontier tables (manuscript, Eq. knowledge-gap).  It is computed from the DISPLAYED rounded group means,
without a new interval, as the table captions say.  Idempotent: a table that already has the column is left
unchanged.

    python code/add_knowledge_gap_column.py Revised_manuscript TMLR_submission
"""
import re
import sys
from pathlib import Path

NUM = r"([+-]\d+\.\d+)"
HEAD = r"\(\Delta\delta^k_T\)"


def add(path, k0_col, k1_col):
    s = Path(path).read_text()
    if "Delta\\delta^k_T" in s:
        return "already there"
    out = []
    for line in s.splitlines():
        if line.startswith("\\begin{tabular}"):
            line = line.replace("@{}}", "l@{}}", 1)
        elif "\\(\\Delta\\overline k_{T,1}\\)" in line:
            cells = line.rstrip("\\").split(" & ")
            cells.insert(k1_col + 1, HEAD)
            line = " & ".join(cells) + "\\\\"
        elif line.rstrip().endswith("\\\\") and "\\scriptsize" in line:
            cells = line.rstrip().rstrip("\\").split(" & ")
            k0 = float(re.match(NUM, cells[k0_col].strip()).group(1))
            k1 = float(re.match(NUM, cells[k1_col].strip()).group(1))
            cells.insert(k1_col + 1, f"{k1 - k0:+.3f}")
            line = " & ".join(cells) + "\\\\"
        out.append(line)
    Path(path).write_text("\n".join(out) + "\n")
    return "added"


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        print(folder, "groupwise:", add(f"{folder}/tab_groupwise.tex", 3, 4), "| frontier:", add(f"{folder}/tab_frontier.tex", 4, 5))
