"""Generate report/week2_2026-11-16_to_11-22.ipynb: the one notebook of week 2 (Monday 16 to Sunday 22 November, London
time) for the project lead: a concise English report with charts (rewritten on 2026-11-23 at the project lead's request;
the earlier, longer Chinese version is in git history at b5b1267).

It reads only result files that already exist under runs/ (no model runs, CPU only) and checks every number quoted in its
text in the cell that shows it (listed in the appendix). It needs the part-vector runs, the public-evaluation and
stage-0 runs, the Gaussian trial, the report media (videos and the .rrd) and, for the stage-2 status, whatever of
runs/gaussians/fit_val82 and runs/stage2 exists at build time. Charts: report/week2_report.py.
"""

import sys
from pathlib import Path

import nbformat as nbf

OUT_PATH = Path(sys.argv[1])
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str, hidden: bool = True) -> None:
    cell = nbf.v4.new_code_cell(text.strip())
    if hidden:
        cell.metadata["jupyter"] = {"source_hidden": True}
    cells.append(cell)


# The cells live in build_report_2026_11_20_week2_parts/part_NN.py (kept under 500 lines each); they run here in order, in this module's namespace,
# exactly as when they were part of this file.
for _part in sorted((Path(__file__).parent / "build_report_2026_11_20_week2_parts").glob("part_*.py")):
    exec(compile(_part.read_text(encoding="utf-8"), str(_part), "exec"), globals())

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
notebook.metadata["language_info"] = {"name": "python"}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT_PATH)
print("wrote", OUT_PATH, "with", len(cells), "cells")
