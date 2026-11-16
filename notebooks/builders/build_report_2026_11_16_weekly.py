"""Generate report/2026-11-16_weekly_summary.ipynb: the weekly wrap-up of 2026-11-09 to 2026-11-15 for the project lead (Chinese).

It reads only result files that already exist under runs/ (no model runs, CPU only) and checks every number quoted in its
text against them (appendix A).
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


# The cells live in build_report_2026_11_16_weekly_parts/part_NN.py (kept under 500 lines each); they run here in order, in this module's namespace,
# exactly as when they were part of this file.
for _part in sorted((Path(__file__).parent / "build_report_2026_11_16_weekly_parts").glob("part_*.py")):
    exec(compile(_part.read_text(encoding="utf-8"), str(_part), "exec"), globals())

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
notebook.metadata["language_info"] = {"name": "python"}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT_PATH)
print("wrote", OUT_PATH, "with", len(cells), "cells")
