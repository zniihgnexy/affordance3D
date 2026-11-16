"""Generate notebooks/mem/S1_01_video_inputs_to_actionable_affordance.ipynb (mem branch, S1 steps 1-4)."""

import sys
from pathlib import Path

import nbformat as nbf

OUT_PATH = Path(sys.argv[1])
cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str, hidden: bool = False) -> None:
    cell = nbf.v4.new_code_cell(text.strip())
    if hidden:
        cell.metadata["jupyter"] = {"source_hidden": True}
    cells.append(cell)


# The cells live in build_mem_s1_01_parts/part_NN.py (kept under 500 lines each); they run here in order, in this module's namespace,
# exactly as when they were part of this file.
for _part in sorted((Path(__file__).parent / "build_mem_s1_01_parts").glob("part_*.py")):
    exec(compile(_part.read_text(encoding="utf-8"), str(_part), "exec"), globals())

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
notebook.metadata["language_info"] = {"name": "python"}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT_PATH)
print("wrote", OUT_PATH, "with", len(cells), "cells")
