"""scripts/merge_notebooks.py: links, headings, ids and the merged layout."""
import importlib.util
from pathlib import Path

import nbformat as nbf

SPEC = importlib.util.spec_from_file_location("merge_notebooks", Path(__file__).resolve().parents[1] / "scripts" / "merge_notebooks.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

ANCHORS = {"notebooks/mem/pipeline.ipynb": "w1-pipeline", "report/a.ipynb": "w1-a"}


def test_link_to_a_merged_notebook_becomes_its_anchor():
    assert M.retarget("pipeline.ipynb", "notebooks/mem", "report", ANCHORS) == "#w1-pipeline"
    assert M.retarget("../../report/a.ipynb", "notebooks/mem", "report", ANCHORS) == "#w1-a"


def test_link_fragment_is_kept_when_it_names_a_heading():
    assert M.retarget("pipeline.ipynb#block-07", "notebooks/mem", "report", ANCHORS) == "#block-07"


def test_link_to_another_file_is_seen_from_the_output_folder():
    assert M.retarget("../../docs/mem/x.md", "notebooks/mem", "report", ANCHORS) == "../docs/mem/x.md"
    assert M.retarget("../docs/y.md#s2", "notebooks", "report", ANCHORS) == "../docs/y.md#s2"


def test_external_links_and_attachments_are_left_alone():
    for target in ("https://arxiv.org/abs/1", "#local", "attachment:fig.png", "data:image/png;base64,AA", "/abs/path"):
        assert M.retarget(target, "notebooks/mem", "report", ANCHORS) == target


def test_headings_go_one_level_down_outside_code_fences_only():
    text = "# Title\n## Sub\n```bash\n# comment [x](pipeline.ipynb)\n```\nsee [p](pipeline.ipynb) #tag"
    out = M.rewrite_markdown(text, "notebooks/mem", "report", ANCHORS)
    assert out.split("\n") == ["## Title", "### Sub", "```bash", "# comment [x](pipeline.ipynb)", "```",
                               "see [p](#w1-pipeline) #tag"]


def test_image_sources_are_re_pointed():
    out = M.rewrite_markdown('<img src="../assets/f.png">', "notebooks/x", "report", ANCHORS)
    assert out == '<img src="../notebooks/assets/f.png">'


def _notebook(*sources: str, ids: tuple[str, ...] = ()) -> nbf.NotebookNode:
    cells = [nbf.v4.new_markdown_cell(s) for s in sources]
    for cell, cid in zip(cells, ids):
        cell["id"] = cid
    return nbf.v4.new_notebook(cells=cells)


def test_merge_keeps_order_adds_banners_and_unique_ids():
    parts = [{"key": "a", "anchor": "w1-a", "path": "report/a.ipynb", "title": "A", "date": "11-09", "builder": "b", "about": "x"},
             {"key": "p", "anchor": "w1-pipeline", "path": "notebooks/mem/pipeline.ipynb", "title": "P", "date": "11-11",
              "builder": "none", "about": "y"}]
    config = {"output": "report/week1.ipynb", "title": "W1", "intro": "intro", "source_commit": "abc123",
              "config_path": "configs/report/week1.yaml", "parts": parts}
    sources = {"report/a.ipynb": _notebook("# A", "[go](../notebooks/mem/pipeline.ipynb)", ids=("same", "x1")),
               "notebooks/mem/pipeline.ipynb": _notebook("# P", "body", ids=("same", "x2"))}
    merged = M.merge(config, lambda path: sources[path])
    texts = [c["source"] for c in merged.cells]
    assert texts[0].startswith("# W1") and "(#w1-a)" in texts[0] and "(#w1-pipeline)" in texts[0]
    assert texts[1].startswith('<a id="w1-a"></a>') and "abc123" in texts[1]
    assert texts[2:4] == ["## A", "[go](#w1-pipeline)"]
    assert texts[4].startswith('<a id="w1-pipeline"></a>') and texts[5:] == ["## P", "body"]
    ids = [c["id"] for c in merged.cells]
    assert len(ids) == len(set(ids)) and "same" in ids and "p-same" in ids
