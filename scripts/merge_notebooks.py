"""Merge executed notebooks into one weekly notebook, keeping their outputs (one notebook per London week).

    python scripts/merge_notebooks.py --config configs/report/week1.yaml [--out <path>]

The config names the output, a cover text and the parts in order. Every part is read as it was in the commit
`source_commit` (git show), so the merge can be repeated after the separate notebooks have been removed. Nothing is
executed: each output is the one saved when that part last ran. For each part:
- a banner cell first: an anchor (<a id="...">), the part's number and title, its original path, date and builder;
- every heading one level lower, so that the banner is the part's top heading (lines inside code fences are left alone);
- every relative link or image of a markdown cell re-pointed: to the part's anchor when it names another merged
  notebook (a #fragment is kept, since headings keep their anchors), otherwise to the same file seen from the output;
- cell ids made unique across the whole notebook (a repeated or missing id gets the part key in front).
"""
import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import nbformat as nbf
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(\]\()([^)\s]+)(\))|((?:src|href)=[\"'])([^\"']+)([\"'])")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)  # URLs, data:, attachment:, anchors, absolute paths
HEADING = re.compile(r"^(#{1,5})(?=\s)")
FENCE = re.compile(r"^\s*(```|~~~)")
ID_CHARS = re.compile(r"[^A-Za-z0-9_-]")
PART_KEYS = ("key", "anchor", "path", "title", "date", "builder", "about")
METADATA = {"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
            "language_info": {"name": "python"}}


def retarget(target: str, source_dir: str, out_dir: str, anchors: Mapping[str, str]) -> str:
    """A link target of a notebook in `source_dir`, as seen from `out_dir` (paths relative to the repository)."""
    if EXTERNAL.match(target):
        return target
    path, _, fragment = target.partition("#")
    resolved = os.path.normpath(os.path.join(source_dir, path))
    if resolved in anchors:
        return "#" + (fragment or anchors[resolved])
    return os.path.relpath(resolved, out_dir) + (f"#{fragment}" if fragment else "")


def rewrite_markdown(text: str, source_dir: str, out_dir: str, anchors: Mapping[str, str]) -> str:
    """Headings one level lower and relative links re-pointed, outside code fences."""
    def link(match: re.Match) -> str:
        if match.group(1):
            return match.group(1) + retarget(match.group(2), source_dir, out_dir, anchors) + match.group(3)
        return match.group(4) + retarget(match.group(5), source_dir, out_dir, anchors) + match.group(6)

    lines, fenced = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            line = LINK.sub(link, HEADING.sub(lambda m: m.group(1) + "#", line))
        lines.append(line)
    return "\n".join(lines)


def banner(number: int, part: Mapping[str, str], commit: str) -> str:
    """The first cell of a part: anchor, title and where it came from."""
    return (f'<a id="{part["anchor"]}"></a>\n\n# 第 {number} 部分 · {part["title"]}\n\n'
            f'<sub>原文件 `{part["path"]}`（{part["date"]}）· 生成方式：{part["builder"]} · 原样并入，'
            f"图和数字是当时运行的结果，没有重跑 · 原文件在提交 `{commit}` 里</sub>")


def cover(config: Mapping, parts: Sequence[Mapping[str, str]]) -> str:
    """The first cell of the merged notebook: title, what it is, table of contents, how to re-run a part."""
    rows = "\n".join(f'| [{i}. {p["title"]}](#{p["anchor"]}) | {p["date"]} | {p["about"]} | `{p["path"]}` |'
                     for i, p in enumerate(parts, 1))
    return (f'# {config["title"]}\n\n{config["intro"].strip()}\n\n| 部分 | 日期 | 内容 | 原文件 |\n| --- | --- | --- | --- |\n'
            f"{rows}\n\n<sub>重跑某一部分：`git show {config['source_commit']}:<原文件> > <原文件>` 取回原文件，"
            "在它原来的文件夹里执行（`bash scripts/run_notebooks.sh <原文件>`），或用它的 builder 重新生成；"
            f"再用 `python scripts/merge_notebooks.py --config {config['config_path']}` 合并。</sub>")


def unique_ids(cells: Sequence, key: str, seen: set[str]) -> None:
    """Give every cell an id not used before in the merged notebook (in place, on the freshly read cells)."""
    for index, cell in enumerate(cells):
        wanted = cell.get("id") or f"{key}-{index}"
        candidate, n = ID_CHARS.sub("-", wanted)[:64], 1
        if candidate in seen:
            candidate = ID_CHARS.sub("-", f"{key}-{wanted}")[:64]
        while candidate in seen:
            n += 1
            candidate = ID_CHARS.sub("-", f"{key}-{wanted}-{n}")[:64]
        cell["id"] = candidate
        seen.add(candidate)


def merge(config: Mapping, read: Callable[[str], nbf.NotebookNode]) -> nbf.NotebookNode:
    """The merged notebook; `read(path)` returns a part's notebook as it was in the source commit."""
    parts, out_dir = config["parts"], os.path.dirname(config["output"])
    anchors = {os.path.normpath(p["path"]): p["anchor"] for p in parts}
    first = nbf.v4.new_markdown_cell(cover(config, parts))
    first["id"] = "cover"
    cells, seen = [first], {"cover"}
    for number, part in enumerate(parts, 1):
        head = nbf.v4.new_markdown_cell(banner(number, part, config["source_commit"]))
        head["id"] = f'{part["key"]}-banner'
        seen.add(head["id"])
        body = read(part["path"]).cells
        unique_ids(body, part["key"], seen)
        for cell in body:
            if cell["cell_type"] == "markdown":
                cell["source"] = rewrite_markdown(cell["source"], os.path.dirname(part["path"]), out_dir, anchors)
        cells += [head, *body]
    notebook = nbf.v4.new_notebook(cells=cells, metadata=dict(METADATA))
    nbf.validate(notebook)
    return notebook


def read_from_git(commit: str) -> Callable[[str], nbf.NotebookNode]:
    """A reader of notebooks as they were in `commit`."""
    def read(path: str) -> nbf.NotebookNode:
        blob = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=REPO_ROOT, capture_output=True, check=False)
        if blob.returncode != 0:
            raise SystemExit(f"cannot read {path} at {commit}: {blob.stderr.decode().strip()}")
        return nbf.reads(blob.stdout.decode("utf-8"), as_version=4)
    return read


def load_config(path: Path) -> dict:
    """The merge config, checked: required keys, one anchor and one path per part."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in ("output", "title", "intro", "source_commit", "parts") if k not in config]
    if missing:
        raise SystemExit(f"{path}: missing {missing}")
    for part in config["parts"]:
        absent = [k for k in PART_KEYS if not part.get(k)]
        if absent:
            raise SystemExit(f"{path}: part {part.get('key', '?')} lacks {absent}")
    for field in ("key", "anchor", "path"):
        values = [p[field] for p in config["parts"]]
        if len(set(values)) != len(values):
            raise SystemExit(f"{path}: repeated part {field}")
    config["config_path"] = str(path)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="merge config, e.g. configs/report/week1.yaml")
    parser.add_argument("--out", type=Path, help="write here instead of the config's output")
    args = parser.parse_args()
    config = load_config(args.config)
    notebook = merge(config, read_from_git(config["source_commit"]))
    out = REPO_ROOT / (args.out or config["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, out)
    print(f"wrote {out.relative_to(REPO_ROOT)}: {len(config['parts'])} parts, {len(notebook.cells)} cells, "
          f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
