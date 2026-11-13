import base64
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_brief.py"
spec = importlib.util.spec_from_file_location("render_brief", SCRIPT)
render_brief = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render_brief)

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def test_render_embeds_figures_and_keeps_tables(tmp_path: Path) -> None:
    (tmp_path / "figs").mkdir()
    (tmp_path / "figs" / "flow.png").write_bytes(PNG)
    brief = tmp_path / "R1.md"
    brief.write_text("# 第 1 轮：标题\n\n![流程图](figs/flow.png)\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n", encoding="utf-8")

    page = render_brief.render(brief)

    assert "<title>第 1 轮：标题</title>" in page
    assert 'src="data:image/png;base64,' in page and "figs/flow.png" not in page
    assert "<table>" in page and "<td>2</td>" in page


def test_render_refuses_a_missing_figure(tmp_path: Path) -> None:
    brief = tmp_path / "R1.md"
    brief.write_text("# t\n\n![x](nowhere.png)\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="nowhere.png"):
        render_brief.render(brief)
