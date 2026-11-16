import json
from pathlib import Path

import pytest

from affordance3d import experiment as E


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "exp.yaml"
    path.write_text("name: demo\noutputs: out\ncompare:\n  seed: 0\n  margins: {AP25: [5.0, 10.0]}\n")
    return path


def test_load_config_applies_overrides_and_records_them(tmp_path: Path) -> None:
    config = E.load_config(_config(tmp_path), ["compare.seed=3", "compare.margins={AP25: [4, 8]}"])

    assert config["compare"]["seed"] == 3 and config["compare"]["margins"] == {"AP25": [4, 8]}
    assert config["_source"]["overrides"] == ["compare.seed=3", "compare.margins={AP25: [4, 8]}"]


@pytest.mark.parametrize("override", ["compare.sed=3", "nothing.seed=1", "compare.seed"])
def test_an_unknown_or_malformed_override_stops(tmp_path: Path, override: str) -> None:
    with pytest.raises(SystemExit):
        E.load_config(_config(tmp_path), [override])


def test_run_folder_records_the_config_and_refuses_a_different_one(tmp_path: Path) -> None:
    config = E.load_config(_config(tmp_path))

    folder = E.run_folder(config, tmp_path)

    assert folder == tmp_path / "out" / "demo"
    assert json.loads((folder / E.RESOLVED).read_text())["config"] == config
    assert E.run_folder(config, tmp_path) == folder  # the same experiment may continue in its folder
    with pytest.raises(SystemExit):
        E.run_folder(E.load_config(_config(tmp_path), ["compare.seed=1"]), tmp_path)
