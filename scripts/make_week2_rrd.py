"""Interactive 3D file of the week-2 memory (report/week2_rrd.py): the orbit video's content in its game style as a Rerun
.rrd file, to open in the Rerun viewer and turn freely (week-2 notebook section 1.2 says how).

    python scripts/make_week2_rrd.py --config configs/report/week2_rrd.yaml

Needs rerun-sdk (requirements-rerun.txt), the finished one-scene part-vector run named by the config's run_config and
the scene's scan and video index. CPU only. Writes the config's .rrd (kept in git under report/assets/) and, in
<outputs>/<name>/, the resolved config, the stages log and rrd.json (what was logged, file size).
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401,E402  (its banner would end up in the log)

import run_part_vectors as R  # noqa: E402

from affordance3d.experiment import load_config, log_stage, run_folder  # noqa: E402
from affordance3d.mem import pipeline as P  # noqa: E402
from report import week2_interaction as G  # noqa: E402
from report import week2_media as M  # noqa: E402
from report import week2_rrd as W  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.sub=value override")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, "rrd", REPO_ROOT)
    run = load_config(REPO_ROOT / config["run_config"])
    scene = P.load_scene(REPO_ROOT / run["data_root"], str(run["visit"]), str(run["video"]))
    run_dir = REPO_ROOT / run["outputs"] / run["name"]
    rows = json.loads((run_dir / "parts.json").read_text())
    truths = {p: R.truth_of(scene, p) for p in {row["part"] for row in rows}}
    parts = M.load_parts(run_dir, config["variant"], truths)
    style = G.Style.from_config(load_config(REPO_ROOT / config["style_config"])["walkthrough"]["interaction"])
    stats = W.export(scene, parts, style, config, REPO_ROOT / config["out"], f"memory of scene {run['visit']}")
    stats["parts"] = len(parts)
    (folder / "rrd.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if not k.startswith("world/parts/")}, indent=1))
    print(f"wrote {config['out']} ({stats['bytes'] / 1e6:.1f} MB) and {folder}")


if __name__ == "__main__":
    main()
