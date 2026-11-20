"""Videos of the week-2 report (report/week2_2026-11-16_to_11-22.ipynb): the part-vector memory of one scene drawn on
its whole walkthrough, and an orbit around its laser scan with every part's vector (report/week2_media.py).

    python scripts/make_week2_media.py --config configs/report/week2_media.yaml [--only walkthrough|orbit]

Needs a finished one-scene part-vector run (the config's `run`, e.g. configs/part_vectors/demo_421069_gravity.yaml)
and the scene's whole video with the dataset's registered cameras. CPU only. Writes the config's mp4 files (kept in
git under report/assets/) and, in <outputs>/<name>/, the resolved config, the stages log and media.json (what each
video shows: frames, parts in view).
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
from report import week2_media as M  # noqa: E402

LEGEND = "orange: memory (rules)   green: annotation   line: axis   arrow: move"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", choices=("walkthrough", "orbit"))
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.sub=value override")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    folder = run_folder(config, REPO_ROOT)
    log_stage(folder, args.only or "all", REPO_ROOT)
    run = load_config(REPO_ROOT / config["run_config"])
    scene = P.load_scene(REPO_ROOT / run["data_root"], str(run["visit"]), str(run["video"]))
    run_dir = REPO_ROOT / run["outputs"] / run["name"]
    rows = json.loads((run_dir / "parts.json").read_text())
    truths = {p: R.truth_of(scene, p) for p in {row["part"] for row in rows}}
    parts = M.load_parts(run_dir, config["variant"], truths)
    media_path = folder / "media.json"
    media = json.loads(media_path.read_text()) if media_path.exists() else {}
    media["parts"] = len(parts)
    if args.only in (None, "walkthrough"):
        w = config["walkthrough"]
        media["walkthrough"] = M.render_walkthrough(scene, parts, REPO_ROOT / w["out"], w["width"], w["fps"], w["tolerance_m"],
                                                    w["length_m"], LEGEND)
        print("walkthrough", media["walkthrough"], flush=True)
    if args.only in (None, "orbit"):
        o = config["orbit"]
        media["orbit"] = M.render_orbit(scene, parts, REPO_ROOT / o["out"], tuple(o["hw"]), o["frames"], o["fps"], o["points"],
                                        o["cut_above_floor_m"], o["elevation_deg"], o["radius_scale"], o["fov_deg"], o["length_m"], LEGEND)
        print("orbit", media["orbit"], flush=True)
    media_path.write_text(json.dumps(media, indent=1))
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
