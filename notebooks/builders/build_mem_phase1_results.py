"""Generate notebooks/mem/phase1_results.ipynb: the results of phase 1 (where should a memory put its tokens?)."""

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


md(r"""
# 阶段 1 结果：记忆的 token 应该写在哪里
**配套文档：** [讨论稿](../../docs/mem/mem分支_问题定义与讨论稿_2026-11-10.md)第 21 节（实验规格，跑之前写定）· [pipeline.ipynb](pipeline.ipynb)（各环节的想法）· [pipeline_examples.ipynb](pipeline_examples.ipynb)（每个环节的一个示例）· SceneFun3D 验证集的 30 个场景 · 2026-11-12

**这次实验回答的问题（预测 A）：** 记忆的预算相同时，由 affordance 决定位置的 token，是否比按其他方式取的 token 更常把正确的证据检索出来。

**读之前请注意**

- **affordance map 是 ORACLE**（标注投影得到，不带部件编号）。所以这里所有数字都是**上限**：它们说明"如果 affordance map 是对的，按它来写记忆值不值得"，不说明现在的模型能做到多少。换成模型预测的 affordance map 是阶段 2 的事。
- **没有训练任何东西。** 图文编码器（SigLIP 2）、通用分割模型（SAM）、视觉语言模型（Qwen3-VL-4B）、单目深度模型（Depth Anything V2）都是现成的公开模型。
- **判定标准在跑实验之前写定**（讨论稿第 21.6 节），本文第 10 节按原样套用，没有因为结果而修改。
- **图里的英文标签：** P0 whole keyframes = 只有整帧；P1 uniform grid = 均匀网格；P2 random places = 随机位置；P3 generic regions = 通用区域；P4 affordance regions = affordance 区域。每种写法在所有图里用同一种颜色。

<sub>重新运行：`python scripts/download_scenefun3d_keyframes.py` 取 30 个场景的关键帧；`bash scripts/remote/queue_phase1.sh` 把每个场景的作业放进 GPU 队列（结果写到 `runs/mem_phase1/`）；`python scripts/aggregate_mem_phase1.py` 汇总；然后 Run All（CPU，几分钟，第 3 节和第 8 节会在一个场景上重新算一遍检索）。</sub>
""")

code(r"""
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from matplotlib_inline.backend_inline import set_matplotlib_formats

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")  # no model-loading chatter in the outputs
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance3d").is_dir())
_saved_fd, _null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
os.dup2(_null, 1)  # silence Open3D's import banner (printed from C++)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        import open3d  # noqa: F401
finally:
    os.dup2(_saved_fd, 1)
sys.path.insert(0, str(ROOT))
from affordance3d.mem import keyframes as K
from affordance3d.mem import phase1_stats as A
from affordance3d.mem import pipeline as P
from affordance3d.mem import search as S
from affordance3d.mem import tokens as T
from affordance3d.mem.encoders import ImageTextEncoder
from report import mem_phase1_helpers as R
from report import mem_steps_helpers as H
from report.mem_demo_helpers import scaled_mono_depths

from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()
plt.rcParams.update({"figure.dpi": 100})
pd.set_option("display.max_colwidth", 120)
torch.set_num_threads(48)  # shared server: leave most cores to others
DATA, RUNS = ROOT / "data" / "scenefun3d", ROOT / "runs" / "mem_phase1"
MONO_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
results, summary = R.load_runs(RUNS)  # written by scripts/run_mem_phase1.py and scripts/aggregate_mem_phase1.py
print(f"{summary['scenes']} scenes, {summary['tasks']} answerable tasks")
""", hidden=True)

md(r"""
## 1. 比较的是什么

**记忆的形式（L2）：** 每个关键帧存一张图，再加若干个 token。一个 token 对应关键帧里的一个小区域，存的是这个区域周围一块画面的图文特征。查询来了以后，用查询句子的特征去和整帧、和 token 分别比较，两个分数相加，给关键帧排序。

**这次只改变一件事：token 取在哪里。** 编码器、裁剪规则、打分公式、查询句子全部相同。

| 编号 | 写法 | token 取在哪里 |
| --- | --- | --- |
| P0 | 只有整帧 | 没有 token。没有 affordance 的记忆只能这样检索 |
| P1 | 均匀网格 | 每个关键帧切成固定的格子，每格一个 token |
| P2 | 随机位置 | 个数和大小与 P4 相同，位置随机。只改变"在哪里" |
| P3 | 通用区域 | 不知道 affordance 的分割模型（SAM）给出的区域，按大小与 P4 匹配后随机取 |
| P4 | affordance 区域 | affordance map（ORACLE）上的每一个连通区域 |

- **预算：** 以一个场景里 P4 的 token 数为 1 倍。主要比较在 1 倍上做，另外在 0.25 到 8 倍之间画曲线。
- **主要指标：** 排在第 1 位的关键帧里有没有任务所指的部件（下面叫"第一个关键帧对"），以及前 3 位里有没有。这个指标不依赖 token 的形状，对各种写法公平。
- **统计：** 同一个场景的任务共用一份记忆，所以按场景做自助抽样，给出 95% 置信区间；两种写法的差在同一批任务上成对比较。P2、P3 各用 5 个随机种子，取平均。
""")

md(r"""
## 2. 数据

- **场景：** SceneFun3D 的 30 个验证场景（Fun3DU 和 UniFunc3D 用的 split0），每个场景用文件清单指定的一段视频。
- **关键帧：** 每段视频 40 帧。把视频按时间分成 40 段，每段取设备位姿转得最慢的一帧（只用输入）。
- **"可回答"的任务：** 任务所指的部件至少在一个关键帧里占 10 个像素。看不到的部件，任何写法都检索不到，所以主要数字只在可回答的任务上统计。

第二张表是关键帧处相机转动的快慢（由设备位姿算出）。相机转得快的帧里，标注和画面会错位，所以选转得慢的帧；"均匀取帧"一行是对照。
""")

code(r"""
display(R.data_table(results, summary))
no_lidar = summary.get("keyframes_without_lidar", {})
print("keyframes whose LiDAR frame has no reliable pixel (depth scale taken from the other keyframes of the scene):",
      {visit: len(frames) for visit, frames in no_lidar.items()} or "none")
display(R.turn_rate_table(DATA, results))
""")

md(r"""
## 3. 五种写法在画面上的样子

下面用一个场景的三个关键帧，画出预算为 1 倍时每种写法的 token 落在哪里。**这个场景不是挑出来的**：取的是"P4 与 P3 的差"在所有场景里居中的那一个（至少有 10 条可回答的任务）。

**怎么看图：** 每一行是同一个关键帧，每一列是一种写法，每个框是一个 token。三行分别是 affordance 区域最多、居中、最少的关键帧。P4 把 token 集中在有可操作部件的画面上；P1 平均铺开；P2、P3 的 token 总数与 P4 相同，但分散在所有关键帧里。
""")

code(r"""
EXAMPLE = R.typical_scene(results)
example = next(result for result in results if result["visit"] == EXAMPLE)
scene = P.load_scene(DATA, EXAMPLE, example["video"])
times = K.scene_keyframes(scene, example["keyframes"], example["n_keyframes"])
mono = scaled_mono_depths(scene, times, MONO_ID, RUNS / "mono" / example["video"])  # cached by the GPU run
frames = [P.input_frame(scene, t, d) for t, d in zip(times, mono)]
labels = P.label_maps(scene, P.gt_frames(scene, times))  # ORACLE with part ids: scoring and drawing only
actions = [T.ACTIONS.index(e.label) if e.label in T.ACTIONS else T.NO_ACTION for e in scene.elements]
action_maps = [T.action_map(label, actions) for label in labels]  # the ORACLE affordance map: action types, no part ids
pool = R.generic_pool(RUNS / "regions" / f"{EXAMPLE}.json", frames[0].image_hw)
sets = R.token_sets(frames, action_maps, pool)  # the same random draws as the GPU run (seed 0)

per_keyframe = np.bincount([c.keyframe for c in sets[A.P4]], minlength=len(frames))
by_count = np.argsort(per_keyframe, kind="stable")
shown = [int(by_count[-1]), int(by_count[len(by_count) // 2]), int(by_count[0])]
empty = {R.POLICY_NAMES[name]: len(frames) - len({c.keyframe for c in sets[name]}) for name in (A.P2, A.P3, A.P4)}
print(f"scene {EXAMPLE}: {len(sets[A.P4])} affordance regions in {len(frames)} keyframes")
print("keyframes without any token:", empty)
set_matplotlib_formats("jpeg")
R.policies_figure(frames, sets, shown)
plt.show()
""")

md(r"""
## 4. 主要结果：预算为 1 倍时

**怎么看图：** 左边是每种写法"第一个关键帧对"的比例，黑线是 95% 置信区间。右边是 P4 减去其他写法的差（同一批任务成对比较），横线是差值的 95% 置信区间；**区间不跨过 0，才算有差别**。
""")

code(r"""
set_matplotlib_formats("png")
R.headline_figure(summary, "first_keyframe_right", "the first keyframe shows the part")
plt.show()
display(R.rates_table(summary["first_keyframe_right"]))
""")

md(r"""
同样的比较，换成"前 3 个关键帧里有没有"（之后交给视觉语言模型看的就是前 3 个关键帧）：
""")

code(r"""
R.headline_figure(summary, "right_keyframe_in_top3", "one of the first three keyframes shows the part")
plt.show()
display(R.rates_table(summary["right_keyframe_in_top3"]))
""")

md(r"""
## 5. 预算曲线：多给 token 能不能追上

**怎么看图：** 横轴是记忆的预算（token 数相对于 P4 的倍数，对数刻度），纵轴是"第一个关键帧对"的比例，色带是 95% 置信区间。虚线是只有整帧（P0）的水平。P4 最多只有 1 倍（affordance 区域就这么多），其他写法一直给到 8 倍。P1 的格子数是固定的几档，所以它的点不落在整数倍上。
""")

code(r"""
R.budget_figure(results, "hit1", "the first keyframe shows the part")
plt.show()
curves = pd.DataFrame([point for name in (A.P4, A.P3, A.P2, A.P1) for point in summary["curves"][name]])
curves[["rate", "low", "high"]] = (100 * curves[["rate", "low", "high"]]).round(1)
display(curves.round({"budget": 2, "tokens": 0}).set_index(["policy", "budget"]))
""")

md(r"""
**多给预算以后再比一次。** 这一项不属于预先写定的判定规则，是为了排除一个可能的偏差而加的检查。

预算为 1 倍时，P2、P3 的 token 是散开取的，会有一些关键帧一个 token 也没有（第 3 节的输出里有那个场景的数目）；而打分公式给没有 token 的关键帧一个固定的低分。对 P4，"这一帧没有可操作的部件"是有用的信息；对 P2、P3，哪一帧没有 token 只是运气。预算到 4 至 8 倍时，几乎每个关键帧都有 token，这个因素就消失了。

下表是 **P4 在 1 倍预算下的结果，减去其他写法在更大预算下的结果**（同一批任务成对比较，单位是百分点）。
""")

code(r"""
display(R.rates_table(summary["affordance_at_one_against_more"]))
""")

md(r"""
**token 比只有整帧多带来什么？** 同样是规格之外的补充：各种写法减去 P0（成对比较，百分点）。最后两行是 P4 只留四分之一和一半的 token，可以看作 affordance map 漏掉一部分部件时的情形。再下面是 P2、P3 在 1 倍预算下各个随机种子的最低和最高结果。
""")

code(r"""
display(R.rates_table(summary["others_against_whole"]))
spread = pd.DataFrame(summary["seed_spread"]).T
spread[["lowest", "highest"]] = (100 * spread[["lowest", "highest"]]).round(1)
display(spread)
""")

md(r"""
## 6. 逐个场景看

总体的差可能来自少数几个场景。下面把每个场景里"P4 减去另一种写法"的差从大到小排开。

**怎么看图：** 每根柱是一个场景。柱子朝上（P4 的颜色）表示这个场景里 P4 更好，朝下（另一种写法的颜色）表示另一种写法更好。每个场景的任务数从 3 条到 27 条不等，任务少的场景柱子会很长（P4 − P0 一格里最朝下的那一根，是只有 3 条任务的场景）。
""")

code(r"""
R.per_scene_figure(results)
plt.show()
""")

md(r"""
## 7. 得分最高的 token 是否落在目标部件上

上面的指标只看关键帧对不对。这里再看一步：对查询得分最高的那个 token，它的区域是否盖住了任务所指的部件。同时给出区域的面积，避免"框大所以容易中"。
""")

code(r"""
display(R.token_table(results))
""")

md(r"""
## 8. 端到端：把前 3 个关键帧交给视觉语言模型

检索之后，把得分最高的 3 个关键帧连同 token 的编号框交给视觉语言模型，让它选一个框。选中的框盖住任务所指的部件，算对。只对 P4 和 P3 做。

**怎么看图：** 与第 4 节的画法相同。注意这一步的绝对数字不高：模型看的是整幅关键帧，小部件在图里很小（[pipeline.ipynb](pipeline.ipynb) 环节 10 里记着这个缺口）。
""")

code(r"""
R.reading_figure(summary)
plt.show()
display(R.rates_table({name: row for name, row in summary["reading"].items()}))
shown_marks = {name: float(np.mean([np.mean(result["reading"][name]["marks_shown"]) for result in results]))
               for name in R.READING_NAMES.values() if name in summary["reading"]}
print("numbered boxes shown to the model per task, mean:", {name: round(value, 1) for name, value in shown_marks.items()})
""")

md(r"""
### 几个具体的任务

还是第 3 节那个场景。在 CPU 上重新算一遍它的检索（与 GPU 上的结果逐条核对），然后按固定的规则取任务：先取"P4 对而 P3 错"的，再取"P4 对而 P2 错"的，再取"P4 错"的；没有就跳过。

**怎么看图：** 每一行是一条任务，每一列是一种写法排在第 1 位的关键帧。彩色框是这个关键帧里得分最高的 token；绿色圆圈是任务所指的部件（只有关键帧对的时候才有）。画面的边框绿色表示关键帧对，红色表示错。
""")

code(r"""
encoder = ImageTextEncoder()
keyframe_vectors = encoder.images([frame.rgb for frame in frames])
vectors = {name: encoder.images(R.crops_of(frames, sets[name])) for name in (A.P2, A.P3, A.P4)}
tasks = [task for task in H.tasks_of(scene, labels) if task.keyframes]
start = time.time()
queries = encoder.texts([task.text for task in tasks])
text_ms = 1000 * (time.time() - start) / len(tasks)
no_tokens = np.zeros((0, keyframe_vectors.shape[1]))
start = time.time()
picks = [{A.P0: R.first_pick(query, keyframe_vectors, [], no_tokens, task.keyframes),
          **{name: R.first_pick(query, keyframe_vectors, sets[name], vectors[name], task.keyframes) for name in (A.P2, A.P3, A.P4)}}
         for query, task in zip(queries, tasks)]
search_ms = 1000 * (time.time() - start) / (4 * len(tasks))

agreement = {}
for name in (A.P0, A.P2, A.P3, A.P4):  # the same tasks, the same token places: the CPU should repeat the GPU run
    stored = next(run for run in example["search"] if run["policy"].startswith(name) and run["seed"] == 0
                  and (name == A.P0 or run["budget"] == 1.0))["hit1"]
    agreement[R.POLICY_NAMES[name]] = f"{sum(int(pick[name].right) == hit for pick, hit in zip(picks, stored))} of {len(tasks)}"
print("same outcome as the GPU run:", agreement)
print(f"ranking the keyframes for one query: {search_ms:.1f} ms on the CPU ({len(frames)} keyframes, {len(sets[A.P4])} tokens); "
      f"turning the query into a vector: {text_ms:.0f} ms on the CPU")

chosen = R.choose_examples(picks, n=3)
set_matplotlib_formats("jpeg")
R.picks_figure(frames, [tasks[i].text for i in chosen], [picks[i] for i in chosen],
               [R.target_pixels(labels, tasks[i].parts) for i in chosen])
plt.show()
""")

md(r"""
## 9. 几何和代价：要不要先重建

L2 记忆不存接触点，查询时从证据关键帧现算。对照做法是先把 40 个关键帧的深度融合成一份点云，再从点云上读。两种读法用同样的关键帧、同样的深度和位姿、同样的 ORACLE 标注；为了只比较读法，这张表按真实的部件编号统计。

**怎么看表：** "误差"是读出的接触点到扫描里这个部件的距离。"3 厘米以内"是能直接去碰的精度。第二行是完整的 L2 读法（部件由记忆自己跨帧关联，没有用部件编号），放在这里作参考。这里的"融合"是把同样的 40 张单目深度图投到一起、按体素去重，没有做多视角一致性检查，也不是 TSDF；所以它只说明"简单融合同样的深度图"并不更准。第二张表是在两种读法都读得出的部件上成对比较。最后一张表里的写入时间不含选关键帧、估计单目深度和预测 affordance map 的时间。
""")

code(r"""
display(R.geometry_table(results, summary))
paired = pd.DataFrame(summary["geometry_paired"]).T
paired["single_frame_closer"] = (100 * paired["single_frame_closer"]).round(1)
display(paired.round(2).rename(columns={"parts": "parts read both ways", "median_difference_cm": "fused minus single keyframe, median (cm)",
                                        "low": "95% interval, low", "high": "95% interval, high",
                                        "single_frame_closer": "single keyframe closer (% of parts)",
                                        "read_ms_per_part": "read time per part (ms)"}))
display(R.cost_table(summary))
""")

md(r"""
## 10. 关口 G1 的判定

判定标准原文（讨论稿第 21.6 节，跑实验之前写定）。预算为 1 倍时：

- **通过：** P4 的"第一个关键帧对"高于 P3，差值的 95% 置信区间不包含 0；并且端到端选对的比例 P4 不低于 P3。
- **部分通过：** P4 只高于 P2（随机位置），与 P3 没有显著差别。
- **不通过：** P4 不高于 P2。
""")

code(r"""
first, reading = summary["first_keyframe_right"], summary["reading"]
verdict = pd.DataFrame({
    "P4 minus P3, first keyframe right (points)": first["P4 minus P3"], "P4 minus P2, first keyframe right (points)": first["P4 minus P2"],
    "model picks the right part, P4 (%)": reading[R.READING_NAMES[A.P4]], "model picks the right part, P3 (%)": reading[R.READING_NAMES[A.P3]],
}).T[["rate", "low", "high"]]
display((100 * verdict).round(1).rename(columns={"rate": "value", "low": "95% interval, low", "high": "95% interval, high"}))
print("G1:", summary["g1"])
""")

md(r"""
## 11. 怎么读这些结果

> **审稿后的说明（2026-11-12）：** 独立审稿指出，下面"结果说明了什么"的第 1 至 3 条把增益归因于"token 取在哪里"，证据还不够：打分公式给没有 token 的关键帧固定的低分，P4 可能只是受益于"知道哪些关键帧里有可操作的部件"。补充检查的计划和结果在讨论稿第 23 节；在那之前，这三条是暂定的。

**按预先写定的规则，G1 的条件都满足：** P4 比 P3 高，差值的区间不含 0；端到端选对的比例 P4 不低于 P3；P4 也高于 P2。**关口是否通过，由项目负责人决定。**

**结果说明了什么**（都是 ORACLE 的上限）

1. **预算相同时，由 affordance 决定位置的 token 明显更好。** "第一个关键帧对"的比例：P4 66.2%，P3 47.3%，P0 45.6%，P2 38.5%。P4 比 P3 高 18.8 个百分点（区间 12.9 到 24.9）。
2. **多给预算追不上。** P3 给到 8 倍的 token 是 51.3%，仍低于 P4 的 1 倍；P4 只留四分之一的 token（52.2%），就与 P3 的 4 到 8 倍相当。P4 在 1 倍预算下比 P3 在 8 倍预算下高 14.9 个百分点（区间 9.2 到 21.3）。
3. **不按 affordance 取的 token，比只有整帧多不出什么。** 1 倍预算时，通用区域比只有整帧高 1.7 个百分点（区间 −3.5 到 +7.5，看不出差别）；随机位置比只有整帧低 7.1 个百分点（区间 −11.7 到 −2.3）。
4. **端到端的差距更大。** 视觉语言模型选对部件的比例：P4 53.6%，P3 10.3%。
5. **先重建没有让接触点更准。** 从单个关键帧现算，中位误差 4.1 厘米；先融合成 1 厘米体素的点云再读，是 5.4 厘米，每个场景还要多存约 22.5 MB。

**结果没有说明什么**

- **这是最有利于 affordance 的设定。** affordance map 是标注投影来的，而 SceneFun3D 的任务只问标注过的功能部件。换成模型预测的 affordance map，会漏、会错，增益会缩小；缩小多少，要到阶段 2 才知道。
- **没有测"普通物体"的查询。** affordance token 对这类查询会不会变差，这次回答不了。
- **P3 只是通用区域的一种取法。** 这里用的是 SAM 的分割结果，按大小与 P4 匹配后随机取。其他取法（例如开放词表检测器给出的区域）没有比。
- **端到端里 P3 的数字主要受候选框限制。** 它的框大多不在可操作的部件上，模型没有对的可选。这是"两种记忆交给模型的候选"的比较，不是模型读图能力的比较。
- **接触精度仍然不够。** 两种读法落在 3 厘米以内的都不到四成。
- **没有成立的一项：** 选相机转得慢的帧，关键帧处的转速中位数从每秒 21.0 度降到 8.1 度，但看得到的部件没有变多，反而略少（561 个部件里 481 个；均匀取帧是 501 个）。

**与规格（讨论稿第 21 节）不同的地方**

- 6 个场景里共有 7 个关键帧，数据集给的 LiDAR 置信度整张为 0，无法按规格用 LiDAR 校正单目深度的尺度。这些帧改用同一场景其余关键帧的尺度中位数（只用输入）。
- 1 个场景里有一个 affordance 区域大到填满了算法线用的窗口，周围没有可用的像素。这种情况改用区域自身的表面算法线；它只影响站立点，不影响接触点的误差。
- 第 5 节后面的两张表（更大预算的比较、与只有整帧的比较）是规格之外加的检查，不参与判定。
- 预算曲线里，P1 较细的网格在 token 少的场景里超过了 8 倍预算的上限，没有跑；图上只画了 30 个场景都有的两档，全部档位在表里（最后一列是场景数）。

完整的记录在[讨论稿](../../docs/mem/mem分支_问题定义与讨论稿_2026-11-10.md)第 22 节。
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}
notebook.metadata["language_info"] = {"name": "python"}
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUT_PATH)
print("wrote", OUT_PATH, "with", len(cells), "cells")
