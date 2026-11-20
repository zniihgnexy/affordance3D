# Cells of build_report_2026_11_20_week2.py, part 1 (overview and demos): run in order by that builder, with md() and code() in scope.

md(r"""
# 第 2 周：2026-11-16（周一）到 11-22（周日）
**视频功能记忆（部件级，记忆方向）** · 写给项目负责人 · 伦敦时间的自然周，一周只有这一份 notebook，周内持续更新 · 最近更新 2026-11-20

先看视频和图（第 1 节），再看 pipeline 图和逐块讲解（第 2 节），然后是本周的结果（第 3 节）、主旨核对和要你决定的事（第 4 节），接着是先存起来的未来方向和新读的论文（第 5 节），然后是阶段 0 五个测试的结果、重新分析和推荐（第 6 节），然后是你对决定 1 到 6 的答复和"一个场景一张图"的讨论（第 7 节），最后是阶段 1 的框架冻结稿和高斯试拟合（第 8 节，等你批准）。所有数字都从 `runs/` 里的结果文件读出来，附录里逐个核对。上一周的全部内容在 [第 1 周 notebook](week1_2026-11-09_to_11-15.ipynb)。
""")

md(r"""
## 先读这里：用大白话讲这一周

**这一周最大的变化：记忆里存什么变了。**
- 上周的记忆按"能对什么做什么"挑画面，帮模型找到要操作的部件。
- 这周按你 11-18 的要求，记忆给每个部件存一条**看得懂的三维向量**：手碰哪里、往哪个方向推拉、绕哪根轴转、轴在哪里。
- 这些量都在世界坐标里。机器人拿到以后，乘一个坐标变换就能用。

**这一周做了什么（按时间）：**
1. **11-16 到 11-17：** 把上周的方法放到公开评测上（SceneFun3D Task 2），和 UniFunc3D 8B 比；代码按你的要求整理成"一个配置、一条命令"。
2. **11-18：** 你逐块看了一遍 pipeline，差异记进《理解差异记录》。我调研了 29 篇相关工作，推荐了新 pipeline。
3. **11-18 到 11-19：** 服务器连不上，在 Mac 上做了两个只用 CPU 的小检验（S0、S0b）：有了部件掩码以后，规则加深度能不能算出三维向量。
4. **11-19 到 11-20：** 服务器恢复。做了事先写好预测的检验 S0c（门轴取重力方向），四条预测都成立。又对照 VLA 方向的 affordance map，加了抓握轴等字段。公开评测的最后两步（投到扫描上、打分）也跑完了。
5. **11-20：** 你提出把 affordance map 做成三维表面图，用来帮忙搜索目标（决定 6）；横向对比的候选卖点先存为未来方向，读了 4 篇 2026 年的新工作（第 5 节）；notebook 改成一周一份。
6. **11-20：** 你同意了实验推进计划；阶段 0 的五个零成本测试（只用 CPU）跑完，据此重新分析了框架的几个决定（第 6 节）。
7. **11-20：** 你答复了决定 1 到 6；决定 3、6 你提出"一个场景一张特征图、新的覆盖旧的"。我查了相关工作，用已有结果模拟了"新的覆盖旧的"，给出两层图的建议（第 7 节）。
8. **11-20：** 你定了有条件覆盖、下载其余 52 段视频、底层隐性加上层可解释；表示问"体素和高斯能不能共存"。我查了高斯方面的工作，建议高斯做主体、体素版做对照（7.11）。
9. **11-20：** 你选了高斯做主体、逐场景拟合算重建、现在试拟合。装好了高斯工具，在 3 号卡上试拟合了两个场景，画了 v4 流程图，写了框架冻结稿（第 8 节）。

**结果三句话：**
- **按键、插座这类原地动的部件：** 规则算出的向量，一半到八成的轴是对的。
- **门窗：** 轴的方向用重力就修好了（给对类型时，68% → 84%）；门轴在哪里（原点）还做不出来，要先把门板分出来。
- **门还是抽屉：** 只看标签最多对 80%；靠几何猜，在抽屉多的房间里反而更差。要问模型"这个拉手装在什么上"。
- **公开评测（Task 2，445 条任务）：** 用 SAM 3 自己建部件图时，明显落后于 UniFunc3D 8B（AP25 25 对 44）；换成标注的部件图就差不多（43 对 44）。差距主要在部件图。按事先写定的读法，这一条要停下来和你讨论（第 4 节决定 5）。

**要记住的前提：** 这一周的向量检验都用标注的部件掩码（ORACLE）。它们只回答"有了部件以后，向量算得对不对"，不回答"部件能不能自己找到"。部件由 SAM 3 自己找，上周的结果是找到 74.8%。
""")

code(r"""
# setup: paths, the scene of the demos (421069) with its 40 keyframes, the demo run's vectors and the annotated motions
import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "affordance3d").is_dir())  # the repository (run from inside it)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
with contextlib.redirect_stdout(io.StringIO()):
    import open3d  # noqa: F401  (its banner would end up in the notebook)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Video, display

import count_label_motions as CLM
import run_part_vectors as RPV
from affordance3d.experiment import load_config
from report import week2_helpers as H

get_ipython().run_line_magic("matplotlib", "inline")  # run_part_vectors switches matplotlib to Agg when imported

pd.set_option("display.max_colwidth", 60)
FIG = ROOT / "docs" / "mem" / "figures"
ASSETS = ROOT / "report" / "assets"
DEMO = load_config(ROOT / "configs" / "part_vectors" / "demo_421069_gravity.yaml")
scene, times, frames, labels = RPV.load_inputs(DEMO, str(DEMO["visit"]), str(DEMO["video"]))
demo_rows = H.run_json("demo_421069_gravity", "parts.json")
truths = {p: RPV.truth_of(scene, p) for p in {r["part"] for r in demo_rows}}
media = json.loads((ROOT / "runs" / "report_media" / "week2_media_game" / "media.json").read_text())


def check(name, got, want, tol=0.05):
    # every number the text quotes is checked against the result files (appendix B)
    assert abs(got - want) <= tol, f"{name}: text says {want}, files say {got}"
    return name, want

print(f"scene 421069: {len(times)} keyframes, {len(truths)} parts with a vector; walkthrough {media['walkthrough']['frames']} frames")
""")

code(r"""
# the week's path, one box per step (done / tested / not built)
import textwrap

from matplotlib.patches import FancyBboxPatch

STEPS = [("11-16/20  Public Task 2 vs UniFunc3D 8B: behind with SAM 3 parts, level with annotated", "tested"),
         ("11-18  Block-by-block walk-through; 29 papers; new pipeline: 3D vector per part", "done"),
         ("11-18/19  S0, S0b on the Mac: rules + depth give 3D vectors?", "done"),
         ("11-19/20  S0c, fixed in advance: door axis = gravity. 4 of 4 predictions hold", "tested"),
         ("11-20  Ideas from the VLA map: grasp axis, robot plan, new metrics", "done"),
         ("Next  SAM 3 door masks + VLM 'mounted on what?' (needs your OK)", "next")]
STYLE = {"done": ("#d7f0d2", "#2e7d32"), "tested": ("#d5e6fb", "#1f5fae"), "next": ("#f3f3f3", "#6b6b6b")}
WIDTH, GAP, HEIGHT = 2.6, 0.4, 1.9
fig, axis = plt.subplots(figsize=(19, 3.0))
for i, (label, status) in enumerate(STEPS):
    x = i * (WIDTH + GAP)
    fill, edge = STYLE[status]
    axis.add_patch(FancyBboxPatch((x, 0), WIDTH, HEIGHT, boxstyle="round,pad=0.05", fc=fill, ec=edge, lw=2.2,
                                  ls="--" if status == "next" else "-"))
    axis.text(x + WIDTH / 2, HEIGHT / 2, textwrap.fill(label, 21), ha="center", va="center", fontsize=11.5, linespacing=1.25)
    if i < len(STEPS) - 1:
        axis.annotate("", xy=(x + WIDTH + GAP - 0.04, HEIGHT / 2), xytext=(x + WIDTH + 0.04, HEIGHT / 2),
                      arrowprops={"arrowstyle": "-|>", "color": "#52514e", "lw": 1.8})
axis.set_xlim(-0.15, len(STEPS) * (WIDTH + GAP) - GAP + 0.15)
axis.set_ylim(-0.15, HEIGHT + 0.15)
axis.axis("off")
fig.tight_layout()
display(fig)
plt.close(fig)
""")

md(r"""
---
# 1. 先看演示

## 1.1 视频一：记忆贴在整段视频上（场景 421069，厨房和客厅）

**这是什么：** 手机在房子里走一圈的原始视频，一共 1025 帧。上面画的是记忆里每个部件"在哪、怎么动"，做成游戏里交互提示的样子（11-20 按你的要求重画）：
- **彩色高亮：** 部件本身。标注的三维点按这一帧的深度画上去，被挡住的部分不画。颜色表示动作类别：蓝色是拉，橙色是推或按，紫色是转，绿色是开门窗（绕离部件较远的轴转）。
- **来回动的箭头：** 记忆里的向量，规则从 40 张关键帧的 LiDAR 深度算出来，只算一次、存一次。
  - 推、按：从外面滑进表面；
  - 拉：从部件往外滑；
  - 门窗：沿开的方向往外；
  - 旋钮：一段绕轴来回摆的弧。
- **标签：** 物体名和动作名。物体名取自这个部件对应任务里的名词，没有任务的部件按标签叫"旋钮""按钮"等；动作名取自它的 affordance 标签。

**看点：** 镜头怎么动，高亮和箭头都留在原来的部件上。因为它们存在三维世界坐标里，不是存在某一张图上。这就是"写一次，从哪个角度都能读"。

**要注意的地方：**
- **这是演示，不是检验。** 规则是 S0c 之后定的，而这个场景 S0 已经用过。检验看第 3 节的 S0c。
- **部件掩码和物体名用的是标注（ORACLE）。**
- **画到画面上用的是数据集里和扫描对齐的相机**，只为画得准。记忆本身用的是手机自己的位姿和 LiDAR。
- **箭头只画记忆的向量，不画标注。** 记忆错的地方，箭头也是错的。
- **烤箱门往下翻，轴是水平的。** 只看标签和深度，规则分不出它是门还是抽屉，所以视频里它的箭头是"往外拉"，不对。这正是"要问它装在什么上"的例子。
- **水龙头：** 记忆把转轴取成了表面的法线，和真的转轴差了约 80°，所以那两段弧的方向不对。灶台旋钮的轴大多是对的（6 个里 4 个在 15° 以内）。
""")

code(r"""
display(Video(str(ASSETS / "week2_walkthrough_421069.mp4"), embed=True, width=540, html_attributes="controls loop muted playsinline"))
w = media["walkthrough"]
print(f"{w['frames']} frames; at least one memory part in view in {w['frames_with_parts']} of them")
""")

md(r"""
## 1.2 视频二：三维的记忆（同一个场景）

**这是什么：** 这个场景的激光扫描点云，去掉了天花板，镜头在上方绕一圈。画法和视频一相同（11-20 重画，和视频一对齐）：每个部件按动作类别上色高亮，箭头来回滑动表示怎么动，标签写物体名和动作名。只画从这个角度看得见的部件：部件的扫描点要和画出来的点云深度对得上，被墙挡住的不画。箭头同样只画记忆的向量。

**看点：** 记忆就是一张"部件表"，这是它在三维里的样子：一眼能看到整个房间里哪些地方能交互、各是什么动作，每一条都能画出来、能检查。
""")

code(r"""
display(Video(str(ASSETS / "week2_memory_orbit_421069.mp4"), embed=True, width=720, html_attributes="controls loop muted playsinline"))
o = media["orbit"]
print(f"{o['frames']} frames, {o['points']:,} scan points; median {o['parts_drawn_median']:.0f} of {media['parts']} parts in view")
""")

md(r"""
## 1.3 图：事先定好预测的检验 S0c（场景 466437，卧室）

这一张来自正式检验 S0c：规则在跑之前就定好了，场景也是事先按规则选的，这个场景以前没用过。

- **上中的房门：** 橙色的轴是竖直的，和标注的门轴几乎重合。门轴取重力方向，这一步修好了。
- **右下的衣柜门：** 轴的方向对了，但橙色的轴画在把手旁边，真门轴在门的另一边。这是"门轴在哪里"的问题，靠深度找不准门板的边。
- **上右、下中的抽屉：** 被几何判断当成了门，画成了竖直轴。错在第一步"门还是抽屉"，不在轴。
""")

code(r"""
display(Image(filename=str(FIG / "s0c_2026-11-19" / "keyframes_466437.jpg"), width=1000))
""")

md(r"""
## 1.4 图：门多的场景 S0b（场景 422842）

S0b 特意挑了门多的场景。衣柜门上，旧规则多数画成往外拉的箭头（当成抽屉），而标注是竖直的门轴。门轴"大多是竖直的"这条规律就是在这里事后看出来的，然后在 S0c 上事先定好再检验。
""")

code(r"""
display(Image(filename=str(FIG / "s0b_2026-11-19" / "keyframes_422842.jpg"), width=1000))
""")

md(r"""
## 1.5 图：机器人拿到记忆会怎么做（场景 421069）

把每条向量变成"手碰到的那一点会怎么走"：
- 推拉的部件：沿方向走 30 cm；
- 绕轴转的部件：绕轴转 60°；
- 旋钮：原地转一圈。

橙色是按记忆里的向量走，绿色是按标注走。每类部件挑两个，挑法是事先定的（出现在最多关键帧里的两个），不看结果好坏，所以对的和错的都有。每张小图的标题写了类型和轴对不对。

往哪边转、往外还是往里，标注里不统一，所以两条路径都按同一条规则定方向：一开始离开表面；按的部件往里。
""")

code(r"""
examples = H.pick_examples(demo_rows, truths, DEMO["main"])
fig = H.motion_paths_figure(examples, truths, frames["lidar"])
display(fig)
plt.close(fig)
for name, row in examples:
    print(f"{name:15s} part {row['part']:3d}  {row['label']:10s}  task: {row['task'][:70]}")
""")
