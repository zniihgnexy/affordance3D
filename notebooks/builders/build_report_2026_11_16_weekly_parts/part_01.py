# Cells of build_report_2026_11_16_weekly.py, part 1 of 4: run in order by that builder, with md() and code() in scope.

md(r"""
# 周报：2026-11-09 到 2026-11-15（第 1 到第 10 轮）
**视频功能记忆（部件级，记忆方向）** · 写给项目负责人 · 2026-11-16 生成
""")

md(r"""
# 先读这里：用大白话讲这一周

> 这一节写给"没有跟着每一轮讨论"的读者。它只讲大局：我们想回答什么问题、这一周做了什么、为什么这么做、结果怎样、下一步为什么这么选。后面的各节是细节和证据，这一节里的每个数字在后面都能找到出处。
""")

md(r"""
## 1. 我们想回答什么问题

**一个例子。** 你拿着手机（或戴着眼镜）在家里走一圈，录一段视频。过了几天，你问："床头柜最上面那个抽屉怎么打开？"系统要在录下的画面里，指出你该用手碰的那个小部件：抽屉的拉手。它还可以告诉你往哪个方向拉。

**我们要做的是一个"记忆"。** 视频只录一次，存成一份记忆；之后可以问很多次，每次都很快地找到答案，不用把整段视频重新看一遍。

**我们的核心想法（"功能记忆"）。** 一般的记忆按"画面里看起来有什么"来存，比如"这里有床头柜"。我们的想法是按"能对什么做什么"来存，比如"这里有一个可以拉的拉手、一个可以转的旋钮"。我们想知道：这样存，是不是更容易找到要操作的那个部件。
""")

md(r"""
## 2. 先认识几个词

| 词 | 用大白话说 |
| --- | --- |
| 关键帧 | 从视频里挑出来的几十张画面，记忆就存在这些画面上（每个房间 40 张） |
| 部件图（affordance map） | 在画面上标出哪些小部件能操作：拉手、旋钮、开关、插座等 |
| 检索 | 来了一个问题，从几十张画面里挑出最相关的几张（我们挑 3 张） |
| 视觉语言模型 | 能看图回答问题的 AI 模型。我们用开源的 Qwen3-VL，有 4B 和 8B 两个大小（8B 更大） |
| 编号框 | 在图上把候选部件框起来、编上号，让模型直接说"选 3 号" |
| ORACLE（标注） | 用人工标好的部件图代替模型预测。它回答的是"如果部件图完美，最好能做到哪"，是上限，不是方法的真实水平 |
| 对照 | 只拿掉一个因素，看结果变了多少，用来判断功劳到底归谁 |
| 事先写定的规则 | 每个实验跑之前，就写好"怎样算成功"。这样不能看了结果再找说法 |
| 关口（G1、G2） | 事先定好的检查点，过了才往下走。G1：理想条件下这个想法行不行（已过）。G2：用现成工具时主张还站不站得住、要不要训练模型、投哪个会议（还没到） |
| SAM 3 | Meta 公司公开的分割模型：给它一个词（比如 handle），它就在图里把对应的东西框出来 |
| 端到端 | 从问题一路走到最后选出部件，看最后答对没有；不只看中间挑画面那一步 |
| 掌舵代理、审稿代理 | 掌舵代理是帮你推进项目、写这份周报的 AI 助手；审稿代理是只读文件、专门挑毛病的独立 AI 审稿人 |
| 第 N 轮 | 每次向你汇报结果、请你做决定的一轮讨论；这一周从第 1 轮到第 10 轮 |
""")

md(r"""
## 3. 这一周的研究路线（按时间讲）

**第一步：先在理想条件下看这个想法行不行（阶段 1，11 月 12 日）。** 在 30 个真实房间、390 个问题上，用标注的部件图建记忆，看能不能挑对画面。结果：第一张就挑对的比例，按功能存是 66%，只按整张画面是 46%，存同样多的普通分割区域（不管能不能操作）是 47%。所以这个想法在理想条件下成立，关口 G1 通过。补充检查发现，大约一半的好处来自一个很简单的信息："这张画面上有没有可操作的部件"。

**第二步：和"传统做法"比（第 7 轮，你的要求）。** 传统做法是让视觉语言模型直接看原视频（40 帧）自己找部件。结果（用的是标注的部件图，是上限）：用我们的记忆，模型选对部件的比例是 53%，直接看原视频只有 28%。读取时我们的记忆还快 4 到 5 倍（只算读取时的模型调用，不含建记忆的时间），看的图少十几倍。看起来很好。

**第三步：追问这 25 个百分点到底是谁的功劳（第 9 轮）。** 我们的做法其实给了模型两样东西：一是记忆挑出的画面，二是画面上的部件编号框。审稿代理问：功劳是哪一样的？于是加了一个对照 c4：给模型同样的部件编号框，但画面不用记忆挑，只按普通的语义相似度挑。结果 c4 也有 48%，只比记忆少 4 到 7 个百分点（两边都用标注的部件图，是上限）。4B 和 8B 两个模型都一样。**结论：这 25 个百分点几乎都是"部件编号框"的功劳，记忆挑画面只贡献了一小部分。** 另外，按"动作类型"（拉、转、按）来检索也没有用：即使直接告诉它真实的动作类型，也只多 3.8 个百分点（标注的部件图，上限）。

**第四步：调研发现，有两件事别人已经做了（第 9 到 10 轮）。** 一是"从视频建部件图"，已有 FunGraph、Functional-SLAM 等工作。二是"给模型看编号框让它选"，已有 Set-of-Mark、AffordBot 等做法。所以"给模型看编号框"不能当作我们的新贡献；"新的部件图构建方法"算不算贡献，是一个推迟的决定（账本 DF-4），等 SAM 3 的结果出来再定。

**第五步：重新想记忆什么时候才有用（第 10 轮评估）。** 在一个房间里，按普通语义挑画面已经够用了，记忆没什么用武之地。记忆真正该起作用的，是"难找"的时候：录得很长、有很多相似的部件（比如一排一模一样的抽屉）。你采纳了这个新主张。

**第六步：第一次检验新主张，没过（阶段 2b）。** 做法是把多个房间的画面混在一起，故意让检索变难，看功能记忆的优势会不会变大。结果（标注的部件图，上限）：功能记忆一直比普通语义多挑对 12 到 18 个百分点的画面，但这个差距不随难度变大。按事先写定的规则，判为"检索这一步不支持新主张"。

**第七步：你选了两件并行的事（现在）。** 一是用现成工具 SAM 3 建部件图，看不用人工标注时还剩多少（示例图的 6 张关键帧里，SAM 3 的框几乎都落在部件上）。二是在混合之后让模型最后选部件，看端到端的结果。
""")

code(r"""
# the research path of the week, one box per step of the section above (done / failed / running)
import textwrap

import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.patches import FancyBboxPatch

STEPS = [("1 Phase 1: idea works with annotated maps (66% vs 46%)", "done"),
         ("2 vs raw-video VLM: 53% vs 28%", "done"),
         ("3 Control c4: the gain is the part boxes (48% vs 53%)", "done"),
         ("4 Survey: map building and numbered marks already exist", "done"),
         ("5 New claim: memory helps when retrieval is hard", "done"),
         ("6 Phase 2b: gap steady (12\u201318 pts), does not grow", "failed"),
         ("7 Now: SAM 3 maps + end-to-end pooled test → gate G2", "running")]
STATUS_STYLE = {"done": ("#d7f0d2", "#2e7d32"), "failed": ("#f9d6d5", "#c0392b"), "running": ("#d5e6fb", "#1f5fae")}
WIDTH, GAP, HEIGHT = 2.35, 0.45, 1.85
fig, axis = plt.subplots(figsize=(19, 3.0))
for i, (label, status) in enumerate(STEPS):
    x = i * (WIDTH + GAP)
    fill, edge = STATUS_STYLE[status]
    axis.add_patch(FancyBboxPatch((x, 0), WIDTH, HEIGHT, boxstyle="round,pad=0.05", fc=fill, ec=edge, lw=2.2))
    axis.text(x + WIDTH / 2, HEIGHT - 0.15, status, ha="center", va="top", fontsize=12, weight="bold", color=edge)
    axis.text(x + WIDTH / 2, HEIGHT / 2 - 0.12, textwrap.fill(label, 17), ha="center", va="center", fontsize=12.5, color="#0b0b0b",
              linespacing=1.25)
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
## 4. 现在站在哪里

- **最可靠的发现：** 给模型看部件编号框，它找对要操作的部件的比例从约 28% 翻到约 50%。这是"部件图"的价值，不是"记忆检索"的价值。目前这个结果用的是标注的部件图，所以是上限。
- **两个最大的未知：**
  1. 用现成工具建的部件图够不够好？SAM 3 正在跑，示例图看起来很好。
  2. 在难找的时候，记忆能不能帮模型最后选对部件？端到端测试正在排队。
- **另外两条线：**
  - B（问"我做过什么"）暂停了，因为只靠手腕跟踪找不出"拿起、放下"的时刻。
  - C（告诉你怎么动）：第一次试跑发现模型的答题格式不对，改好后已重新排队试跑；全部场景还没开始跑。
""")

md(r"""
## 5. 下一步为什么这么选

- **为什么先做 SAM 3 的检查：** 到现在为止，所有好结果都用的是人工标注的部件图。真正的系统没有标注，必须自己建图。
  - 如果现成工具就能建出好的部件图，系统就可以用"现成工具 + 记忆 + 模型"搭起来。那时论文的价值就要看记忆在"难找"时有没有用。
  - 如果现成工具不行，"把部件图建好"本身可能成为贡献。它算不算贡献，本来就是推迟到这一步再定的（账本 DF-4）。
- **为什么还要做端到端的混合测试：** 检索这一步的差距虽然不变大，但模型最后是从挑出的 3 张画面里选部件。房间混得越多，普通语义挑出的 3 张里"一张对的都没有"的情况就越多，这时给再好的编号框也救不回来。所以最后选对部件的差距，有可能随难度变大。这个测试是在第一步没过之后才加的，解读时要更谨慎。
- **两个结果出来以后：**
  - 如果它们支持主张：下一步是阶段 3，去现成的公开基准上做正式的测试（下载要你批）。
  - 如果都不支持：到关口 G2，由你在三条路里选：
    1. 改成做"新的部件图构建方法"：要训练模型。
    2. 换投稿方向，例如 WACV 的应用方向。
    3. 停下来重新想。
""")

md(r"""
## 6. 掌舵代理的看法（一句话）

这一周的结论比一周前更清楚，也更保守：**部件图是真正起作用的东西；"按功能组织的记忆"能不能成为论文的核心，取决于正在跑的两个检查。** 下周最要紧的是拿到这两个结果，而不是再加新的标注对照。
""")

md(r"""
---

**下面是细节。**

# 细节和证据

下面第 0 到第 9 节是各项的细节和证据，节号自成一套；下文说的"第 x 节"都指这一套。

**配套文档：** [章程](../helm/CHARTER.md) · [账本](../helm/LEDGER.md)（每轮的问题和你的原话）· [审稿台账](../helm/REVIEW.md) · [第 10 轮评估](../helm/assessments/2026-11-13.md) · [讨论稿](../docs/mem/mem分支_问题定义与讨论稿_2026-11-10.md) §11–§26 · [阶段 1 结果 notebook](../notebooks/mem/phase1_results.ipynb)

**这份 notebook 只报告，不替你做决定。** 开头的"先读这里"由掌舵代理撰写，原样收入；要你决定的事集中在第 0 节末尾和第 7 节。写这份周报的代理自己的推断，都标为"推断"。

**读数的约定**
- **百分比：** 除非另说，都是 SceneFun3D 验证集（split 0）30 个场景里 390 个"可回答"任务中的比例。"可回答"指任务要找的部件在我们挑的 40 张关键帧里至少有一张看得到；445 个任务里有 390 个。一个任务的例子："Open the top drawer of the cabinet"。
- **个百分点（点）：** 两个比例之差。例如 54.4% 减 47.7% 是 +6.7 个百分点。
- **方括号 [a, b]：** 按场景重抽样得到的 95% 置信区间。直观地说：换一批同类的场景，结果大概率落在这个范围里。**区间下沿**就是 a。
- **ORACLE（上限）：** 用数据集的标注代替还没做出来的部件图（affordance map，即"画面上哪里可以操作、用什么动作"）。这类数字说明"如果部件图是对的，能做到多少"，不说明方法现在能做到多少。表格里有一列"部件图"注明。
- **出处：** 数字后面的 §x.y 是讨论稿的节号。结果文件都在服务器的 `runs/` 下，不进 git。附录 A 列出每个关键数字取自哪个文件，并在运行时逐个核对。
- **模型：** 4B、8B 指视觉语言模型 Qwen3-VL-4B 和 Qwen3-VL-8B；都是现成的公开模型，没有训练。图里的标签用英文。

<sub>重新运行：在仓库根目录执行 `python notebooks/builders/build_report_2026_11_16_weekly.py report/2026-11-16_weekly_summary.ipynb`，再执行 `jupyter nbconvert --to notebook --execute --inplace report/2026-11-16_weekly_summary.ipynb`。只用 CPU，约一分钟；需要服务器上 `runs/` 里的结果文件。</sub>
""")

code(r"""
import contextlib
import io
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore")
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance3d").is_dir())
_saved_fd, _null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
os.dup2(_null, 1)  # silence Open3D's import banner (printed from C++) if an import below pulls it in
try:
    with contextlib.redirect_stdout(io.StringIO()):
        sys.path.insert(0, str(ROOT))
        from report import pipeline_flowchart as FC  # switches matplotlib to Agg; switched back below
        from report import mem_phase1_helpers as R
finally:
    os.dup2(_saved_fd, 1)
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

plt.switch_backend("module://matplotlib_inline.backend_inline")
set_matplotlib_formats("png")
plt.rcParams.update({"figure.dpi": 100})
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.max_rows", 200)
RUNS = ROOT / "runs"
FILES = {  # short name -> result file under runs/ (written by the scripts named in each section)
    "phase1": "mem_phase1/summary.json", "diag": "mem_phase1/diagnostics.json", "c4b": "mem_phase1/controls_summary.json",
    "c8b": "mem_phase1/controls_8b_summary.json", "by_action": "mem_phase1/controls_by_action.json",
    "a": "mem_phase1/control_a.json", "ceiling": "mem_phase1/frame_ceiling.json", "errors": "mem_phase1/map_errors.json",
    "errors_centre": "mem_phase1/map_errors_centre.json", "key": "mem_phase1/action_key.json",
    "pooled": "mem_phase1/pooled_retrieval.json", "b0": "hdepic/b0_events.json", "b0_swapped": "hdepic/b0_events_swapped.json",
    "b1": "hdepic/b1_hand_events.json", "c0": "mem_c0/summary.json", "c1_4b": "mem_c1/preview/4b_summary.json",
    "c1_8b": "mem_c1/preview/8b_summary.json"}
missing = [rel for rel in FILES.values() if not (RUNS / rel).is_file()]
if missing:
    raise FileNotFoundError(f"result files missing under runs/: {missing}")
D = {name: json.loads((RUNS / rel).read_text()) for name, rel in FILES.items()}


def get(name, *path):
    value = D[name]
    for key in path:
        value = value[key]
    return value


def rate(d):  # '66.2% [60.0, 73.3]'
    return f"{100 * d['rate']:.1f}% [{100 * d['low']:.1f}, {100 * d['high']:.1f}]"


def diff(d):  # '+18.8 [+12.9, +24.9]'
    return f"{100 * d['rate']:+.1f} [{100 * d['low']:+.1f}, {100 * d['high']:+.1f}]".replace("-", "−")


def show(fig):
    display(fig)
    plt.close(fig)


INK, MUTED, GRID = R.INK, R.MUTED, R.GRID
BOX, NOBOX, GENERIC, MEMORY = R.POLICY_COLOUR[R.A.P4], "#e34948", R.POLICY_COLOUR[R.A.P3], "#1baf7a"
print(f"{len(FILES)} result files read; phase 1: {get('phase1', 'scenes')} scenes, {get('phase1', 'tasks')} of "
      f"{get('phase1', 'tasks_all')} tasks answerable")
""")

code(r"""
# small figures of this report; the phase-1 and control-c figures come from report/mem_phase1_helpers.py


def headline_figure(c4b, c8b):
    arms = [("c3", "c3: raw video, 40 even frames;\nVLM picks a frame, draws a box"),
            ("c1s", "c1': 3 keyframes chosen by meaning;\nVLM picks one, draws a box"),
            ("c2s", "c2': the memory's 3 keyframes;\nVLM picks one, draws a box"),
            ("c4", "c4: annotated part boxes on 3 keyframes\nchosen by meaning; VLM picks a number"),
            ("p4n", "P4: annotated part boxes on the memory's\n3 keyframes; VLM picks a number")]
    fig, axis = plt.subplots(figsize=(11.5, 4.9))
    for row, (arm, _) in enumerate(arms):
        colour = BOX if arm in ("c4", "p4n") else NOBOX
        for offset, summary, model, alpha in ((-0.2, c4b, "4B", 0.5), (0.2, c8b, "8B", 1.0)):
            d = summary["arms"][arm]["right_part"]
            axis.barh(row + offset, 100 * d["rate"], height=0.36, color=colour, alpha=alpha)
            axis.plot([100 * d["low"], 100 * d["high"]], [row + offset] * 2, color=INK, lw=1.1)
            axis.text(100 * d["high"] + 1.0, row + offset, f"{100 * d['rate']:.1f}% ({model})", va="center", fontsize=8.5, color=INK)
    axis.set_yticks(range(len(arms)), [label for _, label in arms], fontsize=9, color=INK)
    axis.invert_yaxis()
    axis.axhline(2.5, color=MUTED, lw=0.8, ls="--")
    axis.set_xlim(0, 80)
    axis.set_title("Share of tasks where the chosen box lies on the target part (%); light = 4B, dark = 8B; line = 95% interval\n"
                   "red: no part boxes, the VLM draws its own box; blue (below the dashed line): numbered annotated part boxes (ORACLE)",
                   fontsize=10, loc="left", color=INK)
    R._clean(axis)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    R._oracle_note(fig, "390 tasks, 30 SceneFun3D scenes. c4 and P4 use annotated part maps (ORACLE, upper bounds); c2' and P4 "
                        "take keyframes picked by the memory, whose keyframe flag also comes from the annotations.")
    return fig


def timeline_figure():
    days = ["11-09\nMon", "11-10\nTue", "11-11\nWed", "11-12\nThu", "11-13\nFri", "11-14\nSat", "11-15\nSun", "11-16\nMon"]
    commits = [5, 9, 7, 7, 24, 24, 2, 1]
    events = ["dataset report;\nlocal prototype", "rounds 1-2;\nS1 steps 1-5;\nrepository split", "rounds 3-4;\n15 pipeline\nblocks",
              "rounds 5-6;\nphase-1 spec;\nhelm charter", "phase 1;\nrounds 7-9;\ncontrols c, a, b,\nd, c+; R7-B",
              "(work of 11-13\nafter midnight)\nc4, 8B; B-0, B-1,\nC-0; round 10; 2b", "answer 'c+a';\n2b end-to-end\nplanned",
              "queue 'b';\ngate seen,\n'a' (after\nthe week)"]
    fig, axis = plt.subplots(figsize=(12, 4.3))
    colours = [MUTED if i in (0, 7) else BOX for i in range(len(days))]
    axis.bar(range(len(days)), commits, color=colours, width=0.6)
    for x, (n, text) in enumerate(zip(commits, events)):
        axis.text(x, n + 0.8, f"{n}", ha="center", va="bottom", fontsize=9, color=INK, weight="bold")
        axis.text(x, n + 3.2, text, ha="center", va="bottom", fontsize=7.8, color=MUTED, linespacing=1.15)
    axis.set_xticks(range(len(days)), days, fontsize=9, color=INK)
    axis.set_ylim(0, 44)
    axis.set_ylabel("commits", fontsize=9, color=MUTED)
    axis.set_title("Commits per day (commit timestamps) and what happened; rounds by the ledger's dates", fontsize=10, loc="left",
                   color=INK)
    R._clean(axis, grid="y")
    fig.tight_layout()
    return fig


def bars_and_differences(rows, keys, labels, colours, diff_rows, diff_keys, diff_labels, diff_colours, left_title, right_title,
                         note, rule_at=None, rule_text=""):
    height = 0.62 * max(len(keys), len(diff_keys)) + 1.6
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, height), gridspec_kw={"width_ratios": [1.1, 1]})
    R._rate_bars(left, rows, keys, labels, colours)
    left.set_title(left_title, fontsize=10, loc="left", color=INK)
    R._difference_dots(right, diff_rows, diff_keys, diff_labels, diff_colours)
    right.set_title(right_title, fontsize=10, loc="left", color=INK)
    if rule_at is not None:
        right.axvline(rule_at, color=MUTED, lw=1, ls="--")
        right.text(rule_at - 0.3, len(diff_keys) - 0.55, rule_text, fontsize=8, color=MUTED, ha="right", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    R._oracle_note(fig, note)
    return fig


def map_error_figure(errors, centre):
    keeps, alarms = ["1", "0.75", "0.5", "0.25"], ["0", "0.5", "1", "2"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8))
    for row, (data, rule) in enumerate(((errors, "false alarm = generic region with IoU < 0.5 to every part (pre-set rule)"),
                                        (centre, "stricter re-run: no generic region whose centre lies on a part"))):
        for col, (order, title) in enumerate((("random", "parts dropped at random"), ("smallest_first", "smallest parts dropped first"))):
            axis = axes[row, col]
            grid = np.array([[100 * data["grid"][order][k][a]["rate"] for a in alarms] for k in keeps])
            axis.imshow(grid, cmap="Blues", vmin=40, vmax=70)
            for i, k in enumerate(keeps):
                for j, a in enumerate(alarms):
                    better = data["grid"][order][k][a]["against_p3"]["verdict"] == "clearly better"
                    axis.text(j, i, f"{grid[i, j]:.1f}" + ("" if better else "\n= P3"), ha="center", va="center", fontsize=9,
                              color="white" if grid[i, j] > 60 else INK, weight="normal" if better else "bold")
            axis.set_xticks(range(4), ["0", "0.5x", "1x", "2x asked"], fontsize=9)
            axis.set_yticks(range(4), ["100%", "75%", "50%", "25%"], fontsize=9)
            axis.set_xlabel("false alarms, as a multiple of the true parts", fontsize=8.5, color=MUTED)
            axis.set_ylabel("share of true parts kept", fontsize=8.5, color=MUTED)
            axis.set_title(f"{title}\n{rule}", fontsize=8.8, loc="left", color=INK)
    fig.suptitle("First keyframe right (%) when the annotated part map is damaged; '= P3' marks cells not clearly better than generic "
                 f"regions (P3 {100 * errors['p3']:.1f}%, whole keyframes {100 * errors['p0']:.1f}%)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    R._oracle_note(fig, "Perturbed annotations, retrieval only, no VLM. The 2x column held 1.46x (1.39x in the stricter re-run) on average.")
    return fig


def pooled_figure(pooled):
    sizes = pooled["pool_sizes"]
    per = pooled["per_size"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for arm, colour in (("semantic", R.POLICY_COLOUR[R.A.P0]), ("function index", BOX), ("function index + action key", "#eda100")):
        rows = [per[str(k)][arm]["top3"] for k in sizes]
        left.fill_between(sizes, [100 * r["low"] for r in rows], [100 * r["high"] for r in rows], color=colour, alpha=0.12, lw=0)
        left.plot(sizes, [100 * r["rate"] for r in rows], "-o", color=colour, lw=2, ms=6, mec="white", label=arm,
                  ls="--" if "key" in arm else "-")
    left.set_ylabel("right keyframe among the top 3 (%)", fontsize=9, color=MUTED)
    left.legend(frameon=False, fontsize=9)
    left.set_title("Retrieval when K scenes are pooled into one memory", fontsize=10, loc="left", color=INK)
    for offset, key, colour, label in ((-0.06, "function index minus semantic (top3)", BOX, "function index − semantic"),
                                       (0.06, "function index + action key minus function index (top3)", "#eda100",
                                        "action key − function index")):
        rows = [per[str(k)][key] for k in sizes]
        xs = [k * 2 ** offset for k in sizes]
        right.errorbar(xs, [100 * r["rate"] for r in rows], yerr=[[100 * (r["rate"] - r["low"]) for r in rows],
                       [100 * (r["high"] - r["rate"]) for r in rows]], fmt="o", color=colour, capsize=3, label=label)
    reading = pooled["readings"]["function index minus semantic"]["slope_per_doubling"]
    right.axhline(0, color=INK, lw=0.8)
    right.set_ylabel("paired difference, points (95% interval)", fontsize=9, color=MUTED)
    right.set_title(f"Gap per K; slope of the gap: {diff(reading)} points per doubling", fontsize=10, loc="left", color=INK)
    right.legend(frameon=False, fontsize=9, loc="center right")
    for axis in (left, right):
        axis.set_xscale("log", base=2)
        axis.set_xticks(sizes, [str(k) for k in sizes])
        axis.minorticks_off()
        axis.set_xlabel("K: scenes pooled into one memory (the target scene + K-1 others)", fontsize=9, color=MUTED)
        R._clean(axis, grid="y")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    R._oracle_note(fig, "Annotated part maps (ORACLE) on both sides; retrieval only, no VLM. 390 tasks; 5 draws of the other scenes per K > 1.")
    return fig


def hdepic_figure(b0, b0s, b1):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    for x, (summary, label, colour) in enumerate(((b0["summary"], "as written", NOBOX), (b0s["summary"], "box axes swapped (post hoc)", BOX))):
        for offset, key, n in ((-0.18, "tied_share", "questions"), (0.18, "reproduced_share", "rule_questions")):
            left.bar(x + offset, 100 * summary[key], width=0.32, color=colour, alpha=1.0 if offset < 0 else 0.55)
            left.text(x + offset, 100 * summary[key] + 1.5, f"{100 * summary[key]:.1f}%", ha="center", fontsize=8.5, color=INK)
        left.text(x, 108, label, ha="center", va="bottom", fontsize=9, color=colour, weight="bold")
    left.axhline(90, color=INK, lw=1, ls="--")
    left.text(-0.55, 91, "bar: 90%", fontsize=8.5, color=INK, ha="left", va="bottom")
    left.set_xticks([-0.18, 0.18, 0.82, 1.18], ["questions\ntied", "answers\nreproduced", "questions\ntied", "answers\nreproduced"],
                    fontsize=8.5)
    left.set_xlim(-0.6, 1.6)
    left.set_ylim(0, 125)
    left.set_title("B-0 (kitchen P02, 242 questions): questions tied to annotated\nevents; rule answers reproduced (of 54)", fontsize=10,
                   loc="left", color=INK)
    R._clean(left, grid="y")
    summary = b1["summary"]
    right.add_patch(plt.Rectangle((50, 60), 50, 40, color=BOX, alpha=0.08, lw=0))
    right.text(52, 97, "pass: precision >= 50% with recall >= 60%", fontsize=8.5, color=BOX, va="top")
    curve = summary["curve"]
    right.plot([100 * curve[k]["precision"] for k in curve], [100 * curve[k]["recall"] for k in curve], "-o", color=MUTED, ms=5)
    offsets = {"0.05": (6, -4), "0.1": (6, -4), "0.15": (8, -12), "0.2": (-46, 3), "0.3": (6, 2)}
    for k in curve:
        right.annotate(f"{k} m/s", (100 * curve[k]["precision"], 100 * curve[k]["recall"]), textcoords="offset points",
                       xytext=offsets.get(k, (6, -3)), fontsize=7.5, color=MUTED)
    right.plot(100 * summary["precision"], 100 * summary["recall"], "o", color=NOBOX, ms=10, label="pre-set operating point (0.15 m/s)")
    shift = summary["control_shift_15s"]
    right.plot(100 * shift["precision"], 100 * shift["recall"], "s", color=INK, ms=7, label="control: detections shifted by 15 s")
    ceiling = 100 * summary["palm_within_30cm_at_0"] * summary["clock_check_palm_to_object"]["n_at_0"] / summary["truths"]
    right.axhline(ceiling, color=INK, lw=0.8, ls=":")
    right.text(99, ceiling + 1, f"about {ceiling:.0f}%: a tracked palm is within 30 cm at the annotated moment", fontsize=8, ha="right")
    right.set_xlim(0, 100)
    right.set_ylim(0, 100)
    right.set_xlabel("precision (%)", fontsize=9, color=MUTED)
    right.set_ylabel("recall (%)", fontsize=9, color=MUTED)
    right.set_title(f"B-1: pick-ups and put-downs from the hand track alone ({summary['truths']:,} annotated)", fontsize=10, loc="left",
                    color=INK)
    right.legend(frameon=False, fontsize=8.5, loc="lower right")
    R._clean(right, grid="both")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    R._oracle_note(fig, "HD-EPIC kitchen P02, CPU only; annotations used only for scoring. B is paused since round 10.")
    return fig


def c_figure(c0, c1_4b, c1_8b):
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [0.8, 1.4]})
    criteria = c0["criteria"]
    rows = [("every target part has a motion", criteria["motions_for_every_part"]["share"], criteria["motions_for_every_part"]["threshold"]),
            ("motion arrow >= 20 px on the keyframe", criteria["arrow_at_least_20px"]["share"], criteria["arrow_at_least_20px"]["threshold"]),
            ("scorer gives the truth 100%", 1.0 if criteria["scorer_gives_truth_100"]["holds"] else 0.0, 1.0)]
    for y, (label, share, bar) in enumerate(rows):
        left.barh(y, 100 * share, height=0.5, color=MEMORY)
        left.plot([100 * bar] * 2, [y - 0.32, y + 0.32], color=INK, lw=2)
        left.text(2, y, f"{100 * share:.1f}%  (bar {100 * bar:.0f}%)", va="center", fontsize=9, color="white")
    left.set_yticks(range(3), [r[0] for r in rows], fontsize=9)
    left.invert_yaxis()
    left.set_xlim(0, 105)
    left.set_title("C-0: action records, 390 tasks (all three bars met)", fontsize=10, loc="left", color=INK)
    R._clean(left)
    fields = [("contact", "contact on\nthe part"), ("label", "affordance\nlabel"), ("type", "motion type"),
              ("line_ok", "direction line\nwithin 20 deg"), ("joint", "joint success\n(all of these)")]
    arms = [(c1_4b, "M", "M (memory), 4B choice", BOX, 0.5), (c1_4b, "M4", "M4 (no memory), 4B choice", GENERIC, 0.5),
            (c1_8b, "M", "M (memory), 8B choice", BOX, 1.0), (c1_8b, "M4", "M4 (no memory), 8B choice", GENERIC, 1.0)]
    width = 0.19
    for i, (summary, arm, label, colour, alpha) in enumerate(arms):
        xs = np.arange(len(fields)) + (i - 1.5) * width
        values = [summary["arms"][arm][f]["rate"] for f, _ in fields]
        right.bar(xs, [100 * v for v in values], width=width, color=colour, alpha=alpha, label=label)
        for x, f in zip(xs, [f for f, _ in fields]):
            d = summary["arms"][arm][f]
            right.plot([x, x], [100 * d["low"], 100 * d["high"]], color=INK, lw=0.9)
    right.set_xticks(range(len(fields)), [label for _, label in fields], fontsize=9)
    right.set_ylim(0, 100)
    right.set_ylabel("share of 380 tasks (%)", fontsize=9, color=MUTED)
    right.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    right.set_title("C-1 CPU preview: memory (M) vs the same annotated boxes on frames picked without it (M4)", fontsize=10, loc="left",
                    color=INK)
    R._clean(right, grid="y")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    R._oracle_note(fig, "M and M4 both take the contact point and label from annotated regions (ORACLE). The preview has no V or G arm "
                        "(they need the GPU) and is not in the draft yet.")
    return fig
""")
