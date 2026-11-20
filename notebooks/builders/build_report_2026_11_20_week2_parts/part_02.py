# Cells of build_report_2026_11_20_week2.py, part 2 of 3 (the pipeline figure and every block with a real example).

md(r"""
---
# 2. 推荐的 pipeline（v3）和逐块讲解

## 2.1 pipeline 图
""")

code(r"""
display(Image(filename=str(FIG / "proposal_2026-11-18" / "proposal_v3.png"), width=1000))
""")

md(r"""
**怎么看这张图：**
- **上半部分是写入**，每段视频做一次：关键帧 → 部件掩码 → 每个部件的类型和"装在什么上" → 加上深度和位姿，算出每个部件的 affordance 向量 → 存进记忆（部件表）。
- **下半部分是读取**，每个问题做一次：问题 → 拆出物体和动作类型 → 按功能挑关键帧 → 两个头读同一份记忆。
  - 问答头：给自由文字回答，用 OpenEQA 评。
  - 动作头：选出部件，读出它的向量，交给机器人。先告诉它"抓哪里、怎么抓"，再告诉它"抓住以后怎么动"，用 SceneFun3D 评。
- **颜色：** 橙色是论文要主张的部分；蓝色是直接拿来用、不训练的公开模型；绿色是两个头；红线是记忆被读取的路径。
- **虚线：** 还没做的部分。

**和上周五的 v2 比，这周改了三处：**
1. 类型那一步写明还要问"装在什么上"（门、抽屉、面板）；
2. 向量的规则写明门轴取重力方向，并加了抓握轴；
3. 动作头的输出写明是"先抓、再动"。

源文件：`docs/mem/figures/proposal_2026-11-18/proposal_v3.yaml`（cvpr-figure 格式），另有 svg、pdf、pptx。
""")

md(r"""
## 2.2 每一块现在是什么状态

| 块 | 做什么 | 用什么 | 现在 | 这一周的证据 |
| --- | --- | --- | --- | --- |
| W1 关键帧 | 从视频里挑 40 张最稳的画面 | 位姿和画面清晰度 | 有 | 漏掉 15% 到 21% 的部件 |
| W2 部件掩码 | 把能操作的小部件分出来 | SAM 3 | 有；本周的检验用标注代替 | 上周：SAM 3 找到 74.8% |
| W3 深度和位姿 | 每个像素多远、相机在哪 | 手机的 LiDAR 和位姿 | 有 | LiDAR 和渲染深度差不多；少数部件没有 LiDAR 深度 |
| W4 类型和所装的东西 | 推、拉、转；装在门、抽屉还是面板上 | 标签 + VLM | 类型有（来自标注的标签）；"装在什么上"没做 | 只看标签最多对 80% |
| W5 affordance 向量 | 接触点、法线、类型、轴、原点 | 规则 | 原型有 | S0、S0b、S0c |
| 记忆 | 部件表 | — | 有 | — |
| R1 问题解析 | 拆出物体和动作类型 | 大语言模型 | 没做 | — |
| R2 挑关键帧 | 按语义和类型挑 | SigLIP 2 + 部件表 | 部分有 | 上周 |
| R3b 动作头 | 选部件、读出向量 | Qwen3-VL-8B | 选部件有，读向量新 | 公开评测（第 3.4 节） |
| R3a 问答头 | 自由回答 | VLM | 没做 | — |

下面一块一块讲。例子都来自场景 421069 的真实数据，图上英文标注写明了哪些是标注（ORACLE）、哪些是上限。
""")

md(r"""
## 2.3 W1 关键帧

**做什么：** 一段视频有上千帧，记忆只存在其中 40 张上。挑的是最稳的帧：相机转得慢、画面不糊。

**输入 → 输出：** 整段视频和每帧的位姿 → 40 张关键帧（图像、深度、相机）。

**这一周的发现：** 关键帧里一次都没出现的部件，记忆里就没有。三个检验里，这样的部件占 15% 到 21%（见下面的数）。用向量帮忙挑关键帧（你的 M2）之前，要先保证关键帧能覆盖部件。
""")

code(r"""
fig = H.keyframe_strip(frames["lidar"], labels, times)
display(fig)
plt.close(fig)
for name in ("s0_421069", "s0b_val3", "s0c_gravity_val3"):
    c = H.run_json(name)["counts"]
    print(f"{name:18s} parts {c['elements']:3d}, never in the 40 keyframes {c['not_seen']:2d} ({100 * c['not_seen'] / c['elements']:.0f}%)")
""")

md(r"""
## 2.4 W2 部件掩码

**做什么：** 在每张关键帧上，把能操作的小部件（拉手、旋钮、开关、插座……）分出来。

**用什么：** SAM 3。给它一个词，比如 handle，它就在图里把对应的东西框出来、分出来。

**现在的状态：**
- 上周在 30 个场景上，SAM 3 在 40 张关键帧里找到了 74.8% 的部件。
- 这一周的向量检验用的是**标注的掩码**（ORACLE，下图左边两张）。这样只检验"有了部件以后，向量算得对不对"，不把部件漏检的误差混进来。
- 再下面一张是上周 pipeline 的真实例子（场景 421254）：SAM 3 找到的部件编上号，让模型选；蓝框是模型选的。
""")

code(r"""
picks = sorted(range(len(labels)), key=lambda k: -int((labels[k] >= 0).sum()))[:2]  # the two keyframes with the most part pixels
fig = H.masks_figure(scene, frames["lidar"], labels, sorted(picks))
display(fig)
plt.close(fig)
display(Image(filename=str(FIG / "pipeline" / "pipeline_answer_zoom.jpg"), width=300))
print("scene 421254 (week-1 pipeline): SAM 3 parts, numbered for the model; blue = the box the 8B model picked")
""")

md(r"""
## 2.5 W3 深度和位姿

**做什么：** 知道每个像素离相机多远、相机在世界里的哪里。有了这两样，图上的一个点就能变成世界里的一个三维点。

**用什么：** 手机自带的 LiDAR 深度和 ARKit 位姿，都是设备直接给的，不用模型。没有 LiDAR 的设备再用单目深度模型（Depth Anything V2）。

**这一周的发现：**
- LiDAR 和"从激光扫描渲染出的深度"（上限）算出的向量差不多好，所以推荐 LiDAR 优先。
- 但少数部件没有 LiDAR 深度：玻璃、很细的把手、太远的地方。下图中间，白圈标出这一帧里没有 LiDAR 深度的部件。
""")

code(r"""
k = max(range(len(labels)), key=lambda k: int((labels[k] >= 0).sum()))  # the keyframe with the most part pixels
fig, blind, total = H.depth_figure(frames["lidar"][k], frames["rendered"][k], labels[k])
display(fig)
plt.close(fig)
print(f"keyframe #{k + 1}: {blind} of {total} parts have no LiDAR depth at any pixel")
for name, visits in (("s0b_val3", ("422842", "467115", "466916")), ("s0c_gravity_val3", ("466880", "466437", "466841"))):
    rows = [r for v in visits for r in H.run_json(f"{name}/{v}", "parts.json") if (r["depth"], r["rule"], r["frames"]) == ("lidar", "c1", "best")]
    print(f"{name:18s} {sum(r['vector'] is None for r in rows)} of {len(rows)} scored parts without a LiDAR vector")
""")

md(r"""
## 2.6 W4 动作类型和"装在什么上"

**做什么：** 判断每个部件是推、拉还是转，并且它装在什么上：门、抽屉、固定的面板，还是烤箱门、翻盖。

**为什么要问"装在什么上"：** 下图是 30 个验证场景里所有有运动标注的部件。9 类标签里有 7 类，看标签就知道是平移还是转动。但两类"拉"不行：勾着拉的有将近一半是门（转动），捏着拉的约三成是门。所以：
- 只看标签，运动类型最多对 80%；
- 靠几何猜（看拉手在门面上偏不偏），在抽屉多的房间里反而更差（第 3.3 节）；
- 所以要问一句"这个拉手装在什么上"。

**现在的状态：** 类型来自标注的标签；"装在什么上"还没做，是下一步要在服务器上做的检验（第 4 节决定 1）。
""")

code(r"""
visits = (ROOT / "third_party" / "scenefun3d" / "benchmark_file_lists" / "val_scenes.txt").read_text().split()
pairs = Counter(pair for visit in visits for pair in CLM.label_motions(ROOT / "data" / "scenefun3d", visit))
fig = H.label_motion_figure(pairs)
display(fig)
plt.close(fig)
best = sum(max(pairs[(l, "trans")], pairs[(l, "rot")]) for l in {l for l, _ in pairs})
print(f"{len(visits)} scenes, {sum(pairs.values())} parts; label-only motion type at best {best}/{sum(pairs.values())} = {100 * best / sum(pairs.values()):.1f}%")
""")

md(r"""
## 2.7 W5 每个部件的 affordance 向量

**做什么：** 给每个部件算一条三维向量，存进记忆。现在存的字段：

| 字段 | 意思 | 怎么算 |
| --- | --- | --- |
| 接触点 | 手碰的那个点（x, y, z） | 部件掩码中心的像素 + 深度 + 位姿 |
| 法线 | 接触面朝哪边 | 部件周围的深度拟合一个平面 |
| 类型 | 平移还是转动 | 现在：标签（+ 几何判断门还是抽屉）；以后：问"装在什么上" |
| 方向或转轴 | 推拉的方向，或转轴的方向；存成没有正负号的线 | 按的：沿法线往里；拉的：往外；旋钮：轴 = 法线；门窗：轴 = 重力方向 |
| 原点 | 转轴经过的点 | 旋钮：接触点；门窗：门面水平方向上离把手远的那一端（临时，以后用门板掩码） |
| 证据帧 | 哪些关键帧看得到它 | 写入时记下 |

**计划加的字段（11-20 对照 VLA 的 map 后加的，还没实现）：** 接近方向（法线反过来）、抓握轴（手指闭合的方向，主要对长条把手有用）、接触区域（部件的一片点和分数）。

**为什么说它看得懂：** 每个字段都能直接画出来（第 1 节的视频和图就是这样画的）。错了也能看出错在哪：类型错了、点偏了，还是轴歪了。

下面是记忆里这个场景的部件表（前 12 行）。坐标在设备的世界坐标里（ARKit，z 朝上，单位米）；接给机器人时，只要再乘一个刚体变换。
""")

code(r"""
display(H.part_table_frame("demo_421069_gravity"))
""")

md(r"""
## 2.8 读取：R1 问题解析、R2 挑关键帧、R3 两个头

这一周没有改读取这一侧，状态如下：

- **R1 问题解析（没做）：** 从问题里拆出物体、部件、想要的动作类型。打算用现成的大语言模型，和 Fun3DU、UniFunc3D 一样。
- **R2 按功能挑关键帧（部分有）：** 关键帧得分 = 语义相似度 + 部件得分。上周的结果：在标注的部件图上，第一张就挑对的比例，按功能存是 66%，只按整张画面是 46%。
- **R3b 动作头（选部件有，读向量是新的）：** 模型在编号框里选一个部件，再从部件表里读出它的向量。例子就是 W2 里那张图：任务是"打开电视左边柜子左上角的抽屉"，8B 模型选了 10 号（蓝框），选对了。公开评测（第 3.4 节）评的就是"选部件"这一步。
- **R3a 问答头（没做）：** 给模型看挑出的关键帧和每个编号一行简短记录，让它自由回答。计划加一项检验：故意改一条记忆，看回答跟不跟着变，确认它在用记忆。
""")
