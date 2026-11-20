# Cells of build_report_2026_11_20_week2.py, part 5b (stage 0: results, re-analysis, recommendations; the notes cell).

md(r"""
---
# 6. 阶段 0：五个零成本测试的结果和重新分析（11-20）

计划和五个测试的预测都在跑之前写定（`docs/mem/新pipeline_差异_文献_推荐_2026-11-18.md` 第 13 节，配置在 `configs/stage0/` 和 `configs/part_vectors/t3_val30.yaml`）。全部只用 CPU，没有用 GPU。T1、T2 用公开评测已有的结果，什么都没重跑。

| 测试 | 结果 | 事先的预测 | 按事先的读法 |
| --- | --- | --- | --- |
| **T1** Task 2 输在哪一步（445 条任务，SAM 3 部件图） | 对 111 条（24.9%）；目标不在 40 张关键帧里 54 条（12.1%）；SAM 3 在所有帧里都没找到 100 条（22.5%）；找到了但 VLM 没选中 148 条（33.3%）；选中了但投到扫描上没对上 32 条（7.2%）。标注部件图对、SAM 3 部件图错的 105 条里：没选中 44 条、没找到 41 条、没对上 19 条 | 三条成立一条：最多的是"没选中"而不是"没找到"（44 对 41，很接近）；没找到的部件并不明显更小（面积中位数 90 对 113 像素）；不同标签的漏检率相差 51 个点（成立） | 差距主要在读取（挑帧、选部件）和投到扫描上，不全在部件图 |
| **T2** 记忆比直接看原视频强在哪 | 全部 +5.6（区间 0.2 到 11.0）；小部件 +10.3（3.3 到 18.5），大部件 +4.1（−4.9 到 13.4）；场景里有同类部件 +6.6（0.7 到 12.7），只有一个 −1.9（−14.0 到 10.9，只有 52 条） | 两条都成立 | 支持"长视频里找小部件、相似部件多"是记忆最占优的任务；而且记忆只读 3 帧，原视频读 40 帧 |
| **T3** 给定部件时动作向量对不对（30 个场景，标注掩码） | 561 个部件里 78 个（13.9%）从没出现在关键帧里；其余 471 个：类型用标签时 +M 68.4、+MA 48.8、+MAO 38.4；类型给对时 93.2、71.5、53.9。门这类绕板边转的 143 个，类型对时轴对 82.5%，原点只对 24.5% | 四条都成立 | 高于唯一公开参照的粗折算（+MA 38.5、+MAO 30.6）→ Task 3 保留为候选 |
| **T4** Task 3 能不能公平地比（查资料） | 官方没有 Task 3 的打分代码（文档写的是 "Coming soon"），评测服务器没有 Task 3，测试集的运动标注不公开；验证集上没人报过 +M、+MA、+MAO。最接近的是 AffordBot：在同样 30 个验证场景上按指令报 AP25 23.3、加类型 18.3、加类型和方向 10.8（方向只分 6 类）；它报的 Fun3DU 加运动是 18.7、11.5、4.0 | 不预测 | 要自己实现指标（照 OPD 的做法：轴不分正负，原点算到轴线的距离，平移不比原点），另外按 AffordBot 的口径再比一次 |
| **T5** OpenEQA 问不问部件 | 1636 道题，关键词标出 12.9%：物体状态类 44.8%，功能推理只有 4.1%，物体定位 3.8%。人工抽查 60 题：标出的 30 题里 16 题真和部件有关（多是"门、抽屉、窗、马桶盖开着吗"），没标出的 30 题里 2 题有关。校正后，三类合计 15.1% | 两条都成立 | 部件题只作为辅助结果；OpenEQA 整体只用来证明"别的题没退步" |

**一处说明：** T1 的配置里把"对"写成了"和目标的 IoU 至少 0.25"，同时写明是"公开评测打分器的 AP25"。那个打分器的 AP25 实际是"预测点里落在目标上的比例至少 0.25"（Fun3DU 的口径）。脚本用的是打分器的定义，所以"对"和公开评测完全一致；这一点在提交说明里也写了。
""")

code(r"""
S0 = ROOT / "runs" / "stage0"
t1 = json.loads((S0 / "task2_losses" / "summary.json").read_text())
t3 = H.run_json("t3_val30", "summary.json")
t5 = json.loads((S0 / "openeqa_fit" / "summary.json").read_text())

stage_names = {"hit": "hit", "not_in_keyframes": "not in keyframes", "never_found": "SAM 3 never found",
               "not_chosen": "found, VLM chose another", "lost_in_mask": "chosen, mask missed"}
colours = ["#4c9a2a", "#9e9e9e", "#d1495b", "#edae49", "#00798c"]
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1.2]})
for row, (title, counts) in enumerate([("all 445 tasks", t1["stages_memory_arm"]),
                                       ("105 tasks annotated maps get right", t1["lost_to_annotated"]["stages"])]):
    left, total = 0, sum(counts.values())
    for (stage, label), colour in zip(stage_names.items(), colours):
        width = 100 * counts[stage] / total
        axes[0].barh(row, width, left=left, color=colour, label=label if row == 0 else None)
        if width > 6:
            axes[0].text(left + width / 2, row, f"{counts[stage]}", ha="center", va="center", color="white", fontsize=9)
        left += width
axes[0].set_yticks([0, 1], ["all 445 tasks", "annotated right,\nSAM 3 wrong (105)"]); axes[0].invert_yaxis()
axes[0].set_xlabel("% of tasks"); axes[0].legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=3)
axes[0].set_title("T1  where the memory arm loses Task 2")
labels_cmp = t1["predictions"]["T1-P3"]["labels_compared"]
order = sorted(labels_cmp, key=labels_cmp.get)
axes[1].barh(order, [100 * labels_cmp[k] for k in order], color="#d1495b")
axes[1].set_xlabel("% of visible parts SAM 3 never finds"); axes[1].set_title("T1  misses by label (>= 20 parts)")
strata = [("all", "all"), ("small", "small parts"), ("large", "large parts"), ("several_of_label", "look-alikes in scene"),
          ("one_of_label", "only one of its label")]
for i, (key, name) in enumerate(strata):
    s = t1["memory_minus_raw"][key]
    axes[2].errorbar(s["diff"], i, xerr=[[s["diff"] - s["low"]], [s["high"] - s["diff"]]], fmt="o", color="#00798c", capsize=3)
    axes[2].text(s["high"] + 0.8, i, f'{s["diff"]:+.1f} (n={s["tasks"]})', va="center", fontsize=8)
axes[2].axvline(0, color="grey", lw=0.8); axes[2].set_yticks(range(len(strata)), [n for _, n in strata]); axes[2].invert_yaxis()
axes[2].set_xlabel("AP25 hit, memory (3 frames) minus raw video (40 frames), points"); axes[2].set_title("T2  where the memory wins")
plt.tight_layout(); plt.show()

g, og = t3["variants"]["lidar/gravity/best"], t3["variants"]["lidar/oracle_gravity/best"]
s0_checks = [
    check("T1 hit", t1["stages_memory_arm"]["hit"], 111), check("T1 not in keyframes", t1["stages_memory_arm"]["not_in_keyframes"], 54),
    check("T1 never found", t1["stages_memory_arm"]["never_found"], 100), check("T1 not chosen", t1["stages_memory_arm"]["not_chosen"], 148),
    check("T1 lost in mask", t1["stages_memory_arm"]["lost_in_mask"], 32), check("T1 lost to annotated", t1["lost_to_annotated"]["tasks"], 105),
    check("T1 lost: not chosen", t1["lost_to_annotated"]["stages"]["not_chosen"], 44),
    check("T1 lost: never found", t1["lost_to_annotated"]["stages"]["never_found"], 41),
    check("T1 lost: mask", t1["lost_to_annotated"]["stages"]["lost_in_mask"], 19),
    check("T1 spread", round(100 * t1["predictions"]["T1-P3"]["spread"]), 51),
    check("T2 all", round(t1["memory_minus_raw"]["all"]["diff"], 1), 5.6), check("T2 small", round(t1["memory_minus_raw"]["small"]["diff"], 1), 10.3),
    check("T2 several", round(t1["memory_minus_raw"]["several_of_label"]["diff"], 1), 6.6),
    check("T3 not seen", t3["counts"]["not_seen"], 78), check("T3 scored", t3["counts"]["scored"], 471),
    check("T3 gravity +MA", g["all"]["ma"], 48.8), check("T3 gravity +MAO", g["all"]["mao"], 38.4),
    check("T3 oracle_gravity +MA", og["all"]["ma"], 71.5), check("T3 oracle_gravity +MAO", og["all"]["mao"], 53.9),
    check("T3 door axis", og["off_part_rotations"]["ma"], 82.5), check("T3 door origin", og["off_part_rotations"]["mao"], 24.5),
    check("T5 flagged %", round(100 * t5["table"]["all"]["all"]["share"], 1), 12.9),
    check("T5 corrected focus %", round(100 * t5["focus_corrected_share"], 1), 15.1),
    check("T3 label-only +M (c1)", t3["variants"]["lidar/c1/best"]["all"]["m"], 74.3), check("T3 gravity +M", g["all"]["m"], 68.4),
]
carry = json.loads((ROOT / "runs" / "mem_phase1" / "tool_maps_summary.json").read_text())["tools"]["sam3"]["carry_ceiling"]
s0_checks += [check("found on some keyframe %", round(100 * carry["found_anywhere"]["rate"], 1), 74.4),
              check("shown to the VLM %", round(100 * carry["coverage"]["rate"], 1), 58.7),
              check("carry gain points", round(100 * carry["carried_minus_coverage"]["rate"], 1), 3.1)]
print(f"{len(s0_checks)} stage-0 numbers checked against the result files")
""")

md(r"""
## 6.2 重新分析：框架的几个决定，我的推荐

**A. v3 是否正式作为主线：推荐采用，并冻结接口。**
- 30 个场景上，给定部件时，向量的 +MA 是 48.8、+MAO 是 38.4，高于唯一公开参照（训练过的模型）的粗折算。
- 记忆在预测的地方赢了直接看原视频：小部件 +10.3，场景里有同类部件 +6.6，而且只读 3 帧。
- "两个头读同一份部件表"的设计，没有哪项结果推翻它。

**决定 2：每个部件存不存"装在什么上"：推荐 (a) 存，证据比上次更强。**
- 类型是向量最大的误差：类型给对以后，+MA 从 48.8 升到 71.5，多了 22.7 个点。
- 靠几何猜"门还是抽屉"，类型反而比只看标签更差（+M 68.4 对 74.3）。
- "装在什么上"正是用来定类型的。它还顺带记下部件属于哪个物体（门、抽屉、柜子），按物体搜就是按这个字段搜。

**决定 3 和 6：affordance map 的形式。推荐 3 选 (a)：以逐部件的记录为核心，再加一个"可见性列表"字段。整屋的三维表面图（决定 6）先不做，留作阶段 5 的候选创新点。**
- 三维表面图对"搜索"最实在的好处，是按"哪一帧看得清"来挑帧。T1 里"找到了但没选中"是最大的一类损失，其中一部分就是目标所在的帧没被挑出来给 VLM 看：SAM 3 在某张帧上找到目标的任务占 74.4%，但被挑出来给 VLM 看到的只有 58.7%。这件事只要给每个部件存一个列表就能做：在哪些帧里看得见、有多大、离多远、正不正。不需要融合整屋的表面。
- 三维表面图解决不了另外两类损失：
  - SAM 3 在所有帧里都没找到（22.5% 的任务）。把找到的部件带到别的视角，覆盖只多 3.1 个点；要靠提示词和检测来补，漏得最多的是转的旋钮、按键和门把手。
  - 目标根本不在 40 张关键帧里（12.1% 的任务）。要回到原视频里去找帧。
- 现在的候选 bench（Task 2、Task 3、OpenEQA 的部件题）都用不到整屋几何。只有 VSI 那类空间题用得到，而它不考 affordance。
- 这样也不用推翻 11-11 第 3 轮"不融合稠密几何"的决定。
- **这和你 11-20 提的三维表面图想法有出入，需要你确认。** 我的建议是：它的核心好处（按可见性挑帧）现在就以"可见性列表"的形式放进框架；整屋表面图本身，等阶段 3 看挑帧是不是仍是瓶颈，或者要做"站不站得过去、门开不开得开"这类可行性问题时，再按事先写定的检验加上去。

**决定 5（公开评测落后以后怎么走）：诊断已经做完（T1）。推荐接受现在落后，先不花 GPU。**
到阶段 5 再按 T1 的分解定改进的顺序：
1. 读取：挑帧和选部件，占差距的 42%；
2. 几类标签的漏检，占 39%；
3. 掩码，占 18%。

**决定 1（服务器上用门板掩码、问 VLM"装在什么上"）和决定 4（S0d）：推荐都并进阶段 2 的跑通。**
- W4 最简单的真实版本，就是问 VLM"装在什么上"。方案先写好给你看，即决定 1 选 (a)。
- S0d 要的接触点和失败归类指标，并进 Task 3 的打分代码。

## 6.3 bench 适配表（初步；阶段 3 会用真实输入再测一遍）

| bench 或指标 | 适配 | 依据 | 我们的数据现在多强 |
| --- | --- | --- | --- |
| SceneFun3D Task 3（自己实现 OPD 式的指标），另按 AffordBot 的口径比一次 | **适合，最可能是主卖点** | T3、T4 | 给定部件时高于参照；没有现成的对手数字，要自己算 |
| 联合指标：部件找对并且动作对 | **适合，只有我们能报** | 同一批任务两种标注都有 | 还没测 |
| SceneFun3D Task 2 | 适合，是"找部件"唯一有公开对手的比较，但现在落后 | T1 | 落后 UniFunc3D 8B 19.1 个点，差距已分解 |
| 读得少（M2，Task 2 上 3 帧对 40 帧） | **适合** | T2 | +5.6；小部件 +10.3 |
| OpenEQA | 只当辅助 | T5 | 部件题约 15%，多是开关状态 |
| VSI-Bench | 不适合：不考 affordance | — | — |
| OPDMulti | 没测；输入是单张图，可以作为动作向量的补充 | — | — |

## 6.4 如果你同意，框架冻结时要改的接口

部件记录加两个字段：
- **装在什么上：** 门、抽屉、固定面板、翻盖等；
- **可见性列表：** 帧、面积、距离、角度。

如果要做 OpenEQA 的部件题，再加一个可选字段：**看到时的状态**（开或关）。其余接口不变。

## 6.5 主旨核对

| 你的主旨 | 阶段 0 |
| --- | --- |
| M1 可解释的三维向量 | **符合。** 30 个场景上给定部件时，向量高于公开参照 |
| M2 帮忙挑关键帧 | **符合。** 记忆在小部件、相似部件多时赢原视频；可见性列表是具体做法 |
| M3 affordance 加深度算出动作向量 | **符合。** T3；最大的误差是类型，靠"装在什么上"补 |
| M4 两个头 | **部分。** 问答头还没做；OpenEQA 只能当辅助，问答头要另找能体现部件的题 |
| M5 不训练 | **符合** |
| M6 资源少 | **符合。** 阶段 0 全部只用 CPU |

**和你的想法有出入的一处：** 决定 6 我推荐暂缓整屋三维表面图（理由见 6.2），需要你确认。
""")

md(r"""
## ✏️ 你的批注

这一周的意见和更新写在这一格里，我读到以后逐条回复。上一周的 pipeline 工作文档已并入第 1 周 notebook 第 8 部分；以后的批注都写在当周 notebook 的这一格里。

（空）
""")
