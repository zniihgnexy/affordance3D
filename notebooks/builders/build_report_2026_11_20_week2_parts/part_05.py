# Cells of build_report_2026_11_20_week2.py, part 5 (main-idea check, decisions, commands, number checks).

md(r"""
---
# 4. 主旨核对和需要你决定的事

## 4.1 主旨核对（这一周整体）

| 你的主旨 | 这一周 |
| --- | --- |
| M1 可解释的三维向量，能给机器人用 | **符合。** 每个部件一条三维向量，在世界坐标里，视频和点云里都能画出来（第 1 节）。下一步加抓握轴，每条记录就能解码成"抓哪里、怎么抓、抓住以后怎么动"。 |
| M2 用向量帮忙挑关键帧 | **没覆盖。** 相关的发现：40 张关键帧漏掉 15% 到 21% 的部件，要先保证关键帧能覆盖部件。 |
| M3 affordance 加深度算出动作向量 | **符合**，三轮检验测的就是这一步。但见下面第一条提醒。 |
| M4 两个头 | **部分。** 动作头的"选部件"在公开评测上评了（第 3.4 节）：用 SAM 3 的部件图时明显落后于 UniFunc3D 8B，用标注的部件图时和它差不多。读向量和问答头还没做。 |
| M5 现成模型加规则，不训练 | **符合。** 这一周全是规则和现成模型，没有训练。 |
| M6 用尽量少的资源 | **符合。** 向量检验每个不到一分钟 CPU；两个视频也只用 CPU；公开评测的 GPU 部分 11-17 走队列跑完，后两步只用 CPU。 |

**提醒：**
1. **偏移：只有 affordance 标签加深度不够。** 三轮检验都指向这一点：门还是抽屉，要知道拉手"装在什么上"。记忆里每个部件最好也存这一项，这会改变 affordance map 存什么（决定 2）。
2. **可能的理解差异："affordance map"可以指两种东西。** 一种是网络预测的逐像素稠密图（VLA 那边就是这种），一种是这边逐部件的记录。逐部件的记录需要时也能投影成逐像素的图（决定 3）。
3. **门窗的动作向量现在只有方向可靠，位置不可靠。** 机器人开门时，转动半径取决于门轴在哪里，要等门板掩码。
4. **这一周的向量检验都用标注的部件掩码。** 公开评测说明，部件由 SAM 3 自己找时，差距主要就出在这一步。
""")

md(r"""
## 4.2 需要你决定的事（聊天里也会列）

1. **服务器上的下一步：用 SAM 3 门板掩码，再问 VLM"这个拉手装在什么上"，在 30 个验证场景上检验。** 要用 GPU，排队后不到 1 小时。
   - (a) 我先把方案写好给你看：场景、提问方式、打分方法、事先的预测。你确认后再排队。
   - (b) 写好方案就直接排队，排在你自己的 vid_tokenizer 作业前面。

   我推荐 **(a)**：这是第一个要用 VLM 回答"装在什么上"的检验，问法会直接影响结果。
2. **记忆里每个部件要不要也存"装在什么上"（门、抽屉、固定面板、烤箱门或翻盖）？**
   - (a) 存，affordance map 每个部件多一个字段。
   - (b) 不存。那样拉手的运动类型最多对 80%。

   我推荐 **(a)**。
3. **记忆里的 affordance map 用哪种形式？**
   - (a) 逐部件的记录（现在的做法），需要时投影成逐像素的图。不用训练。
   - (b) 由网络预测的逐像素稠密图。要训练。

   我推荐 **(a)**：M5 是先不训练；投影出来的图也能交给需要这种输入的下游。
4. **S0d 做不做：** 在 S0b、S0c 的 6 个场景上补算抓握轴和接触区域，加上接触点和失败归类的指标，重新打分。只用 CPU，几分钟。
   - (a) 做，放在服务器那一步之前。
   - (b) 不做，等服务器那一步时一起算。

   我推荐 **(a)**。

5. **公开评测"明显落后"以后怎么走？**（事先写定的读法要求停下来讨论，见第 3.4 节）
   - (a) 接受落后，不把 Task 2 的准确度当卖点，如实报告；继续新 pipeline。
   - (b) 先只用 CPU 看 SAM 3 漏掉的是哪些部件：按标签、大小、离相机多远分开数。用已有的结果，几分钟。再决定要不要花 GPU 补部件图。
   - (c) 直接花 GPU 补部件图（换提示词、多帧合并），争取追上。不训练。

   我推荐 **(b)**：不花 GPU，先弄清差在哪类部件。部件找不找得到，新 pipeline 也绕不开：Task 3 也要先定位对。

**完整的背景和理由**在 `docs/mem/新pipeline_差异_文献_推荐_2026-11-18.md`：第 9 节（S0c）、第 10 节（对照 VLA 的 map）和第 11 节（公开评测）。
""")

md(r"""
---
# 附录

## A. 命令（都在仓库根目录，服务器上跑；除公开评测的 GPU 一步外都只用 CPU）

| 内容 | 命令 |
| --- | --- |
| S0、S0b、S0c | `python scripts/run_part_vectors.py --config configs/part_vectors/<s0_421069 / s0b_val3 / s0c_gravity_val3>.yaml` |
| 门窗诊断 | `python scripts/diagnose_door_hinges.py --config configs/part_vectors/<s0b_val3 / s0c_gravity_val3>.yaml` |
| 演示用的向量（场景 421069） | `python scripts/run_part_vectors.py --config configs/part_vectors/demo_421069_gravity.yaml` |
| 两个视频 | `python scripts/make_week2_media.py --config configs/report/week2_media.yaml` |
| 标签和运动类型、部件形状 | `python scripts/count_label_motions.py --scenes third_party/scenefun3d/benchmark_file_lists/val_scenes.txt`；`count_part_shapes.py` 同样的参数 |
| 公开评测 | `python scripts/run_public_eval.py --config configs/public_eval/split0_8b.yaml --stage <queue / lift / score --official>` |
| 这份 notebook | `python notebooks/builders/build_report_2026_11_20_week2.py report/2026-11-20_week2_summary.ipynb`，再用 `jupyter nbconvert --to notebook --execute --inplace` 执行 |

**文档：** 推荐 pipeline 和三轮检验 `docs/mem/新pipeline_差异_文献_推荐_2026-11-18.md`；理解差异 `docs/mem/理解差异记录.md`；实验索引 `EXPERIMENTS.md`。

## B. 数字核对

正文里引用的数字，下面逐个从结果文件里重新算一遍；对不上时这一格会报错。
""")

code(r"""
s0b_doors = H.run_json("s0b_val3", "door_hinges.json")["counts"]["lidar"]
checks = [
    check("S0c parts", s0c["counts"]["elements"], 80), check("S0c scored", s0c["counts"]["scored"], 64),
    check("S0c never in keyframes", s0c["counts"]["not_seen"], 12),
    check("S0c true door axes vertical", doors["true_axis_vertical"], 16), check("S0c doors", doors["doors"], 19),
    check("S0c oracle_kind door +MA %", round(off["oracle_kind"]["ma"]), 68), check("S0c oracle_gravity door +MA %", round(off["oracle_gravity"]["ma"]), 84),
    check("S0c oracle_kind door +MAO %", round(off["oracle_kind"]["mao"]), 16), check("S0c oracle_gravity door +MAO %", round(off["oracle_gravity"]["mao"]), 21),
    check("S0b doors", s0b_doors["doors"], 44), check("S0b true axes vertical", s0b_doors["true_axis_vertical"], 42),
    check("S0b old rule axis right", s0b_doors["rule"]["axis_ok"], 21), check("S0b vertical+handle axis right", s0b_doors["up_handle"]["axis_ok"], 39),
    check("S0b side given, axis+origin", s0b_doors["up_right_end"]["axis_and_origin_ok"], 21),
    check("label-only ceiling %", round(100 * best / sum(pairs.values()), 1), 80.3),
    check("walkthrough frames", media["walkthrough"]["frames"], 1025),
]
mine = {rule: [r for r in rows_c if (r["depth"], r["rule"], r["frames"]) == ("lidar", rule, "best") and drawer(r)] for rule in ("c1", "geometric")}
checks += [check("S0c drawers", len(mine["c1"]), 35), check("S0c drawers right, label only", sum(r["score"]["m"] for r in mine["c1"]), 31),
           check("S0c drawers right, geometric", sum(r["score"]["m"] for r in mine["geometric"]), 20)]
print(f"{len(checks)} numbers checked, all match the result files")
""")
