# Cells of build_report_2026_11_20_week2.py, part 6 (appendix: commands and number checks).

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
| 这份 notebook | `python notebooks/builders/build_report_2026_11_20_week2.py report/week2_2026-11-16_to_11-22.ipynb`，再用 `jupyter nbconvert --to notebook --execute --inplace` 执行 |

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
    check("oracle_chosen AP25", round(pe["means"]["oracle_chosen"]["AP25"], 1), 51.5),
]
mine = {rule: [r for r in rows_c if (r["depth"], r["rule"], r["frames"]) == ("lidar", rule, "best") and drawer(r)] for rule in ("c1", "geometric")}
checks += [check("S0c drawers", len(mine["c1"]), 35), check("S0c drawers right, label only", sum(r["score"]["m"] for r in mine["c1"]), 31),
           check("S0c drawers right, geometric", sum(r["score"]["m"] for r in mine["geometric"]), 20)]
print(f"{len(checks)} numbers checked, all match the result files")
""")
