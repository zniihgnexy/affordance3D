# Cells of build_report_2026_11_20_week2.py, part 3 of 3 (results of the week; the public evaluation is in part 4).

md(r"""
---
# 3. 这一周的结果

## 3.1 规则算出的三维向量：S0、S0b、S0c

三个检验问的是同一个问题：**有了部件掩码（标注）以后，规则加深度能不能算出对的三维向量？** 都只用 CPU，每个不到一分钟。

| 检验 | 场景 | 做法 |
| --- | --- | --- |
| S0 | 421069（1 个） | 第一次试：规则行不行 |
| S0b | 422842、467115、466916（门最多的 3 个验证场景，事先按规则选） | 换场景看 S0 的结论还成不成立；事后诊断门窗错在哪 |
| S0c | 466880、466437、466841（门第 4 到第 6 多的验证场景，事先选） | 检验事后看出的"门轴竖直"：规则和预测在跑之前提交 |

**怎么读下面的表：** 每格是三个百分比，"+M / +MA / +MAO"：
- **+M**：运动类型（平移还是转动）对了；
- **+MA**：类型对，并且轴或方向和真值的夹角在 15° 以内（只比直线，不比正反）；
- **+MAO**：再加上转动的原点离真轴不到 0.25 m（平移没有原点，这一项同 +MA）。

这是 SceneFun3D Task 3 论文里的定义。下面都是 LiDAR 深度、每个部件用画面最大的一帧。

列的意思：
- **c1**：只看标签；
- **geometric**：标签，再加上按拉手在门面上的位置猜门还是抽屉；
- **gravity**：geometric，但门轴取重力方向（S0c 新加）；
- **oracle**（类型给对）：用标注的运动类型。真实系统里没有它，用来看"类型对了以后，几何算得准不准"。
""")

code(r"""
BASE = {"c1": "lidar/c1/best", "geometric": "lidar/geometric/best", "oracle_kind": "lidar/oracle_kind/best"}
NEW = {"c1": "lidar/c1/best", "geometric": "lidar/geometric/best", "gravity（新）": "lidar/gravity/best",
       "oracle_kind": "lidar/oracle_kind/best", "oracle_gravity（新）": "lidar/oracle_gravity/best"}
for title, name, variants in (("S0：场景 421069", "s0_421069", BASE), ("S0b：3 个门多的验证场景（合并）", "s0b_val3", BASE),
                              ("S0c：3 个新的验证场景（合并）", "s0c_gravity_val3", NEW)):
    print(title)
    display(H.rates_table(name, variants))
""")

md(r"""
**读法：**
1. **按、插的部件：** 三轮都在一半到八成之间。规则加深度对这类部件够用。
2. **旋钮：** 很差，因为旋钮周围的表面法线估不准（旋钮很小，周围常是斜面或曲面）。
3. **门窗类：** 只看标签几乎全错，因为"拉"在标签里分不出门和抽屉。类型给对以后，S0c 的新规则（oracle_gravity）把 +MA 从 68% 提到 84%，但 +MAO 只有 21%：门轴的位置还是找不准。
4. **全部部件：** S0c 里 geometric 的 +M 比只看标签还低，原因见 3.3。
""")

md(r"""
## 3.2 S0c：事先写好的四条预测

预测和采用规则写在 `configs/part_vectors/s0c_gravity_val3.yaml` 里，在运行前提交（提交 9cd8daf）。下面的数直接从结果文件读出来。
""")

code(r"""
s0c = H.run_json("s0c_gravity_val3")
doors = H.run_json("s0c_gravity_val3", "door_hinges.json")["counts"]["lidar"]
off = {rule: s0c["variants"][f"lidar/{rule}/best"]["off_part_rotations"] for rule in ("oracle_kind", "oracle_gravity")}
rows_c = [r for v in ("466880", "466437", "466841") for r in H.run_json(f"s0c_gravity_val3/{v}", "parts.json")]
key = lambda r: (r["visit"], r["depth"], r["frames"], r["part"])
by_rule = {rule: {key(r): r for r in rows_c if r["rule"] == rule} for rule in ("geometric", "gravity", "oracle_kind", "oracle_gravity")}
checked = differ = 0
for a_rule, b_rule in (("geometric", "gravity"), ("oracle_kind", "oracle_gravity")):
    for k, a in by_rule[a_rule].items():
        b = by_rule[b_rule][k]
        if a["label"] not in ("hook_pull", "pinch_pull", "hook_turn") or b["vector"] is None or b["vector"]["kind"] != "rot":
            checked, differ = checked + 1, differ + (a["score"] != b["score"])
p1 = 100 * doors["true_axis_vertical"] / doors["doors"]
p2 = off["oracle_gravity"]["ma"] - off["oracle_kind"]["ma"]
predictions = pd.DataFrame([
    ("P1", "真轴至少 80% 是竖直的", f"{doors['true_axis_vertical']}/{doors['doors']} = {p1:.0f}%", p1 >= 80),
    ("P2", "轴：oracle_gravity 的 +MA 比 oracle_kind 高至少 15 个点",
     f"{off['oracle_gravity']['ma']:.0f}% 对 {off['oracle_kind']['ma']:.0f}%，高 {p2:.1f} 个点", p2 >= 15),
    ("P3", "原点：oracle_gravity 的 +MAO 不低于 oracle_kind", f"{off['oracle_gravity']['mao']:.0f}% 对 {off['oracle_kind']['mao']:.0f}%",
     off["oracle_gravity"]["mao"] >= off["oracle_kind"]["mao"]),
    ("P4", "别的部件分数不变（构造上的检查）", f"{checked} 次比较，{differ} 处不同", differ == 0),
], columns=["编号", "预测", "结果", "成立"]).set_index("编号")
display(predictions)
""")

md(r"""
**结论：** 四条都成立，其中 P1、P2 刚好过线（P2 只差 3 个部件）。按事先定的规则，"门轴 = 重力方向"写进了推荐 pipeline。"原点 = 门面远端"也按规则写进去了，但标成临时：4 个对的原点里，只有 2 个来自"门面远端"。
""")

md(r"""
## 3.3 门窗：错在哪一步

左边是 S0b（事后诊断），右边是 S0c（新场景）。对每个门窗类部件，比较四种定门轴的办法：
- 旧规则：门面的边；
- 竖直轴，过把手；
- 竖直轴，过门面远端；
- 竖直轴，过两端里对的那一端：相当于告诉它门轴在哪一边，是这类办法的上限。

蓝色是"轴对"，橙色是"轴和原点都对"。
""")

code(r"""
fig = H.doors_figure({"S0b (after the fact)": "s0b_val3", "S0c (new scenes)": "s0c_gravity_val3"})
display(fig)
plt.close(fig)
drawer = lambda r: r["label"] in ("hook_pull", "pinch_pull") and r["true_kind"] == "trans"
for rule in ("c1", "geometric"):
    mine = [r for r in rows_c if (r["depth"], r["rule"], r["frames"]) == ("lidar", rule, "best") and drawer(r)]
    print(f"S0c drawers, rule {rule:9s}: type right {sum(r['score']['m'] for r in mine)} of {len(mine)}")
""")

md(r"""
**门窗的问题拆成三步，难度不一样：**
1. **门还是抽屉（类型）：** 只看标签最多对 80%。靠几何猜，在 S0b（门多）里有帮助，在 S0c（抽屉多）里反而把 11 个抽屉当成了门（上面最后两行）。所以要问"装在什么上"。
2. **门轴方向：** 用重力方向就修好了，两轮里大多数门轴都是竖直的。例外是烤箱门、翻盖这类水平轴，也要靠"装在什么上"来区分。
3. **门轴位置（原点）：** 最难。就算告诉它门轴在哪一边，也只对一半（S0b）到三分之一（S0c）。原因是靠深度找门面的边找早了：测到的门面宽度中位数只有约 0.3 m，比真门窄。要把整块门板分出来（SAM 3 门板掩码），这是下一步。
""")
