# Cells of build_report_2026_11_20_week2.py, part 4 (the public evaluation, discussion draft section 27.2).

md(r"""
## 3.4 公开评测：SceneFun3D Task 2，和 UniFunc3D 8B 比（445 条任务）

**测的是什么：** 上周的 pipeline 能不能在公开协议下"指出要操作的部件"。流程是：
1. SAM 3 建记忆；
2. 挑 3 张关键帧；
3. Qwen3-VL-8B 在编号框里选一个部件；
4. SAM 3 把选中的部件分出来；
5. 投到激光扫描上，变成扫描上的点集，和标注比。

**比的对象是 UniFunc3D 8B：** 同样大小的模型，每个问题看 64 帧。用他们公开的预测和我们同一套打分代码比。怎么读在跑之前就写定了（讨论稿 27.2）。

**这一周做了什么：** GPU 那一步 11-17 走队列跑完；这周五补完了"投到扫描上"和"打分"两步，只用 CPU。

**指标：** AP25 是"预测的点集和标注重合够多（IoU 至少 0.25）"的比例；mIoU 是平均重合度。都是百分比，越高越好。
""")

code(r"""
PE = ROOT / "runs" / "public_eval" / "public_eval_split0_8b"
pe = json.loads((PE / "summary.json").read_text())
cmp = {p.stem: json.loads(p.read_text()) for p in sorted((PE / "compare").glob("*.json"))}
METRICS = ("AP25", "mIoU", "AP50", "mAP")
ARMS = {"p4t_sam3": "记忆，SAM 3 建的部件图（P4，主要的一组）", "c3": "模型直接看 40 张均匀帧（c3）", "sam3q": "用任务文字直接提示 SAM 3",
        "p4n": "记忆，标注的部件图（上限）", "oracle_chosen": "投法上限：模型选的那一帧上，完美的掩码", "oracle_best": "投法上限：最好的那一帧上，完美的掩码"}
ref = cmp["p4t_sam3_vs_reference"]["first_minus_second"]
unifunc = {m: pe["means"]["p4t_sam3"][m] - ref[m]["rate"] for m in METRICS}  # the reference's own score, on the same tasks
rows = [{"组": name, **{m: round(pe["means"][arm][m], 1) for m in METRICS}} for arm, name in ARMS.items()]
rows.insert(1, {"组": "UniFunc3D 8B（他们公开的预测，我们的打分代码）", **{m: round(v, 1) for m, v in unifunc.items()}})
display(pd.DataFrame(rows).set_index("组"))

VERDICT = {"comparable": "相当", "behind": "落后", "clearly behind": "明显落后", "unclear": "说不清"}
PAIRS = {"p4t_sam3_vs_reference": "记忆（SAM 3）减 UniFunc3D 8B", "p4n_vs_reference": "记忆（标注部件图）减 UniFunc3D 8B",
         "p4t_sam3_vs_c3": "记忆（SAM 3）减 直接看原视频", "c3_vs_reference": "直接看原视频 减 UniFunc3D 8B",
         "sam3q_vs_reference": "直接提示 SAM 3 减 UniFunc3D 8B"}
diff = lambda d: f"{d['rate']:+.1f}（{d['low']:+.1f} 到 {d['high']:+.1f}）"
display(pd.DataFrame([{"成对比较（同样 445 条任务）": label, "AP25 差": diff(cmp[k]["first_minus_second"]["AP25"]),
                       "mIoU 差": diff(cmp[k]["first_minus_second"]["mIoU"]), "按事先的规则": VERDICT[cmp[k]["verdict"]]}
                      for k, label in PAIRS.items()]).set_index("成对比较（同样 445 条任务）"))
pe_checks = [check("P4 AP25", round(pe["means"]["p4t_sam3"]["AP25"], 1), 24.9), check("UniFunc3D 8B AP25", round(unifunc["AP25"], 1), 44.0),
             check("P4 minus UniFunc3D AP25", round(ref["AP25"]["rate"], 1), -19.1), check("p4n AP25", round(pe["means"]["p4n"]["AP25"], 1), 43.1),
             check("P4 minus c3 AP25", round(cmp["p4t_sam3_vs_c3"]["first_minus_second"]["AP25"]["rate"], 1), 5.6),
             check("oracle_best AP25", round(pe["means"]["oracle_best"]["AP25"], 1), 88.1)]
print(f"{len(pe_checks)} numbers of this section checked; differences in percentage points, 95% intervals from resampling the 30 scenes")
""")

md(r"""
**读法（跑之前写定的）：**
- **记忆（SAM 3 部件图）比 UniFunc3D 8B 明显落后。** AP25 低 19.1 个点（区间 −25.9 到 −12.4），mIoU 低 10.4 个点（−14.2 到 −7.0）。两个区间都整个低于门槛（AP25 −10，mIoU −5），按规则是"明显落后"。规则说这时要停下来讨论：论文不能靠准确度，只能看代价站不站得住（第 4 节决定 5）。
- **差距主要在部件图。** 换成标注的部件图（上限），AP25 是 43.1，和 UniFunc3D 8B 的 44.0 分不出高低（差 −0.9，区间 −7.0 到 6.4）。SAM 3 只找到约四分之三的部件（74.8%），没找到的部件，后面再好也选不对。
- **投到扫描上这一步损失不大。** 如果在最好的那一帧上有完美的掩码，AP25 能到 88.1。
- **记忆比"直接看原视频"好一点。** AP25 高 5.6 个点（0.2 到 11.0），mIoU 差不多，按规则是"相当"。
- **官方打分器的参照分数没算出来。** 它检查一个任务时，发现我们导出的点数和标注对不上，报错停了。上面的数都来自和 UniFunc3D 相同的打分代码，不受影响；导出的问题下周修。

**这对新 pipeline 意味着什么：** 新 pipeline 的向量也要先有部件。S0 到 S0c 用的都是标注的部件掩码，而这里说明，部件用 SAM 3 自己找时，差距主要就出在这一步。Task 3 的打分也是先要定位对（AP25 之上再看运动），所以"部件找不找得到"是新 pipeline 绕不开的问题。
""")
