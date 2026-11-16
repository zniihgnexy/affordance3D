# Cells of build_report_2026_11_16_weekly.py, part 2 of 4: run in order by that builder, with md() and code() in scope.

md(r"""
## 0. 总览：状态、数字和出处

开头的"先读这里"讲了大局。这一节只把状态和关键数字列在一处，每一条指向后面的细节和讨论稿的节号。

**北极星和贡献类型（章程）：** NeurIPS 2027 主会为主目标；是否先投 ICCV 2027、WACV 的 Applications track 是否加入，都到关口 G2 定。贡献类型是应用：构建一条新的 pipeline，核心不完全在算法创新（你 2026-11-12 的定义）。贡献必须"主要做memory system而不是robotics"（你的原话，2026-11-10），想法本身要立得住、不是插件。
- **方向（你的原话，2026-11-10）：** "我构建的mem system包含的不只是semantic的内容更是每个object possible的action结果……我希望探索这种action的sequence或者human和obj的affordance map能不能辅助用来帮助构建mmeory system，这种变为4d的search效果是否会比传统静态3d搜索好"

**项目现在在哪里**

| 项 | 现在的状态 |
| --- | --- |
| 阶段 | validate（验证）：在小规模上检验主张，还没到写论文的阶段 |
| 已过的关口 | 理解、方向、流程（限阶段 1 的范围）、主张、G1（第 7 轮："通过，按收窄后的含义"） |
| 没到的关口 | G2：主张是否成立、是否训练写入器、投哪个会议；G3："4D"是否留在标题里 |
| 采纳的主张 | 第 10 轮（2026-11-13，你回答"1 同意"）：一次走动视频写成的功能记忆，在检索变难的时候（录得长、相似部件多），比"按语义挑画面 + 同样的部件框"更可靠地找到要操作的部件，而且越难优势越大；在单个房间这种容易的情形，起作用的是部件图本身（评估第 4 节） |
| 这个主张的第一次检验 | 预测 2 的第一步（阶段 2b：把 30 个场景的关键帧混在一起检索）在**检索这一步没有通过**：功能索引一直比语义检索多 12 到 18 个百分点，但这个差不随混入的场景变多而变大（§26，ORACLE） |
| 章程提案 P-4 | 暂缓（你 2026-11-15 的原话："P-4 先不批"）。所以章程的"当前主线"一行仍按第 9 轮的方案乙写着；实际执行按账本里你第 10 轮和 11-15 的回答 |

**本周 5 个发现**（数字的出处见括号；附录 A 逐个核对）
1. **编号部件框最可靠：** 选对部件从约 28%（直接读 40 帧原视频：4B 28.5%、8B 27.7%）到约 50%（P4：4B 53.1%、8B 54.4%）；框来自标注，ORACLE；机制别人做过（第 3.2、3.4 节；§24.5、§24.8；评估第 5 节）。
2. **单个房间里记忆挑画面只多 4 到 7 个百分点：** P4 减 c4，4B +4.4 [−0.9, +9.6]，8B +6.7 [+1.8, +11.5]，按事先的规则都是"接近"；ORACLE（第 3.4 节；§24.8）。
3. **动作方向没有证据，不再当主张：** 动作类型之间 p = 0.16；动作键 +0.8 [−0.9, +2.4]，真值上限 +3.8 [+0.9, +7.5]；ORACLE（第 3.6 节；§24.5、§24.10）。
4. **2b：检索变难时优势没有变大：** 前 3 张里有对的，功能索引减语义从 +13.8 [+8.7, +19.2]（1 个场景）到 +11.8 [+8.3, +15.7]（16 个场景），斜率每翻一倍 −0.7 [−2.2, +0.7]；ORACLE，只看检索（第 3.7 节；§26）。
5. **B 停，C 往前走：** B-1 召回 27.6%、精度 15.5%，没过；C-0 通过；C-1 的 CPU 预览里 M 减 M4 只多 1 到 2 个百分点（ORACLE，预览没记进讨论稿）（第 3.8、3.9 节；§25.4）。

**11-16 和现在在跑什么**（详见第 7 节；截至 2026-11-16 晚上，服务器时间）
- 网络断后队列有 793 个作业；你答"b"：示例图和 C-1 试跑插到前面，2b 端到端仍排在你的作业后面。
- 示例图你看过，定"a，插到前面"：只用 SAM 3、Qwen 的检测不做，流程关口算通过；SAM 3 的全部运行已排在前面并开始写出结果（本 notebook 不读）。
- C-1 4B 试跑因答案格式 V、G、V'、G' 都是 0；改了 V 的提问（D-11，缓存 c1-v2），重新排队（第 3.9 节；§25.5）。

**下一个决定点：** SAM 3 版的 2a 按 §24.9 的读法看预测 1；2b 端到端按 §26.2 的读法看预测 2；两个都不支持时，到 G2 由你在"新的构建方法、换会议、重新想"之间选（评估第 10 节）；P-4 等这两个结果再改写。

**最大的风险**（评估第 7 节）：现成工具建的部件图漏掉小部件；或者检索变难时功能索引仍赢不了"语义挑画面 + 同样的部件图"。第 3.5 节的模拟里，先丢小部件、只剩四分之一时检索上的优势就没了（对标注的扰动，不是真实工具）。

**掌舵代理的看法：** 完整原文照录在细节部分的第 6 节（"Research scope 大局"），并在那里展开；开头"先读这里"的第 6 小节是它的一句话版本。

**一张图看本周最可靠的结果：** 横条是选对部件的比例。上面三组没有部件框（模型自己画框），下面两组给了标注的编号部件框。
""")

code(r"""
show(headline_figure(D["c4b"], D["c8b"]))
""")

md(r"""
**怎么看：** 有没有部件框，差约 20 到 25 个百分点（上三组对下两组）。有了部件框之后，"记忆挑的画面"（P4）比"按语义挑的画面"（c4）只多几个点。没有部件框时，记忆挑的画面（c2'）比按语义挑的（c1'）也只多几个点。
""")

md(r"""
## 1. 本周时间线

下图是每天的提交数和那天发生的事。**注意日期的口径：** 账本按工作日记录，第 7 到第 10 轮都记在 11-13；但 11-13 晚上的工作一直做到 11-14 凌晨，所以 11-14 的 24 个提交大多属于 11-13 的那几轮。
""")

code(r"""
show(timeline_figure())
""")

md(r"""
**按天看**

| 日期 | 做了什么 | 提交数 |
| --- | --- | --- |
| 11-09（周一） | 组会的数据报告（`report/2026-11-09_datasets_report.ipynb`）；本地原型：RGB-D 几何、DINOv2 特征、体素记忆；SceneFun3D 的读取和 E0 对齐检查。这一天记忆方向还没有单独分出来 | 5 |
| 11-10（周二） | 记忆方向开题（讨论稿 §1–§10）；第 1、2 轮；S1 第 1–5 步（ORACLE 版：只用视频、深度和位姿算接触点、表面朝向和站立点）；SMPL-X 人形和短动画；整条 pipeline 的演示 notebook；仓库拆分：VLA 方向移到独立仓库 `affordance-vla`；工作环境迁到实验室的服务器 | 9 |
| 11-11（周三） | 服务器环境；流程细化成 15 个环节（pipeline 工作 notebook）；第 3 轮（记忆带时间、不做稠密重建、存储倾向隐式）；第 4 轮（存储形式定为 L2） | 7 |
| 11-12（周四） | L2 记忆的代码，每个环节一个示例；第 5 轮（研究层面的评估和路线，提案）；第 6 轮（采纳"功能记忆"，以 NeurIPS 为主，批准阶段 1，规格和 G1 规则在跑之前写定）；接入 research-helm（章程、账本、提案 P-1）；阶段 1 的工具 | 7 |
| 11-13（周五） | 阶段 1 跑完并记录（§22）；审稿后的归因检查（§23）和三轮审稿；第 7 轮简报改写成一份白话文档、附流程图；G1 通过；第 8 轮复述你描述的论文；对照 c；R7-B 模拟；动作类型的检验；第二批 a、b、d、c+；第 9 轮简报和你的回答 | 24 |
| 11-14（周六，提交日期） | 账本记在 11-13 的工作：P-3 写进章程；现成工具的桌面核对；B-0、B-1、C-0；c4、c1'、c2'（4B 和 8B）；动作键检查；C-1 的代码和 M4；第 10 轮评估和你的回答；2b 的计划、代码和结果；P-4 | 24 |
| 11-15（周日） | 你对 2b 结果的回答"c+a；P-4 先不批"；§26.2 端到端版本的计划和代码 | 2 |
| 11-16（周一，本周之后） | 服务器网络断了一阵；你对队列位置的回答"b"；两个小作业插到前面并跑完；示例图给你看过，你定"a，插到前面"（只用 SAM 3），SAM 3 的全部运行排到前面；C-1 试跑发现答案格式问题，改了 V 的提问（D-11）并重新排队 | 1（只列到本 notebook 第一版之前的提交） |

**按轮看**（第 1–6 轮在接入 research-helm 之前完成，账本只做索引，没有逐字记下你的回答；这几轮写的是讨论稿记下的决定，有原话的标了引号）

| 轮 | 日期 | 问了什么 | 你的回答（原话，节选） | 之后做了什么 |
| --- | --- | --- | --- | --- |
| 1 | 11-10 | 核心主张、输入、人形的用途、定位、能否自录数据 | 无逐字记录；章程引用的原话："自录的非公开数据难以证明效果，也无法与其他基准对比" | 定位为"部件级 4D 功能记忆"；从视频开始，三维重建只当训练和评测的真值；人形只用于展示；以 H3（交互帮助构建）、H2（事件帮助检索）为主（§11）；设计 v0.2（§12） |
| 2 | 11-10 | 深度、位姿、人形展示、动作词表、顺序 | 无逐字记录 | 单目深度为主、LiDAR 作对比；允许设备位姿；用 SceneFun3D 的 9 类动作；依次推进 S1→S2→S3；逐步向导师汇报（§14） |
| 3 | 11-11 | 对需求的理解、"不做稠密重建"、精度分两档、阶段安排、总览 v2 | 无逐字记录 | 记忆带时间、记录实际发生的动作；不做稠密重建；先"找到并走到跟前"、再"碰到"；存储倾向隐式（§18） |
| 4 | 11-11 | 隐式到哪一档、哪些保持显式、几何量、谁来读记忆、用哪种特征 | 无逐字记录 | 存储形式 L2：每张关键帧的图，加部件 token 和薄的显式索引；接触点等几何在读取时现算（§19）；15 个环节各一个示例 |
| 5 | 11-12 | 确认总览 v2；你要一次研究层面的点评 | "流程基本理解，也大致没有问题"（章程引用）；同时认为作为研究课题还不够 | 评估和路线（提案）：主张"功能记忆"，预测 A（写在哪里）、B（事件）、C（代价），四个阶段、三个关口（§20） |
| 6 | 11-12 | 主张、目标会议、阶段 1、人形 | 无逐字记录 | 采纳"功能记忆"；NeurIPS 为主、ICCV 到 G2 定；批准阶段 1；人形降为补充材料里的演示；阶段 1 的规格和 G1 的判定规则在跑之前写定（§21） |
| 7 | 11-12 到 11-13 | 阶段 1 的结果：G1 是否通过、贡献怎么划分、补哪些对照、用什么真实场景讲 | "通过，按收窄后的含义"；"……最好还是使用传统的vlm模型直接读取原video来完成整体系统的对比"；"都做，但是先做c在做其他"；"我希望优先一和三，二作为补充说明"；另外要求"之后每次阶段性汇报的时候，都给我一个流程图" | G1 过；对照 c 里加了"直接读原视频"一组（D-6）；使用场景以家里的使用指导、无障碍与远程协助为主；以后每次阶段汇报附流程图、决定集中在一份文档 |
| 8 | 11-13 | 复述你在聊天里描述的论文（CA 新构建方法、CB 按功能搜索、CC 更快更准、CD 动作类更强），5 处确认 | "1 对；2 同意；4 参考其他模型的比较，尽量能够横向对齐；5 同意"；""action 的方向"指AC，但是为了能够具有平行比较的能力，希望能够同时做B" | 4 个调研代理（已有的构建方法、对手的设置、B 和 C 的可行性、四项贡献的撞车检查）；章程提案 P-2 |
| 9 | 11-13 | 对照 c 和调研之后：论文重心放在甲、乙、丙哪个；B、C 的头两步；换 8B；补 c4、c1'、c2' | "1 乙；1b 同意你的建议；2 批，并且先做问题 7 和问题 6a；3 批；4 可以；5 批；6a 批，插到队列前面；6b 同意你的建议；7 同意你的建议" | 收回第 8 轮"新构建方法是贡献"那一条，记为 DF-4；P-3 批准并写进章程；SAM 3 许可证"可以，同意许可证，从镜像下载"、GPU"批到 7 小时"；B-0 修补后"1 接受"；c4 接近 P4，按规则停下讨论（"同意你的建议"） |
| 10 | 11-13 | 评估：主张改成"检索变难时"；恢复问题 2 的检查；停的规则；2b；动作方向；B；C-1 的规则 | "1 同意；2 恢复；3 b；4 批；5 b；6 a；7 同意你的建议，a" | 2b 当天写计划、跑完：检索这一步没过；P-4 写好待批 |
| — | 11-15 | 2b 之后怎么走：(a) 测端到端；(b) 接受结果、到 G2；(c) 先只做现成工具的检查 | "c+a；P-4 先不批" | 两件都做；§26.2 的计划写明"这是第一步没过之后才加的检查"；代码写好、排队 |
| — | 11-16 | 网络断后队列里有 793 个作业：两个小作业要不要插到前面 | "b" | 示例图和 C-1 试跑插到前面；2b 端到端不动 |
| — | 11-16 | 看过示例图之后：(a) 只用 SAM 3 全部跑，Qwen 这次不做；(b) 先改 Qwen 的提示词再看；(c) 两个都跑 | "a，插到前面" | 流程关口算通过；只跑 SAM 3（以它为准），Qwen 的检测不做；这项检查的 GPU 作业插到前面 |
""")

md(r"""
## 2. 流程图：现在的流程和第 10 轮的目标

**怎么看这张图**
- **下面两行是今天的流程。** "写入"每段视频做一次，结果就是记忆；"读取"每来一个问题做一次。
- **最上面一行是第 10 轮采纳的目标里要变的东西。** 灰色表示还没做；虚线指向它要替换或改变的那个方框。
- **颜色：** 绿色只用输入算；橙色暂时用标注（ORACLE）；蓝色是没训练的公开模型；灰色还没做。斜线表示这个方框吃的是 ORACLE 方框的输出，所以它的数字也是上限。
- **红框和红标签（Pred1、Pred2、Pred3）：** 第 10 轮主张的三个预测落在流程的哪里。图下面每一行的方括号是这个预测的证据现在到哪一步（照记录写，不是新结果）。灰色"design"一行是不声称的机制。
""")

code(r"""
show(FC.draw(FC.CHARTS["now_target_r10"]))
""")

md(r"""
| 标记 | 对应的预测 | 证据现在到哪一步 |
| --- | --- | --- |
| Pred1 | 预测 1：现成工具从视频建出的部件图，交给同一个视觉语言模型，比它直接读原视频、比"用问题文字直接提示 SAM 3"都更准 | **没测。** 阶段 2a 已批（GPU 上限 7 小时）；11-16 你看过示例图，定"a，插到前面"：只用 SAM 3，全部运行已排到前面。只有上限：标注部件图的 P4 54.4% 对直接读原视频 27.7%（8B，§24.8） |
| Pred2 | 预测 2：同样的部件框下，检索越难，功能索引比语义挑画面强得越多 | **检索这一步不支持**（2b，§26）：差一直在（+11.8 到 +17.8），斜率 −0.7 [−2.2, +0.7]。端到端版本排队中（§26.2） |
| Pred3 | 预测 3：把写入算进去，同一个场景问到一定次数之后，写一次的记忆比每次重读视频省 | **没测。** 只量了读：每个问题 P4 0.7 秒、3 张图，读原视频 3.5 秒、41 张图（8B 中位数，只算模型调用）。写入的 24 秒不含选关键帧、单目深度和建部件图。"划算点"暂定 15 个问题，测之前再定 |
| design（不声称） | 编号部件框 | 可靠（4B、8B 一致，约 28% → 约 50%），但别人做过，只当基线结果 |
""")

md(r"""
## 3. 实验结果

每一项按同样的五步写：**问的是什么**；**事先定的规则**（都在跑之前写进讨论稿）；**数字**；**按规则的判定**；**这意味着什么**。

### 3.1 阶段 1（P0–P4）和 §23 的归因检查

**问的是什么：** 记忆预算相同时，token 写在 affordance 区域上（P4），是否比其他写法更常把正确的关键帧排在第一位（预测 A）。五种写法只有"token 写在哪里"不同：P0 只有整帧；P1 均匀网格；P2 随机位置；P3 通用区域（SAM 分出的、不知道哪里能操作的区域）；P4 affordance 区域。打个比方：在一本 40 页的相册里找"能开灯的开关"。P0 一页页看整张照片；P4 事先告诉你哪几页有可操作的东西，并把它们框出来。

**事先定的规则（§21.6，预算为 1 倍）：** "通过"要两条：P4 的"第一个关键帧对"高于 P3，差的区间不含 0；端到端选对部件的比例 P4 不低于 P3。"部分通过"：P4 只高于随机位置 P2。"不通过"：P4 不高于 P2。

**数字（ORACLE，§22.3；Qwen3-VL-4B 读图，阶段 1 的提示词和打分标准）：**
""")

code(r"""
names = {"P4": "P4 affordance 区域", "P3": "P3 通用区域", "P0": "P0 只有整帧", "P2": "P2 随机位置"}
rows = []
for key, label in names.items():
    top3 = get("phase1", "right_keyframe_in_top3", key)
    reading = get("phase1", "reading", {"P4": "P4 affordance regions", "P3": "P3 generic regions"}[key]) if key in ("P4", "P3") else None
    rows.append({"写法": label, "第一个关键帧对": rate(get("phase1", "first_keyframe_right", key)), "前 3 个关键帧里有": rate(top3),
                 "端到端选对部件（4B）": rate(reading) if reading else "没有做", "部件图": "标注（ORACLE）" if key == "P4" else "不用部件图"})
display(pd.DataFrame(rows).set_index("写法"))
print("P4 minus P3, first keyframe right:", diff(get("phase1", "first_keyframe_right", "P4 minus P3")), "points;",
      "rule category:", get("phase1", "rule_21_6_category"), "(whether G1 passes is the lead's decision)")
show(R.headline_figure(D["phase1"]))
""")

md(r"""
**审稿之后加的归因检查（§23，只用 CPU，计划在算之前写定）：** P4 同时得到了两样东西：token 的内容，和"哪些关键帧里有可操作部件"这一帧级标记（下文叫"比特"）。下图把两样分开。带斜线的两组拿到了和 P4 完全相同的关键帧。
""")

code(r"""
show(R.attribution_figure(D["diag"]))
""")

md(r"""
| 成对比较（第一个关键帧对，ORACLE，§23.3） | 差（个百分点） |
| --- | --- |
| 只用比特（不写任何 token）减 P0 | +9.0 [+6.1, +12.5] |
| 只用 token（去掉空关键帧的扣分）减 P0 | +17.4 [+11.7, +23.7]（仍带着一部分比特的作用） |
| 两样都用（P4）减 P0 | +20.5 [+14.4, +26.9] |
| 同一批关键帧里，P4 减通用区域 | +7.0 [+2.8, +11.3]；token 的权重降到 0.5 或 0.25 时缩到 +1.3 到 +4.9 |

**按规则的判定：** 属于"通过"一档；§23 属于判读表的第一行（帧级信息之外，token 写在 affordance 区域上还有贡献）。你在第 7 轮的回答是"通过，按收窄后的含义"。

**这意味着什么：**
- 在 ORACLE 下，"按功能标记关键帧"本身就值 9 个点。"token 写在哪里"本身约 7 个点，而且随打分方式变化。
- 端到端 53.6% 对 10.3% 的大差距，主要来自"给模型的候选是什么"：给模型看的框里有一个在目标部件上的任务，P4 是 83.6%，P3 只有 20.0%。
- 作为参照，随便挑一张关键帧，对的概率是 9.2%。
- 没成立的：选相机转得慢的帧没有多看到部件（481 个对均匀取帧的 501 个，共 561 个标注部件）；从一张画面现算的接触点，3 厘米以内的只有 38.5%（按 481 个真实部件算，§22.5）。

### 3.2 对照 c：不用记忆，让模型自己找（第一批，§24.1–24.5）

**问的是什么：** 你在第 7 轮要求"使用传统的vlm模型直接读取原video来完成整体系统的对比"。不用我们的记忆，同一个 4B 模型自己找部件，能不能接近 P4？
- **P4：** 记忆挑 3 张关键帧，标注区域加编号框，模型选编号。
- **c3：** 直接读原视频：按时间均匀取 40 帧，模型先挑一帧，再在这一帧的原分辨率上画框。
- **c2：** 记忆挑的 3 张关键帧，没有框，模型一次看 3 张、自己画框。
- **c1：** 按整张画面的语义挑 3 张，没有框，模型自己画框。

**事先定的规则（§24.3）：** 统一的"对"：框的中心落在目标部件上（部件向外放宽 8 个像素）。P4 减某组的区间下沿低于 5 个百分点算"接近"；差本身大于 10、只是区间太宽才低于 5，算"说不清"。c1 或 c3 接近 P4：不用记忆也能做到，停下来讨论。P4 领先：在 ORACLE 下是意料之中，只报告为"阶段 2 有多少余地"。
""")

code(r"""
VERDICT = {"behind": "落后于 P4", "close": "接近", "unclear": "说不清"}


def arm_table(summary, arms):
    rows = []
    for arm, label, what, part_map in arms:
        a = summary["arms"][arm]
        against = summary["against_p4"].get(arm)
        rows.append({"组": label, "给模型看什么": what, "选对部件": rate(a["right_part"]), "选中的画面里有目标": f"{100 * a['right_frame']['rate']:.1f}%",
                     "每题秒数（中位数）": a["seconds_per_question"], "每题图数": int(a["images_per_question"]),
                     "P4 减这一组": diff(against) if against else "—",
                     "按规则": VERDICT.get(against["verdict"], against["verdict"]) if against else "—", "部件图": part_map})
    return pd.DataFrame(rows).set_index("组")


ORACLE_MAP, NO_MAP = "标注（ORACLE，上限）", "不用（不是上限）"
MEMORY_FRAMES = "不用；画面由记忆挑（用到标注的关键帧标记）"
display(arm_table(D["c4b"], [
    ("p4n", "P4", "记忆挑的 3 张，标注区域加编号框", ORACLE_MAP),
    ("c3", "c3", "均匀的 40 帧原视频；先挑帧再画框", NO_MAP),
    ("c2", "c2", "记忆挑的 3 张；一次看 3 张、自己画框", MEMORY_FRAMES),
    ("c1", "c1", "按语义挑的 3 张；一次看 3 张、自己画框", NO_MAP)]))
print("c2 minus c1 (choosing keyframes with the memory, the model draws):", diff(get("c4b", "choosing_keyframes")), "points")
print(f"c3's 40 even frames show the target in {get('ceiling', 'shown_in_uniform_frames')} of {get('ceiling', 'tasks')} tasks "
      f"({100 * get('ceiling', 'share'):.1f}%)")
show(R.controls_figure(D["c4b"], D["by_action"]["actions"], between_types_p=get("by_action", "heterogeneity_p4_minus_c3", "p_value")))
""")

md(r"""
图的右半边按目标部件的动作类型拆开，第 3.6 节再讲。

**按规则的判定：** 没有一组"接近"。P4 减 c3 是 +24.6 [+19.2, +31.5]，P4 减 c1 是 +37.9 [+31.2, +45.4]：都"落后于 P4"，只算阶段 2 的余地。c2 减 c1 是 +2.1 [−1.1, +5.4]：只靠记忆挑画面、让模型自己画框，看不出帮助。

**这意味着什么：**
- P4 的领先主要来自**标注给出的部件框**，不能算成记忆的功劳（第 9 轮审稿的阻断项 V-38）。
- c3 低不是因为画面里没有目标：均匀的 40 帧在 98.2% 的任务里看得到目标部件。
- 读法有影响：画面挑对时，c3（先挑一帧、再在原分辨率上画框）框对约 52%（28.5 ÷ 54.4），c2（一次看 3 张直接画）只有约 26%（17.2 ÷ 66.9）。所以第 9 轮又补了 c1'、c2'（第 3.4 节）。

### 3.3 第二批：a、b、d、c+（4B，§24.7）

**问的是什么：**
- **a（只做检索）：** 把全部通用区域、或全部网格档位都写进记忆，能不能在检索上追上 P4？
- **b：** P3 的通用区域用 5 个随机种子，和 P4 同样的读法、中性提示词。
- **d：** 在全部通用区域里得分最高的 40 个上加编号框，模型挑。
- **c+：** 不用记忆，把我们挑的 40 张关键帧给模型，先挑一张再画框（读法同 c3）。

**事先定的规则：** 沿用 §24.3。a 接近 P4 说明"全写进去也行"，P4 只赢在存得少；c+ 减 c3 是"我们挑的关键帧"对"均匀取帧"的价值，不算记忆的价值。
""")

code(r"""
display(arm_table(D["c4b"], [
    ("p4n", "P4", "记忆挑的 3 张，标注区域加编号框", ORACLE_MAP),
    ("cplus", "c+", "我们挑的 40 张关键帧；先挑帧再画框", NO_MAP),
    ("d", "d", "记忆挑的 3 张上得分最高的 40 个通用区域", "不用（通用区域）"),
    ("b", "b", "和 P4 一样多的通用区域（5 个种子）", "不用（通用区域）")]))
a_names = {"P4": "P4 affordance 区域", "a: every grid level": "a：全部网格档位", "a: every generic region": "a：全部通用区域",
           "P3": "P3 通用区域（1 倍）", "P0": "P0 只有整帧"}
display(pd.DataFrame([{"记忆里写什么": label, "第一个关键帧对": rate(get("a", "arms", key)),
                       "每个场景的 token（中位数）": int(get("a", "arms", key, "tokens_per_scene")),
                       "P4 减这一组": diff(get("a", "against_p4", key)) if key.startswith("a:") else "—"}
                      for key, label in a_names.items()]).set_index("记忆里写什么"))
print("c+ minus c3 (our keyframes vs even frames, no memory):", diff(get("c4b", "our_keyframes_vs_even_frames")), "points")
rows = {arm: get("c4b", "arms", arm, "right_part") for arm in ("p4n", "cplus", "d", "b")}
a_rows = {key: get("a", "arms", key) for key in a_names}
fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 4.2))
R._rate_bars(left, rows, list(rows), ["P4: annotated boxes, memory's frames", "c+: our 40 keyframes, VLM draws",
                                      "d: top 40 of every generic region", "b: generic regions as many as P4's"], [BOX, NOBOX, GENERIC, GENERIC])
left.set_title("End-to-end, 4B: chosen box on the target part (%)", fontsize=10, loc="left", color=INK)
R._rate_bars(right, a_rows, list(a_rows), [f"{key} ({int(a_rows[key]['tokens_per_scene'])} tokens per scene)"
                                           for key, label in a_names.items()], [BOX, R.POLICY_COLOUR[R.A.P1], GENERIC, GENERIC, MUTED])
right.set_title("a, retrieval only: first keyframe right (%)", fontsize=10, loc="left", color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 1))
R._oracle_note(fig, "P4 uses annotated part maps (ORACLE). b and d write generic regions into the memory (b matched to P4 in number and size); c+ uses no map.")
show(fig)
""")

md(r"""
*（上图只画了端到端的四组；检索那一组 a 见上面第二张表。）*

**按规则的判定：** 四组都"落后于 P4"。c+ 减 c3 是 −2.1 [−6.6, +2.7]。a：全部网格档位 53.6%、全部通用区域 50.0%，都低于 P4 的 66.2%（ORACLE）。

**这意味着什么：** 在 ORACLE 下，标注的部件框比任何通用区域的框都好得多，不论只放和 P4 一样多（b）还是全部写进去（d）。P4 不只存得少（每个场景约 63 个 token，对两三千个），检索也更准。把均匀帧换成我们挑的关键帧，对直接读画面的模型没有帮助。

### 3.4 c4、c1'、c2'，以及换成 8B（§24.8）

**问的是什么：** 记忆本身值多少。
- **c4：** 同样的标注框，但画面不用记忆挑，而是按整张画面的语义挑 3 张（c1 的画面）。P4 减 c4 是"记忆挑画面"的价值（仍是上限：记忆挑画面时用到了标注的关键帧标记）。
- **c1'、c2'：** c1、c2 改用 c3 的读法（先挑一张，再在原分辨率上画框）。c2' 减 c1' 是同一种读法下"挑画面"的价值。
- **8B：** 换成 2026 年论文里默认的 Qwen3-VL-8B，在 390 个任务上把这几组都再跑一遍。

**事先定的规则（4B、8B 各读一次）：** P4 减 c4 的区间下沿低于 5 个百分点记为"接近"：第 9 轮选的方案乙以"按功能建索引"为重心，所以停下来讨论。c2' 减 c1' 的区间下沿大于 0 记为"有价值"，否则"看不出"。8B 的 c1 或 c3 接近 P4：停下来讨论。
""")

code(r"""
arms8 = [("p4n", "P4", "记忆挑的 3 张，标注区域加编号框", ORACLE_MAP),
         ("c4", "c4", "按语义挑的 3 张，标注区域加编号框", ORACLE_MAP),
         ("c2s", "c2'", "记忆挑的 3 张；先挑一张再画框", MEMORY_FRAMES),
         ("c3", "c3", "均匀的 40 帧原视频；先挑帧再画框", NO_MAP),
         ("c1s", "c1'", "按语义挑的 3 张；先挑一张再画框", NO_MAP),
         ("c1", "c1", "按语义挑的 3 张；一次看 3 张、自己画框", NO_MAP)]
table8 = arm_table(D["c8b"], arms8)
table8.insert(1, "4B 的结果", [f"{100 * get('c4b', 'arms', arm, 'right_part', 'rate'):.1f}%" for arm, *_ in arms8])
table8.insert(2, "8B 减 4B", [diff(get("c8b", "against_4b", arm)) for arm, *_ in arms8])
print("Qwen3-VL-8B (4B for comparison):")
display(table8)
print("c2' minus c1' (choosing keyframes, same reading):  4B", diff(get("c4b", "choosing_keyframes_same_reading")),
      "  8B", diff(get("c8b", "choosing_keyframes_same_reading")))
diffs = {"P4 − c4 (4B)": get("c4b", "against_p4", "c4"), "P4 − c4 (8B)": get("c8b", "against_p4", "c4"),
         "c2' − c1' (4B)": get("c4b", "choosing_keyframes_same_reading"), "c2' − c1' (8B)": get("c8b", "choosing_keyframes_same_reading"),
         "P4 − c3 (4B)": get("c4b", "against_p4", "c3"), "P4 − c3 (8B)": get("c8b", "against_p4", "c3")}
rows = {arm: get("c8b", "arms", arm, "right_part") for arm in ("p4n", "c4", "c2s", "c3", "c1s", "c1")}
show(bars_and_differences(
    rows, list(rows), ["P4 (annotated boxes, memory's frames)", "c4 (annotated boxes, frames by meaning)", "c2' (memory's frames, VLM draws)",
                       "c3 (raw video, VLM draws)", "c1' (frames by meaning, VLM draws)", "c1 (frames by meaning, one-shot)"],
    [BOX, BOX, MEMORY, NOBOX, NOBOX, R.POLICY_COLOUR[R.A.P0]], diffs, list(diffs), list(diffs),
    [BOX, BOX, MEMORY, MEMORY, NOBOX, NOBOX], "8B: chosen box on the target part (%)", "Paired differences, points (line = 95% interval)",
    "P4 and c4 use annotated part maps (ORACLE). c2' and P4 use the memory's keyframes.", rule_at=5,
    rule_text="'close' if the lower end\nis below +5"))
""")

md(r"""
**按规则的判定：**
- **c4 接近 P4，4B 和 8B 都是：** 4B +4.4 [−0.9, +9.6]，8B +6.7 [+1.8, +11.5]。8B 的区间下沿大于 0 但低于 5：记忆挑画面有一点用，但不到"明显不接近"的程度。按规则停下讨论，这就是第 10 轮评估的起点。
- **c2' 减 c1'：** 4B +3.6 [−0.9, +8.1]，8B +3.6 [−0.8, +8.2]：同一种读法下，记忆挑画面看不出价值。
- **8B 没有接近 P4 的不用记忆的组。** 换成 8B 几乎没有变化：各组 8B 减 4B 都在 ±3.3 个百分点以内。c1'、c2' 的 4B、8B 总数恰好相同，记录里核对过：780 个答案全都不一样，只是总数巧合（§24.8）。

**这意味着什么：** 不论 4B 还是 8B，P4 比直接读原视频高出的约 25 个百分点，几乎都来自"有部件的框"（affordance map 本身），而不是记忆挑画面。按事先的规则，这一条让第 9 轮的方案乙（以按功能建索引为重心）失去了大半支撑。

**时间（只算模型调用，中位数，不含写入）：** 8B 的 P4 每题 0.7 秒、3 张图；c3 3.5 秒、41 张图。4B 的 P4 1.2 秒、c3 5.7 秒。8B 反而比 4B 快，原因没查（可能是不同的卡、不同的负载），这些时间只能粗看。

### 3.5 R7-B：漏检和误检的模拟（§24.6）

**问的是什么：** 自己建的部件图会漏掉部件（漏检），也会把不能操作的地方当成部件（误检）。漏多少、错多少以内，按功能建的记忆在检索上仍然比通用区域好？只用 CPU，扰动的是标注。

**事先定的规则：** 每一格和 P3（通用区域，47.4%）成对比较：区间下沿大于 0 记为"明显好于 P3"。**门槛：** 误检为真部件数的 0.5 倍时，从留 100% 往下，一直明显好于 P3 的最低留存比例。如果留 100%、误检 0.5 倍时已经不明显好于 P3，说明对误检很敏感。

**数字：** 下图上排是事先写定的规则；下排是第 9 轮第 2 次审稿后的更严的重算（中心落在任何部件上的通用区域都不算误检；规则在算之前写定）。格子里是"第一个关键帧对"的比例，标"= P3"的格子不明显好于 P3。
""")

code(r"""
show(map_error_figure(D["errors"], D["errors_centre"]))
cell = get("errors", "grid", "random", "1", "0.5")
print("kept 100%, false alarms 0.5x, minus P3:", diff(cell["against_p3"]), "points;",
      "thresholds (pre-set rule):", get("errors", "threshold_keep_share"), "; stricter re-run:", get("errors_centre", "threshold_keep_share"))
""")

md(r"""
**按规则的判定：** 两种丢法的门槛都是 25%，也就是试到的最低一档。留 100%、误检 0.5 倍时仍明显好于 P3（+15.2 [+10.4, +20.3]），按读法记为"在检索这一步对误检不敏感"。更严的重算结论不变（留 100%、误检 0.5 倍：随机丢 63.3%、先丢小的 62.3%，原来都是 62.6%）。

**这意味着什么：**
- 在检索这一步，**漏检比误检要紧**：误检到 1 倍时少 4 到 5 个百分点；漏掉一半部件时少 7 到 8 个；先丢小的、只剩四分之一时（47.2%），和 P3 看不出差别。
- 措辞要窄（第 10 轮评估的审稿 V-73）："对误检不敏感"只指检索这一步、误检和部件一样大的时候，而且一部分来自"空关键帧扣 3 分"这个常数（只留 25% 时加误检分数反而上升）。读图那一步对误检可能敏感得多：b、d 在通用区域的框里挑，只有 11% 到 14%。
- 真实工具的误检常常是整个物体的大框，这里模拟不到；要靠阶段 2a 的真实输出来量。

### 3.6 动作类型：差距在动作类型之间是否不同（p = 0.16），和按动作类型检索（§24.5、§24.10）

**问的是什么：** 你在第 7 轮写过，希望"一些关于action的方向上又格外高的performance"。两个检查：
1. **检验（§24.5，规格之外，计算前写定）：** P4 对 c3 的差距（第 3.2 节图的右半边），在动作类型之间是否真的不同。在每个场景内部打乱动作标签 10,000 次，看实际的离散程度是否大于随机分组。规则：p < 0.05 记为"确实不同"，否则"看不出不同"。
2. **动作键检索（§24.10，只用 CPU，上限）：** 给每个区域加一个"功能的键"（例如 "rotate a knob or a dial"），检索时按任务文字匹配这个键。五组：不加键（就是 P4）、动作短语、只有部件词、只有动词、真值动作类型（按真值筛选的上限）。规则：动作短语减不加键的区间下沿大于 0 记为"有用"；真值减不加键本身不到 5 个百分点，记为"按动作类型检索的余地很小"。
""")

code(r"""
het = get("by_action", "heterogeneity_p4_minus_c3")
print(f"test: {het['tasks']} tasks, {het['action_types']} action types, {het['shuffles']:,} shuffles within scenes, p = {het['p_value']:.3f}")
key_names = {"no key": "no key (= P4)", "action phrase": "action phrase", "part words only": "part words only", "verb only": "verb only",
             "true type": "true action type (upper bound)"}
rows = {key: get("key", "arms", key) for key in key_names}
against = {key: get("key", "against_no_key", key) for key in key_names if key != "no key"}
display(pd.DataFrame([{"检索键": label, "第一个关键帧对": rate(rows[key]), "减不加键": diff(against[key]) if key in against else "—"}
                      for key, label in key_names.items()]).set_index("检索键"))
print("verb only minus part words only:", diff(get("key", "verb_minus_part")), "points")
show(bars_and_differences(
    rows, list(rows), list(key_names.values()), [BOX, "#eda100", "#eda100", "#eda100", MUTED], against, list(against),
    [f"{key_names[k]} − no key" for k in against], ["#eda100", "#eda100", "#eda100", MUTED],
    "First keyframe right (%), annotated maps", "Paired difference against no key, points", "Annotated part maps and annotated action "
    "types (ORACLE). Every SceneFun3D task names its action and part.", rule_at=5, rule_text="'little room' if the true\ntype adds under 5"))
""")

md(r"""
**按规则的判定：** 检验 p = 0.16（389 个任务、8 个动作类型），记为"看不出不同"。动作短语 +0.8 [−0.9, +2.4]，记为"看不出有用"；真值 +3.8 [+0.9, +7.5]，不到 5 个点，记为"余地很小"；动词和部件词没有差别（0.0 [−2.4, +2.2]）。

**这意味着什么：**
- SceneFun3D 的每个任务都点明了动作和部件（"Rotate the knob…"），所以按语义检索本身已经带上了动作的信息；动作类型又几乎就是部件类型（转的是旋钮、拉的是拉手）。
- 第 10 轮你选了 (b)：在 2b 里再加一组，看相似部件多时动作键有没有用。结果也没有用（第 3.7 节）。所以按评估，"按动作类型检索"不再当主张，动作方向改由 C（能直接拿去做的回答）来体现。

### 3.7 阶段 2b：把多个场景的关键帧混在一起检索（§26）

**问的是什么：** 预测 2 的第一步：检索变难时，功能索引挑画面比语义挑画面强多少，这个差是否随难度变大。做法：对每个目标场景，把它的 40 张关键帧和另外 K − 1 个场景的关键帧混成一个记忆（K = 1、2、4、8、16；别的场景随机抽 5 次）。目标场景的任务只能在目标场景里找到对的；混进来的场景带来更多同类的部件（别的抽屉、别的开关）。两边用同样的标注部件图，只是挑画面的办法不同。

**事先定的规则：** 看"前 3 张里有没有对的"（读图那一步看的就是前 3 张）。功能索引减语义的差对 log2 K 的斜率，区间下沿大于 0，而且 K = 16 时差本身至少 5 个百分点，才算"支持"。动作键那一组同样读法。
""")

code(r"""
rows = []
for k in get("pooled", "pool_sizes"):
    per = get("pooled", "per_size", str(k))
    rows.append({"K（混进来的场景数）": k, "同动作类型的标注区域（中位数）": round(per["median_same_type_regions"]),
                 "语义": rate(per["semantic"]["top3"]), "功能索引": rate(per["function index"]["top3"]),
                 "功能索引 + 动作键": rate(per["function index + action key"]["top3"]),
                 "功能索引减语义": diff(per["function index minus semantic (top3)"]),
                 "动作键减功能索引": diff(per["function index + action key minus function index (top3)"])})
display(pd.DataFrame(rows).set_index("K（混进来的场景数）"))
for name, reading in get("pooled", "readings").items():
    print(f"{name}: slope {diff(reading['slope_per_doubling'])} points per doubling, gap at K = 16 "
          f"{100 * reading['gap_at_largest']:+.1f}, supported: {reading['supported']}")
show(pooled_figure(D["pooled"]))
""")

md(r"""
**按规则的判定：** 斜率 −0.7 [−2.2, +0.7]，区间包含 0：**预测 2 的第一步在检索这一步不支持。** 差一直在（+11.8 到 +17.8），但不随难度变大。动作键那一组斜率 +0.1 [−0.7, +0.9]，K = 16 时 +0.5：相似部件多时也没有用。

**这意味着什么：**
- 只作描述：检索这一步，功能索引在每个 K 都多找到 12 到 18 个百分点的对画面。但在单个房间里，读图那一步有了部件框就把这个差抹掉了大半（c4，第 3.4 节）。K 变大时，语义挑的前 3 张里常常一张对的都没有（K = 16 时有对的只占 32.7%），读图就补不回来。所以端到端的差会不会随 K 变大，这次没测。
- **局限：** 混进来的是别的房间，房间本身的样子会帮语义检索把它们分开，比"同一个长视频里的相似部件"容易；标注的部件图，是上限。
- **这就是 §26.2 端到端版本的来由。** 它是第一步没过之后才加的检查，§26.2 和账本都写明了这一点：8B 读编号框，K = 1、4、16，每个 K 一组抽样；斜率区间下沿大于 0 且 K = 16 时差至少 5 个点，记为"端到端支持预测 2"，否则按评估第 10 节到关口 G2。K = 1 应当接近 8B 的 P4 和 c4（54.4%、47.7%），作为核对。现在排在你的作业后面。

### 3.8 B-0、B-1：HD-EPIC 上问"做过什么"的头两步（§25.1、§25.2、§25.4）

**问的是什么：** B 是"做过什么"类的问题（例如"抽屉开了几次""这个物体被挪到了哪里"）。你在第 8 轮说"希望能够同时做B"以便平行比较。头两步只用 CPU，在厨房 P02 上：
- **B-0：** HD-EPIC 的 242 道题能不能对上标注里的事件，用论文的规则能不能重新算出答案。
- **B-1：** 不用标注，只靠 Aria 眼镜算出的手掌位置（第 9 轮你回答"可以"当输入），能不能找出"拿起、放下"的时刻。

**事先定的规则：** B-0：至少 90% 的题对得上、规则算出的答案至少 90% 对，才往下走；否则是数据问题，先修数据。B-1：精度至少 50% 时召回至少 60% 才往 B-2 走；否则不用标注的事件需要 GPU 上用画面的检测器，B 的代价要由你重新和 A、C 比。
""")

code(r"""
b0, b0s, b1 = get("b0", "summary"), get("b0_swapped", "summary"), get("b1", "summary")
display(pd.DataFrame([
    {"运行": "B-0 按写定的规则", "对上标注事件的题": f"{b0['tied']}/{b0['questions']} ({100 * b0['tied_share']:.1f}%)",
     "规则答案对": f"{b0['reproduced']}/{b0['rule_questions']} ({100 * b0['reproduced_share']:.1f}%)", "判定": "数据问题"},
    {"运行": "B-0 框的横纵坐标换过来（看到结果后加的修补）", "对上标注事件的题": f"{b0s['tied']}/{b0s['questions']} ({100 * b0s['tied_share']:.1f}%)",
     "规则答案对": f"{b0s['reproduced']}/{b0s['rule_questions']} ({100 * b0s['reproduced_share']:.1f}%)", "判定": "你接受后记为通过（\"1 接受\"）"}]).set_index("运行"))
ceiling = b1["palm_within_30cm_at_0"] * b1["clock_check_palm_to_object"]["n_at_0"] / b1["truths"]
print(f"B-1: {b1['truths']:,} annotated pick-ups and put-downs, {b1['detections']:,} detections; recall {100 * b1['recall']:.1f}%, "
      f"precision {100 * b1['precision']:.1f}% (bar: recall >= 60% at precision >= 50%) -> {b1['verdict']}")
print(f"     detections shifted by 15 s (chance): recall {100 * b1['control_shift_15s']['recall']:.1f}%, "
      f"precision {100 * b1['control_shift_15s']['precision']:.1f}%")
print(f"     a tracked palm within 30 cm of the object at the annotated moment: {100 * ceiling:.0f}% of all annotated moments "
      f"({b1['clock_check_palm_to_object']['n_at_0']:,} have a palm reading, {100 * b1['palm_within_30cm_at_0']:.1f}% of those)")
print(f"     alignment check: wearer-to-object distance error, median {100 * b1['clock_check_wearer_distance']['median_at_0']:.1f} cm "
      f"at offset 0 (best offset {b1['clock_check_wearer_distance']['best_offset_s']:.1f} s)")
show(hdepic_figure(D["b0"], D["b0_swapped"], D["b1"]))
""")

md(r"""
**按规则的判定：**
- **B-0：** 按写定的规则判为数据问题（16.9% 和 11.1%）。原因：题目里的框和标注里的框是同一组数字，只是横、纵坐标的顺序反了（202 道带框的题里 201 道换过来就完全一样）。换过来以后 94.2% 和 94.4%，都过了 90%。**这个修补是看到结果之后加的**；哪一边的顺序是对的要看一帧视频才能确定，视频没下载。你回答"1 接受"，B-0 记为通过，原判定和修补都留在记录里。
- **B-1：** 召回 27.6%、精度 15.5%，**没过**。换别的速度门槛（只作描述）最多到召回 32.1%、精度 20.3%。只有约 52% 的标注时刻手部跟踪里恰好有一只手在物体 30 厘米以内（推断：由结果文件算出，见上面打印的数），所以光靠手部跟踪很难到 60%。

**这意味着什么：** B 没有不用标注的写入方法。第 10 轮你选了 (a)：B 先停，B-2（下载约 10 GB 视频）不做；标题里的"4D"仍按章程到关口 G3 定。为 B 搭好的东西（HD-EPIC 标注、手腕数据、轨迹对齐，对齐误差中位数 1.9 厘米）都留着。

### 3.9 C-0 和 C-1 的 CPU 预览（§25.3、§25.5；预览不在讨论稿里）

**问的是什么：** C 是"能直接拿去做的回答"：除了"哪个部件"，还要答出接触点、动作类型、往哪个方向动、铰链在哪一侧。只对着 SceneFun3D 的标注打分，不真的执行（围栏的要求，也和兄弟项目 affordance-vla 分开）。
- **C-0（只用 CPU）：** 能不能从标注里为每个任务建一条"动作记录"当真值。
- **C-1（4B，GPU）：** 四组：V（普通视觉语言模型直接答）、G（几何规则）、M（我们的记忆：P4 挑出的帧和区域，接触点和类别来自标注，是上限）、O（真值，只用来核对打分程序）；第 10 轮评估的审稿后加了 M4（c4 的选择：同样的标注框，画面按语义挑，不用记忆）。

**事先定的规则：** C-0：至少 90% 的任务每个目标部件都有运动标注；至少 80% 的任务投到画面上的方向至少 20 像素；打分程序给真值打 100%。三条都满足才往 C-1 走。C-1 主结果：M 减 V 的"联合成功"（接触点在部件上、类型对、方向的直线误差在 20 度以内）区间下沿至少 10 个百分点，且在接触点对的任务上方向至少高 5 个点，C 才单独成立。M 减 M4 另报（账本 D-10），区间下沿低于 5 算"接近"。第 10 轮你回答"同意你的建议，a"：照原规则按 M 减 V 判定，M 减 M4 照样报告，并标明原规则包含标注的功劳。

**数字：** 下表是 C-1 的 **CPU 预览**：只有 M、M4、O 三组（V、G 要 GPU，还没跑），在 380 个任务上；4B 和 8B 指 M 和 M4 用的是哪个模型的 P4、c4 选择。预览的结果文件在 11-14 凌晨和 M4 的代码一起生成（账本记为第 10 轮那天的工作），还没有记进讨论稿，也没有审稿。
""")
