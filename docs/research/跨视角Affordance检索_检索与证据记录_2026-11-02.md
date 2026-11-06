# 跨视角 affordance 环境记忆检索：检索与证据记录

**检索截止：2026-11-02｜类型：立项决策用定向范围综述**

对应[立项评估报告](跨视角Affordance检索_文献调研与立项评估_2026-11-02.md)、[逐篇结构化证据表](跨视角Affordance检索_文献证据表_2026-11-02.json)和[持续追加文献档案](相关文献档案_持续追加.md)。本记录只描述实际进行的查询与来源核查，不把范围综述包装成穷尽式系统综述。

## 1. 问题与范围

核心问题：移动视角 RGB 视频形成环境记忆，之后按实例及功能要求检索对象和有用历史证据帧；判断该组合与预算化功能证据保留是否仍有研究空间。

- 主范围：环境/对象记忆、跨视角实例回忆、功能部件 grounding、任务条件帧/视角选择、证据保留与相关数据。
- 时间：主要查 2019–2026-11-02；根据引用链保留更早基础。arXiv 首发年份、会议/期刊年份分别记录。
- 纳入：直接相关的原始论文及作者/官方项目、代码或数据文档；多版本按论文身份合并。
- 排除：仅有二手介绍的论断、与当前检索问题无直接联系的通用 VLA 排名；纯视频生成仅保留 I3DM 作明确标注的方法参考。
- 未覆盖：所有机器人控制/VLA 训练论文、完整 SLAM/NeRF 发展史、所有视频问答和 moment retrieval 基准。它们不用于证明本题存在研究空白。

检索通过联网搜索及原始页面阅读进行，主来源为 arXiv、CVF、PMLR、作者实验室/机构、官方数据文档与作者 GitHub。没有宣称完成 Scopus、Web of Science 或 Semantic Scholar 的全库导出；搜索引擎索引可能遗漏最新或未公开工作。

## 2. 查询记录

以下保留可追溯的主题、分支和发表核查查询。还通过命中论文、已有项目条目及项目/代码链接进行向前定位；以下数量不是全部搜索结果数、去重候选数或 PRISMA 流程数字。没有保存足够完整的命中清单，因此不报告命中总量和排除比例。

### 2.1 主主题查询（16 条）

1. affordance RAG memory view image retrieval mobile manipulation
2. affordance guided keyframe retrieval cross view video memory
3. functional region visibility keyframe selection affordance
4. affordance 3d memory retrieval instance selection
5. Ego4D VQ2D visual query localization object retrieval recurrent memory
6. EAGLE episodic appearance geometry memory visual query localization
7. Ego4D VQ3D video object retrieval 3d keyframes
8. 3D Mem snapshot scene memory retrieval 2025
9. functional keyframe selection affordance information view selection robotics
10. task aware keyframe selection video memory retrieval affordance
11. 2026 affordance memory cross view keyframe retrieval
12. 2026 video memory functional object retrieval affordance benchmark
13. "Adaptive Keyframe Sampling for Long Video Understanding" arxiv
14. "Fun3DU" functional understanding scenes video
15. "TASA" "affordance" 2026
16. "EAGLE" "Episodic Appearance" arxiv

### 2.2 Affordance 分支及资源核查（15 条）

1. AFFORDMEM affordance memory retrieval robotics 3D paper
2. "3D Affordance Memories" object retrieval views paper
3. "3D Affordance Memories" robotic affordance retrieval video
4. "affordance memories" 3D visual memory robot 2025
5. "affordance memory" object-centric video scene keyframe retrieval 3D robot
6. AFFORDMEM github project scenes memory Wang He Pan
7. "3D Affordance Memories" "video" robot
8. "3D Affordance Memory" RGB video affordance memory robotics
9. "3D affordance memories" paper arxiv spatial affordance
10. "3D affordance memories" computer vision research paper video
11. "3D affordance memories" memory video object retrieval scene benchmark
12. Actron3D paper github datasets code
13. "Affordance RAG" "WholeHouse-MM" github dataset Korekata
14. "WholeHouse-MM" benchmark github
15. "Affordance RAG" Korekata code project

“3D Affordance Memories”是搜索假设词，并未据此收录一篇未经核实的同名论文；相关结果按已验证的实际标题和 arXiv/DOI 归档。搜索不到专门下载页也不等于资源一定未发布。

### 2.3 视频—机器人邻域查询（4 条）

1. robot visual memory object affordance retrieval cross view video keyframe functional region
2. robot manipulation affordance driven active view selection object memory multi view affordance visibility
3. open vocabulary robot object memory retrieval egocentric video scene graph cross view affordance
4. affordance retrieval viewpoint invariant functional region object image matching robotics

### 2.4 发表状态复核（2 条）

1. "Affordance RAG" "2706" IEEE
2. "Affordance RAG" "10.1109"

随后实际打开[作者实验室发表目录](https://smilab.org/en/publications/)及[Keio 机构记录](https://keio.elsevierpure.com/en/publications/affordance-rag-hierarchical-multimodal-retrieval-with-affordance-/)，核实 RA-L 2026 和 DOI；没有把搜索摘要单独当作最终证据。

## 3. 入选与阅读等级

最终比较 **23 篇不同论文**：既有档案 P03、P04、P11–P13，加上本次新增 P14–P31。按 canonical title、arXiv ID 和 DOI 去重；作者项目、GitHub、预印本与期刊记录不另计论文。该数字是报告所选比较集合，不能解释为领域全部相关论文数量。

| 等级 | 数量 | 实际含义 |
| --- | --- | --- |
| A | 13 | 阅读到方法及实验相关章节，足以核查本报告涉及的局部机制；不代表逐字精读全部附录或复现实验 |
| B | 10 | 仅原始摘要、作者项目页或方法概况；用于领域定位，不能支撑细节级否定判断 |

A 级：P03 AROS、P04 O³Afford、P11 AFFORDMEM、P13 STaR、P14 Affordance RAG、P15 Actron3D、P16 TASA、P18 3D-Mem、P19 SpatialMem、P20 Memory Over Maps、P21 ReMEmbR、P22 R4DSG、P24 AKS。P03/P04 沿用既有方法阅读，其余在本轮核查相关章节。

B 级：P12、P17、P23、P25–P31。全文入口存在也不自动升级为 A；逐篇来源、标题和阅读状态见 JSON 证据表。

本次新增 18 条档案均包含原始来源链接、摘要中文概述、方法示意图、贡献 bullet points 和本项目相关性。示意图由方法概述改绘，并非论文原图；摘要为概述，原始 abstract 仍见论文链接。

## 4. 对立项判断最重要的证据及限制

| 证据 | 实际核查 | 能支持什么 / 不能支持什么 |
| --- | --- | --- |
| Affordance RAG | [arXiv §III–VI](https://arxiv.org/html/2512.18987)及发表记录 | 直接否定“首次 affordance 环境物体/图像检索”主张；离线评测不含最终 ASR，真实任务含用户选择，不能夸大自动可执行性 |
| TASA | [Method、帧选择及消融](https://arxiv.org/html/2511.11702) | 功能词加权帧选择已存在。HTML Eq.(4) 有归一化形式疑点，未核对作者实现，不能声明精确复现 |
| AFFORDMEM | [§3–4](https://arxiv.org/html/2605.11616) | 有示例检索和多视角实例处理；输入用到深度/位姿，不是纯 RGB 推理基线 |
| SpatialMem | [v4 §3–5](https://arxiv.org/html/2601.14895v4) | RGB 视频到空间对象/证据记忆已有；离线静态与在线动态不可等同 |
| 3D-Mem / STaR / MoM | 原文方法、实验及作者仓库页面 | 紧凑快照与任务证据选择有直接先例；源码页面访问不等于安装、运行或复现 |
| AKS | [§3–4](https://arxiv.org/html/2502.21271) | query 已知时固定预算的相关性＋覆盖选帧已有；不直接回答 query 未知时写入 |

“未知查询下的小部位证据保留”被列为**候选研究假设**，未证明是无人研究的空白；后续正式投稿前还需围绕最终形式作引用链追踪和更新检索。没有统一重跑所有方法，因此不将跨论文结果排列为 SOTA 榜单。

## 5. 数据和资源核查边界

- 已打开 [SceneFun3D 标注](https://scenefun3d.github.io/documentation/dataset/annotations/)与[下载资产说明](https://scenefun3d.github.io/documentation/toolkit/data-downloader/)，区分扫描部件 mask、RGB/深度/位姿与待补的帧级可见性/parent instance/证据帧标签。
- 已查 [Ego4D 标注说明](https://ego4d-data.org/docs/data/annotation-guidelines/)、[Ego-Exo4D 概览](https://docs.ego-exo4d-data.org/overview/)、[EPIC/VISOR](https://epic-kitchens.github.io/VISOR/)。其现有对象或动作标注不能自动当作功能部位 GT。
- Affordance RAG / WholeHouse-MM 的专门公开下载入口和衍生许可尚未确认。Actron3D 仓库存在不代表完整运行实现已验证。SpatialMem 的完整可用代码也未验证。
- 部分项目/出版页访问受阻：TASA 项目页无法读取；EAGLE 的 AAAI 页面受阻后使用 arXiv；VISOR 旧 /site 路径失效后已核实根页面。SFI-Bench 页面未成功读取，不纳入本报告的方法比较与创新判断证据。
- 没有下载数据、安装依赖、运行模型、验证 GPU 显存需求或检查完整许可证。资源可用性栏中的“未验证”不表示“一定不可用”。

## 6. 文档维护与审核

报告经过独立只读检查；已补清固定预算下全帧参考的计费方式，以及物理存在状态与视觉证据不足的区别。论文陈述、综合推断和建议实验在正文中分开表达。

项目总控及其门户副本保持不变。原档案只在文末追加新论文及 P14 发表状态补充，不重写旧观点；门户副本仅追加同步缺失的尾部。此次文件整理不表示 pipeline 已获批准变更。
