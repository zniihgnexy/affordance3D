# Affordance 特征写入记忆：检索与证据记录

**日期与截止：2026-11-04｜定向方法核查，非穷尽系统综述。**

对应[构建范例与非机器人评估](Affordance特征写入记忆_构建范例与非机器人评估_2026-11-04.md)。本轮不修改总控、Pipeline v3 或此前独立记忆／VLA综述；新候选设计并非项目方向批准。

## 1. 问题、筛选与阅读等级

问题：有哪些将学习特征加入空间／视频记忆的明确实例？怎样借鉴它们加入 affordance 特征，且用不依赖机器人执行的实验评价？

优先纳入：能查清特征来源、写入载体、关联／更新与查询步骤，并有独立定位／记忆实验的原始论文。选 AMEGO、ConceptFusion、CLIP-Fields 为构建对照，既有 AFFORDMEM 为功能记忆直接近邻；不是“最新四篇”或跨协议 SOTA 排名。既有 KeyRec、R4DSG 等仍见上一轮综述，本轮不重复收录。

A＝已核查相关方法与实验章节；不代表复现、下载数据或核验所有结果。B＝仅摘要／部分方法。下列四篇均 A，均未安装、运行或复现。作者项目／代码页只用于来源和可用入口检查，不能代替运行验证。

## 2. 实际查询

主查询：

1. `"AMEGO" "Active" "Memory" egocentric video`
2. `"Grounding by Remembering" affordance memory`
3. `"ConceptFusion" open set multimodal 3D mapping feature fusion`
4. `video memory affordance implicit feature interaction memory evaluation`
5. `"CLIP-Fields" "Robotic Memory" paper`
6. `"Implicit Affordance Acquisition" "Causal Action-Effect" modeling`

并行核查分支的查询：

1. `AFFORDMEM 2605.11616 affordance memory 3D memory feature experiment`
2. `ConceptFusion open-set multimodal 3D mapping 2302.07241 CLIP feature fusion evaluation`
3. `CLIP-Fields weakly supervised semantic fields robotic memory arxiv 2022 semantic retrieval`
4. `AMEGO Active Memory from long EGOcentric videos ECCV 2024 AMB benchmark object location interaction dino`
5. `AMEGO Active Memory Ego videos benchmark AMB tracklet object state memory 2024`
6. `AMEGO active memory egocentric videos github AMB online object location`

题名／URL直达与章节内查找另计；不记录未导出的全库命中数或筛选比例。两位阅读协作者分别核查空间记忆与视频记忆，主整理核对论文方法／实验并综合，不将协作者建议写成论文结果。

## 3. 核心证据与容易误读的地方

| 文献 | 核查来源／章节 | 构建证据 | 评价证据与边界 |
| --- | --- | --- | --- |
| P114 AMEGO | [原文](https://arxiv.org/html/2409.10917) §3–5及补充；[ECCV补充](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02032-supp.pdf) | 手物交互 tracklet；DINOv2 对象特征，SWAG 地点特征；实例／地点聚类和时间线 | AMB五选一准确率；片段单独评价仅两视频，AIoU-P／GT、ΔN，补充有身份切换；不是通用自由文本QA或有界存储证明 |
| P115 ConceptFusion | [原文](https://arxiv.org/html/2302.07241) §IV-A–D、§V | 像素对齐的局部／全局特征；深度＋位姿投影；置信权重融合到三维表面 | UnCoCo三维定位mIoU及IoU阈值命中；语义分割mAcc／频率加权mIoU；不是动态事件历史 |
| P116 CLIP-Fields | [原文](https://arxiv.org/html/2210.05663) §IV–V；[RSS版](https://www.roboticsproceedings.org/rss19/p074.pdf) | 坐标编码＋预测头；CLIP／Detic／Sentence-BERT教师；对比目标训练场景特征场 | HM3D少样本实例／语义、零样本语义与噪声鲁棒性；少样本有人工标注，评价查询场仍用深度／位姿；视图定位含定性展示 |
| P11 AFFORDMEM | [原文](https://arxiv.org/html/2605.11616) §3.2–3.3、§4.1–4.3、限制 | 来源标注区域投影为示例；类别字符串索引；冻结VLM＋SAM3；目标候选池和空间图 | SceneFun3D AP25/AP50、AR25/AR50、mIoU；部分消融是代表性子集；非新学习的隐式功能向量检索 |

额外发现：[Implicit Affordance Acquisition via Causal Action–Effect Modeling in the Video Domain](https://aclanthology.org/2023.ijcnlp-main.55/)，IJCNLP-AACL 2023。仅核查摘要与部分内容，B。与功能编码／动作后果预训练有关，不提供本轮所需的记忆写入维护范例，因此不列为四篇主文献，也不借其声称本项目已具备因果能力。

## 4. 本轮新增及保留范围

- 持续档案新增 P114–P116，P11只在文末追加核查。未重排或覆盖旧条目。
- 主文中的候选数据结构、时间状态版本、写入时与查询时生成的消融、公平容量对照，均为本项目建议；不能归因于四篇论文已做。
- 本轮不承诺数据集足以覆盖全部功能／动态标签；具体资源还需按标注类型及使用权限筛选。
- 暂不接 VLA 时，评价无需机器人执行，但研究也不能据此声称真实动作可行或机器人泛化已得到证明。
