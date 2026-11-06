# 人类示范到机器人 affordance：数据与方法简表

**2026-11-02｜组会稿备查材料。先看[简版讨论稿](组会讨论稿_视频环境记忆与机器人学习_2026-11-02.md)。**

## 1. 这个扩展可以做，但先区分两个目标

**目标A：人执行，系统建立交互记忆。** 从视频记录接触部位、移动方式、物体前后变化，再按任务检索。这一阶段可以独立研究，暂时没有机器人控制也成立。

**目标B：机器人利用这些信息执行。** 将人类交互线索映射到具体机器人，再用实际任务检验。人的手指运动与机器人夹爪/关节不同；可以用机器人数据训练策略，也可以用几何迁移、逆动力学或现有规划器实现对齐。

我们想学的“过程”至少有三层：

| 层次 | 英文关键词 | 例子 |
| --- | --- | --- |
| 在哪里交互 | Contact Region / Affordance Grounding | 抓抽屉把手 |
| 接触后怎么动 | Interaction Motion / Trajectory | 沿向外方向拉动 |
| 发生什么、下一步做什么 | State Transition / Task Sequence | 抽屉打开后才能取杯子 |

**单张 affordance map 主要表达第一层，不能单独覆盖整个流程。** 实际碰过的位置是一次交互证据，也不等于全部可行区域；没碰过的地方不应自动当作不可交互负例。

## 2. 最值得看的数据

下表中的六套主体数据都是人类图像/视频，不能把其中的人手运动直接当作目标机器人控制命令。

| 数据与原始入口 | 实际有的监督 | 适合训练什么 | 缺口与获取状态 |
| --- | --- | --- | --- |
| **[OPRA / Demo2Vec](https://github.com/kuanfang/opra)**，CVPR 2018 | 人类产品操作短视频、动作类别、另一张目标物图像上的交互点及其热图 | **看人操作→预测目标物上交互哪里/做什么** | 是点生成热图，非逐帧功能mask/3D轨迹；作者提供标注和YouTube获取脚本，部分原视频可能失效，未实测下载 |
| **[AGD20K](https://github.com/lhc1224/Cross-View-AG)**，CVPR 2022 | 静态人—物交互图与目标物图；36类功能，弱类别监督及评测点标注 | 跨视角功能区域预训练/对照 | **不是连续视频**；不能直接学步骤。官方代码页已查，但具体数据下载入口本轮未确认 |
| **[HOI4D](https://hoi4d.github.io/)**，CVPR 2022 | 第一人称RGB-D视频；动作分段、手/物体3D位姿、分割和相机参数 | **手—物三维关系、交互时序、变化表示** | 不等于现成功能部件图或完整状态机标签；官网有分项下载，适合先查少量训练片段 |
| **[EPIC-KITCHENS + VISOR](https://epic-kitchens.github.io/VISOR/site)**，VISOR NeurIPS 2022 | 厨房连续视频、原始动作/叙述；VISOR手和交互物体mask、手—物关系 | **自然任务中的对象跟踪、交互事件、历史来源回忆** | 对象mask不是部位affordance；稠密插值与人工标注分开。作者给[标注下载](https://doi.org/10.5523/bris.2v6cgv1x04ol22qp9rm9x2j6a7) |
| **[EgoDex](https://github.com/apple-aiml-research/ml-egodex)**，2025首发 / ICLR 2026 | 大规模第一人称操作视频、语言、采集时跟踪的手/手指/上身3D位姿 | 人类运动表征、任务条件动作预训练 | 不含现成dense affordance或目标机器人动作；官方列训练/测试下载包，整体数据量大 |
| **[HOT3D](https://facebookresearch.github.io/hot3d/)**，CVPR 2025 | 同步头戴多视角、手/物体/相机3D位姿与模型；部分设备是灰度相机 | 精细跨视角手—物几何和位姿一致性 | 不直接提供接触力或完整功能标签；官方有工具与下载说明，不能默认所有流都是RGB |

VISOR还包含 **Where Did This Come From?** 来源定位测试，与你“记住物体之前在哪里”的需求很接近；因此物体来源回忆本身已有先例。可进一步研究局部功能证据与变化信息是否帮助后续动作。[官方任务说明](https://epic-kitchens.github.io/VISOR/site)

**优先选择：** 做时序三维记忆，先查HOI4D；做自然任务过程，先查EPIC/VISOR；只验证人示范到交互区域，OPRA最直接，但先确认原始视频可用。AGD20K作为静态对照，EgoDex/HOT3D按后续运动或几何需求再引入。无须同时下载六套。

## 3. 三篇能说明“怎样学”的方法

- **[VRB](https://arxiv.org/html/2304.08488)，CVPR 2023：** 从人视频提取接触位置和接触后手腕路径，训练场景图像到交互预测。**贡献：把“哪里碰、之后怎么动”作为机器人学习接口。** 不是自动学习完整任务步骤；其当前图像推理也不等于运行时读取长视频。官方已有[推理代码和权重入口](https://github.com/shikharbahl/vrb)，不能因独立数据集仍标Coming Soon就认为代码未发布。
- **[Actron3D](https://arxiv.org/html/2510.12971)，2025首发：** 从少量人类RGB示范建立物体中心的几何/接触/运动函数，再检索对齐到机器人目标场景。**贡献：三维功能与运动表示的迁移。** 机器人部署有RGB-D、抓取和控制模块，不能称任意视频即可通用执行。
- **[EgoMimic](https://arxiv.org/html/2410.24221)，2024首发：** 对齐人类第一人称手部数据与机器人遥操作数据，共同训练策略。**贡献：让两种来源共享学习表征，同时用机器人标签训练其动作输出。** 不是显式affordance模型；也不要求人、机器人逐帧配对。作者的[数据文件列表](https://huggingface.co/datasets/gatech/EgoMimic/tree/main)有人和机器人分别采集的轨迹。

组会稿中的 **RAM** 则代表“检索接触点/路径→几何迁移→已有规划控制”的路线。它与VLA共训练是两种不同实现选择，不能把两者当成同一模型。[RAM原文](https://arxiv.org/html/2407.04689)

## 4. 对你当前项目的建议

先用人类交互视频研究：

**视频 → 对象与接触/运动线索 → 交互前后变化 → 环境证据记忆 → 按任务检索。**

第一阶段评价同实例关联、部位/轨迹预测、变化记录，以及遮挡后的证据检索；接触和变化标签不足时，使用人工核验的小样本或明确标记的伪标签。相机运动与手/物体运动需要分离。

第二阶段再选择一种机器人接口：**现有VLA适配**，或 **RAM/Actron3D式几何迁移与规划**。前者通常需要同步机器人动作数据；后者在限定任务中可不采目标任务示范，但仍需经过验证的几何和本体控制。两者最终都用闭环任务评价。

如果暂时只有人类视频，可以完成第一阶段的研究；对“机器人学会了任务”的主张，需要后续执行证据。

<details>
<summary>来源与核查记录（备查，无须组会主讲）</summary>

检索截至2026-11-02，属于针对当前问题的定向核查，不是穷尽综述。原始来源为CVF/PMLR/arXiv、作者项目和官方数据页；没有下载数据、运行模型或确认本机可训练。库文件与权重链接存在，不等于完整复现已通过。

- RAM、VRB、Actron3D、EgoMimic核查方法与实验；数据集主要核查原文标注说明/摘要、官方schema及下载页面，不称全部全文精读。
- OPRA的[原始论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Fang_Demo2Vec_Reasoning_Object_CVPR_2018_paper.pdf)说明交互点标注；AGD20K的[原始论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Luo_Learning_Affordance_Grounding_From_Exocentric_Images_CVPR_2022_paper.pdf)需区分弱监督训练与评测热图。
- HOI4D、VISOR官网标数据CC BY-NC 4.0；EgoDex作者标数据CC-by-NC-ND；HOT3D有专门数据协议及额外手模型条件。正式使用按各数据条款核对，代码许可证不能代替素材许可。
- EgoMimic的HF根页面读取受限，但作者链接的tree/main文件列表可读。RAM代码库有执行/检索文件和数据链接，也保留TODO；只记录已见资源，未下载验证。
- EgoDex测试包可用于检查格式；训练应使用官方训练划分。不得把测试集并入训练来降低下载成本。
- 本次新增文献归档P67–P73；AGD20K、VRB及Actron3D沿用P12、P30、P15，并在文末追加澄清。

主查询：OPRA Demo2Vec human demonstration affordance dataset contact points official；EgoDex human manipulation dataset Apple tracking license video；human robot affordance transfer RAM Kuang 2407.04689 dataset；EPIC KITCHENS VISOR HOI4D affordance training interaction annotation。

分支查询记录：

- AGD20K dataset affordance grounding human object interaction annotation affordance ground truth github
- OPRA object part affordance video dataset human demonstrations affordance regions dataset
- HOI4D dataset egocentric 4D human object interactions annotation rgbd action contact download license
- EPIC KITCHENS VISOR dataset hand object segmentation video annotations license downloads
- EgoDex dataset Apple 2025 egocentric dexterous human manipulation hand pose tracking dataset access license
- HOT3D hand object tracking 3D dataset egocentric Meta Aria Quest annotations dataset license
- AGD20K github affordance grounding dataset download train exo ego 20k license
- OPRA Demo2Vec dataset github download affordance 2018 Fang
- "AGD20K" dataset download github Luo Xu 2022
- AGD20K dataset train labels annotation ego images ground truth test point annotation
- "OPRA" "Demo2Vec" dataset unavailable download Google drive
- EgoDex Apple dataset license BY NC ND download aws
- VRB affordances from human videos as versatile representation for robotics 2304.08488 contact trajectory robot dataset
- RAM kuang25a retrieval affordance transfer manipulation human videos robot learning
- Actron3D human video affordance robot 3d action affordance
- EgoMimic human video robot data alignment action label training human robot 2025
- Actron3D github code dataset neural affordance function 2510.12971
- Egomimic github gatech dataset hand robot code
- RAM retrieval affordance transfer github memory dataset kuang

</details>

