# Literature Review v2：面向机器人 VLA 的隐式交互记忆

**检索日期：2026-10-28**  
**范围与状态：** 聚焦视频/几何、交互经验、隐式记忆及机器人 VLA；这是研究设计所需的针对性检索，不是穷尽式系统综述。论文效果是作者在各自协议下报告的结果；新颖性需继续核验全文、代码和对照。未下载数据、运行论文代码或复现实验。

## 1. 直接结论

- “显式几何仅用于训练、推理时用隐式几何记忆驱动 VLA”已有很近工作（GeoMem-VLA）。
- “把已执行动作与后续观察的结果写入隐式 VLA 记忆”已有先例（MemBodied）。
- “将 affordance 训练监督转入 VLA 的隐式特征”也已有先例（AffordVLA）。
- 因此，不能把几何监督、动作结果记忆、双流 slots 或 VLA 接入单独当作新贡献。候选问题是：**局部机器人—物体—环境关系如何与已发生交互结果形成可迁移记忆，并改善遮挡、目标变化和未见几何组合下的同一策略？** 这仍是待验证假设，不是已证明空白。

## 2. 最重要的直接近邻

| 工作 | 输入、表示与训练 | 与 v2 的重合 / 需要核查的差异 |
| --- | --- | --- |
| [GeoMem-VLA](https://arxiv.org/abs/2610.29866)；2026-10 预印本 | 训练时用冻结几何教师的 3D point map 和未来末端轨迹监督几何 tokens；检索并更新时序记忆，生成机器人动作。推理不要求几何教师或显式重建。 | 与“训练用显式 3D、运行用隐式记忆、接 VLA”直接重合。需要全文核查其局部物体—环境关系、交互结果和对照；不能仅凭摘要宣称缺少这些。代码和复现资源需逐项确认。 |
| [MemBodied](https://arxiv.org/abs/2609.28256)；2026-09 预印本 | 固定大小关联记忆及起始场景 anchor；先读记忆出动作，再将已执行动作和后续观察写入记忆；输入 VLA。 | “action → outcome → 隐式记忆”已存在。v2 的局部关系与几何条件须在匹配动作数据、写入频率和计算预算下带来可验证增益。 |
| [UniMPA](https://arxiv.org/abs/2609.11875)；2026-09 预印本 | 将未来状态预测、历史视觉—动作记忆及动作生成统一为 action-grounded transition 建模。 | “记忆 + 结果预测 + 行动”亦已有直接近邻；需全文检查数据、memory bank 和交互级关系。 |
| [AffordVLA](https://arxiv.org/abs/2605.17517)；2026-05 预印本 | 训练时用任务条件的视觉 affordance teacher 对齐 VLA 隐式表征，推理无外部 teacher。 | 单独把 affordance 蒸馏到隐空间并非新贡献；此处优先研究可随时间更新的关系和已发生的交互。 |
| [AffordanceWAM](https://arxiv.org/abs/2609.22332)；2026-09 预印本 | 联合未来 RGB、时空 affordance 标量/热图和连续动作；人类视频监督预测流、机器人轨迹监督动作。 | 将未来和 affordance 的隐式特征接动作也已有先例；其方法使用显式 affordance 数据，需要按完整协议比较。 |
| [Embodied-SlotSSM / LIBERO-Mem](https://ojs.aaai.org/index.php/AAAI/article/view/37337)；AAAI 2026 | 维持时序一致的 object slots，并以关系编码对齐动作预测。 | 时序 slots 和关系网络本身也不是新贡献；需要比较非类别绑定局部几何及交互后果。 |
| [REKI](https://kyutai.org/reki/)；2026-10 | 在动作专家中用 register tokens 缓存观察历史，动作学习联训；项目页提供模型和代码链接。 | 实用的隐式观察历史基线；并非“已经实现 v2 的局部几何交互写入”。模型规模和资源需实测。 |
| [GeomVLA](https://arxiv.org/abs/2609.13812)；2026-09 | 将 RGB、深度与标定用于机器人坐标系的 3D token，场景运动 token 参与动作生成。 | 几何—运动—动作直接对齐已有先例；深度和标定在其推理条件中，不能说它等同纯 RGB。 |
| [RAM](https://proceedings.mlr.press/v270/kuang25a.html)；CoRL 2025 | 从多来源示范提取 affordance memory，用语言检索并迁移到机器人。 | affordance 记忆和跨对象迁移不是空白；记忆形式、交互和监督需具体比较。 |

更宽相关工作：[O³Afford](https://proceedings.mlr.press/v305/tian25b.html) 研究物体—物体 3D affordance 和规划；[ActAIM](https://actaim.github.io/) 从深度和交互学习多种 goal-conditioned interaction modes；[PointWorld](https://pointworld-iclr26.github.io/) 预测动作条件的三维场景点流。[AROS](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1076780/full) 是可选的局部几何参考，不是 v2 的必需组件。

## 3. 与 SpatialMem 的关系

[SpatialMem](https://arxiv.org/html/2601.14895v4) 将三维结构锚点、对象节点和证据组织为显式可查询记忆；OpenEQA 的同评测设置中，文本序列化 35.1、blind 44.5、记忆检索 8 帧 62.0、均匀采样 8 帧 61.6。对象覆盖、几何估计、选择与消费接口均可能影响表现；结果不证明显式记忆一定比隐式差。

v2 希望研究不同问题：保留局部几何和真实动作后变化的可学习隐式状态，直接服务 action prediction。SpatialMem 可作为研究背景与可能的显式参考；它原任务、输入和评测与操控不同，不能不做适配就拿论文数字作为同任务基线。

## 4. 数据、动作和 benchmark 的实际约束

- [ManiSkill](https://github.com/mani-skill/ManiSkill)：用于自建受控仿真，采集 RGB-D、相机信息、实际执行动作和后果；不自带本项目的配对局部关系 benchmark。其[示范文档](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html)提示默认演示常不含视觉，需要回放；回放不总能从所保存状态精确恢复，必须验证。
- [RoboMME](https://robomme.github.io/)：16 个记忆相关操作任务，提供不同类型的时间记忆评测；[训练数据格式](https://github.com/RoboMME/robomme_benchmark/blob/main/doc/h5_data_format.md)列 RGB-D、相机内外参及末端绝对动作。具体在线模式的[允许观察](https://github.com/RoboMME/robomme_benchmark/blob/main/doc/env_format.md)需按配置记录；增加深度不应默称 RGB-only 比较。
- [MIKASA-Robo-VLA](https://mikasarobo.github.io/concepts.html)：90 个语言条件操作任务，公开机器人动作轨迹。其标准数据/评测以 RGB、机器人状态和动作等为主；不能假设公共轨迹自带深度和位姿。如果训练使用额外几何真值，须通过可验证的回放/采集获得并单独报告。推理 RGB-D 的额外模态与官方 RGB 协议分开比较。
- [SceneFun3D](https://scenefun3d.github.io/)：真实三维场景功能区域和语言标注，可作静态 affordance grounding 辅助测试；它不能单独证明机器人时序记忆和控制。
- [SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla) 是较小的可选对接起点，但 memory adapter 并非现成能力，需要改模型和用机器人轨迹训练；官方示例的硬件/耗时不等于学校 GPU 或 5060 的实际能力。[OpenVLA](https://github.com/openvla/openvla) 的官方训练要求较高，切勿假设 5060 可直接训练。

## 5. 直接支持下一步决策的问题

1. GeoMem-VLA 的时序几何 tokens 与局部 object–environment 关系学习有何具体差异？需检查教师监督对象、可观测场景、写读条件和消融，不能只依据摘要。
2. MemBodied 已利用动作与结果写入；加入相同预算的局部关系条件是否提供实际增益？
3. 单帧、普通视频历史、隐式几何记忆、纯交互记忆以及联合方法，在相同机器人数据和 action head 下怎样比较？
4. 时序/几何记忆带来的收益能否在无类别标签、未见几何组合、意图变化和至少一项 memory-dependent 的机器人闭环任务中被重现？
5. 几何/交互教师、接触/失败记录是额外监督；报告其成本，并对照同等计算量的非几何辅助训练。

机制与实验设计见 [Pipeline v2](隐式交互记忆_Pipeline_v2.md)。
