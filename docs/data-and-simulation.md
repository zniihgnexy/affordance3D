# 数据与仿真平台

**固定维护入口** · 更新日期：2026-11-06

本页汇总数据集、仿真平台、模型与实验组合的调研资料。表格是候选清单，不代表数据已经下载、平台已经部署或硬件兼容性已经验证。

开始测试时先读[最新研究记录与实验交接](research/实验前研究交接_2026-11-06.md)，再按下方Mac→Isambard计划安排。近期资源以本周仅有Mac、后续申请远程GPU为准；历史设备条件不视为当前可用性保证。

## 最新快速验证安排 — 2026-11-05

- [环境记忆 Affordance 层：Mac → Isambard-AI 快速验证计划](research/环境记忆Affordance层_快速验证实验计划_Mac_Isambard_2026-11-05.md)：依据本周仅有 M1/16GB Mac、后续远程 GPU 的条件，安排 SceneFun3D 小子集、UMD 备用、本地几何与记忆检查、远程轻量训练及本地可视化。仅为计划，尚未下载或训练。

## 当前资料

- [Pipeline v3：非仿真 affordance 数据与离线测试](outputs/pipeline_v3_research_202611/Pipeline_v3_Real_Data_Affordance_Pilots.xlsx)：2026-11-04 的真实数据候选、标注类型和四项离线测试。
- [Pipeline v3：数据、仿真与 VLA 候选](outputs/pipeline_v3_research_202611/Pipeline_v3_Core_Data_Simulation_VLA.xlsx)：2026-11-03 的数据、平台、模型和组合调研；VLA 与仿真不是首轮实验的前置条件。
- [环境记忆数据与仿真调研表（英文）](outputs/pipeline_v3_research_202611/Environment_Memory_Data_and_Simulation_English.xlsx)：英文资源整理，内容与适用条件需在使用前重新核验。
- [环境记忆 Affordance 特征层：可行性与实现建议](research/环境记忆Affordance特征层_可行性与实现建议_2026-11-05.md)：以环境为主体的补充研究建议，不自动替换 Pipeline v3。
- [Pipeline v2 数据与仿真平台调研表](outputs/pipeline_v2_research_20261029/Pipeline_v2_数据与仿真平台调研.xlsx)：包含优先实验组合、数据候选、仿真平台、模型与基线、硬件与运行、训练与评测六个工作表。
- [早期数据、平台与实验组合表](outputs/research_20261028/机器人视频三维记忆_数据平台与实验组合.xlsx)：包含数据集、仿真平台与模型、实验组合三个工作表。
- [首个对照实验方案](research/首个对照实验_遮挡下交互预测_v1.md)：说明候选 ManiSkill 自定义场景及需要自行实现的实验。

## 可视化资料

Pipeline v2 调研：[优先实验组合](outputs/pipeline_v2_research_20261029/优先实验组合.png) · [数据候选](outputs/pipeline_v2_research_20261029/数据候选.png) · [仿真平台](outputs/pipeline_v2_research_20261029/仿真平台.png) · [模型与基线](outputs/pipeline_v2_research_20261029/模型与基线.png) · [硬件与运行](outputs/pipeline_v2_research_20261029/硬件与运行.png) · [训练与评测](outputs/pipeline_v2_research_20261029/训练与评测.png)。

早期调研：[数据集](outputs/research_20261028/数据集.png) · [仿真平台与模型](outputs/research_20261028/仿真平台与模型.png) · [实验组合](outputs/research_20261028/实验组合.png)。

## 实验条件与维护约定

[项目总控](项目总控.md)保留了先前提供的学校远程 GPU 和 NVIDIA 5060 仿真设备条件；2026-11-05 的最新实验计划以当周只有 Mac、后续申请远程 GPU 为准，不能保证早先设备目前可用。具体 GPU 规格、驱动、显存、平台安装及渲染可行性仍待确认。平台和数据使用前应检查其实际模态、相机位姿、动作格式、许可和访问条件。

后续新增调研或安装验证时，更新本入口和相应表格，标注日期、来源、可用性及验证状态；保留早期记录以便追溯。

## English guide — 5 November 2026

- [Video-Based Environment Memory: Pipeline, Datasets and Simulation](research/Pipeline_Datasets_and_Simulation_Guide_2026-11-05.md): the current environment-centred memory design, functional and temporal datasets, controlled simulation options, official images, and an evaluation route that can precede VLA integration.
