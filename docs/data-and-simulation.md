# 数据与仿真平台

**固定维护入口** · 更新日期：2026-10-29

本页汇总数据集、仿真平台、模型与实验组合的调研资料。表格是候选清单，不代表数据已经下载、平台已经部署或硬件兼容性已经验证。

## 当前资料

- [Pipeline v2 数据与仿真平台调研表](outputs/pipeline_v2_research_20261029/Pipeline_v2_数据与仿真平台调研.xlsx)：包含优先实验组合、数据候选、仿真平台、模型与基线、硬件与运行、训练与评测六个工作表。
- [早期数据、平台与实验组合表](outputs/research_20261028/机器人视频三维记忆_数据平台与实验组合.xlsx)：包含数据集、仿真平台与模型、实验组合三个工作表。
- [首个对照实验方案](research/首个对照实验_遮挡下交互预测_v1.md)：说明候选 ManiSkill 自定义场景及需要自行实现的实验。

## 可视化资料

Pipeline v2 调研：[优先实验组合](outputs/pipeline_v2_research_20261029/优先实验组合.png) · [数据候选](outputs/pipeline_v2_research_20261029/数据候选.png) · [仿真平台](outputs/pipeline_v2_research_20261029/仿真平台.png) · [模型与基线](outputs/pipeline_v2_research_20261029/模型与基线.png) · [硬件与运行](outputs/pipeline_v2_research_20261029/硬件与运行.png) · [训练与评测](outputs/pipeline_v2_research_20261029/训练与评测.png)。

早期调研：[数据集](outputs/research_20261028/数据集.png) · [仿真平台与模型](outputs/research_20261028/仿真平台与模型.png) · [实验组合](outputs/research_20261028/实验组合.png)。

## 实验条件与维护约定

根据[项目总控](项目总控.md)，学校 GPU 通过 SSH 使用，另有配备 NVIDIA 5060 的仿真设备；具体 GPU 规格、驱动、显存、平台安装及渲染可行性仍待确认。平台和数据使用前应检查其实际模态、相机位姿、动作格式、许可和访问条件。

后续新增调研或安装验证时，更新本入口和相应表格，标注日期、来源、可用性及验证状态；保留早期记录以便追溯。
