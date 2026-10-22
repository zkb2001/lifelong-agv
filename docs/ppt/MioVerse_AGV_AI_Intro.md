# MioVerse AGV 项目介绍 PPT 讲稿（40 页 · 约 10–15 分钟）

对应文件：`docs/ppt/MioVerse_AGV_AI_Intro.pptx`  
生成脚本：`docs/ppt/build_mioverse_ppt.py`

---

## 01 封面
仓库多 AGV 终身路径规划与任务调度 · MRSC × StageNet × FACN

## 02 目录
竞赛 / 评分 / 演进 / AI 架构 / 实验 / 消融 / MCP / 总结

## 03 竞赛背景
Siemens Xcelerator · MioVerse · 决赛 AI/LLM 40%

## 04 问题定义
20×20 仓、12 AGV、~100 任务、6 取货站、16 分拣口 + 场景图

## 05 关键约束
单车单货、FIFO、站点互斥、碰撞、运动学、急件

## 06 初赛评分
送达 / 急件 / 碰撞 / 同分规则

## 07 决赛 AI 40%
专家分权重表 + 叙事口令（姿态/油门/红绿灯）

## 08 核心难点
吞吐 vs 冲突、终身流、异构场景、可解释、无 map-id、墙钟

## 09–10 系统演进
Baseline → ECBS → Hier → RL-RH-PP → Horizon → MRSC+FACN

## 11 基线数字
PIBT / M0 / PP / ECBS / Hier 对照表

## 12 为何引入 AI
搜索慢、纯学习难保证 VALID → 混合栈

## 13–16 RRSC/MRSC 与 Meeting
三层架构图、AI 管线图、六步 Meeting、pipeline 示意图

## 17 AI 模块职责表
StageNet / FACN / ASA·AFA / PGCP·PGFA / RuleNet / CGPN·TERN

## 18–19 StageNet + ASA
网络结构图 + 几何授权门控

## 20–24 FACN 专章
概念、示意图、AFA、v2 结果表、SH20 案例

## 25–27 协议与探索模块
PGCP/PGFA、RuleNet 负结果、训练方法汇总

## 28–29 主结果
vs FULL −9%；SH01–40 VALID 40/40

## 30–32 消融
概念图、掉队表、重要性排序（PGCP > StageNet > PGFA > FACN > RuleNet）

## 33–34 对照与墙钟
早期 vs Hier；ECBS/Hier vs MRSC wall

## 35–36 MCP/LLM 与可解释性

## 37–38 局限与 Roadmap

## 39–40 Takeaways + Q&A

---

### 讲解节奏建议
| 段落 | 页码 | 时间 |
|------|------|------|
| 背景与问题 | 1–8 | 2.5 min |
| 演进与动机 | 9–12 | 1.5 min |
| **AI 架构（重点）** | 13–24 | 4.0 min |
| 结果与消融 | 25–34 | 3.0 min |
| LLM / 总结 | 35–40 | 1.5 min |
