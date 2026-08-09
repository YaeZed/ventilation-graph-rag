# Task Plan: Meeting PPT Report

## Final Status Update

- 主体 PPT 已完成，最终文件为 `projects/meeting-ppt-report/exports/meeting-report-stage-media-ready_20260608_101406.pptx`。
- 最终版本共 23 页，包含阶段汇报主体、7 个系统截图占位页位置和 1 个演示视频占位页。
- 第 17-21 页已经用 PPTX 形状层叠加媒体占位框，用户后续可直接在 PowerPoint 中替换，也可把素材放入 `sources/screenshots/` 和 `sources/demo-video/` 后继续自动嵌入。
- 验证已完成：`check-plan` 为 `ok=120, warn=0, error=0`，最终 PPTX 回读确认 23 页和关键占位文本存在。

## Goal

基于原开题答辩 PPT 和当前 `ventilation-graph-rag` 辨识系统，制作一份面向会议汇报的阶段成果 PPT。汇报重点从“研究设想”升级为“系统已实现能力、关键技术路径、阶段性验证结果、后续工作计划”。

## Output Target

- 保留原文件不动：`C:/Users/Administrator/Desktop/研究生期间材料/论文/毕业论文/开题/张明俊-开题答辩PPT.pptx`
- 建议新文件名：`张明俊-煤矿通风智能辨识系统阶段汇报PPT.pptx`
- 建议输出目录：与原 PPT 同目录，或另建 `会议汇报/` 子目录。

## Constraints

- 用户提供了 `ppt-master` GitHub 地址：`https://github.com/hugohe3/ppt-master`。已从该仓库安装到本地 `D:/codex-home/skills/ppt-master`，但当前会话的可用 skill 列表不会动态刷新；本轮可按本地 `SKILL.md` 和 `workflows/template-fill-pptx.md` 执行，后续新会话需重启 Codex 才会正式显示该 skill。
- 本任务匹配 `ppt-master` 的 `template-fill` 工作流：使用原生 PPTX 作为模板库，选择/复用/重排合适页面，直接替换 OOXML 文本生成可编辑 PPTX。该 workflow 不走 SVG 转换。
- 不覆盖原开题 PPT；所有修改生成新副本。
- 继续复用原 PPT 的学校/学院风格、封面信息、章节切换页和已有实验平台素材。
- 会议汇报不应再停留在“拟开展”，需要明确区分：已完成、正在验证、仍有风险。
- 技术表述要服务听众理解：先讲解决了什么问题，再讲 GraphRAG、Neo4j、Milvus、Qwen3.5-Omni、Django/Vue 的作用。

## Proposed Deck Structure

建议控制在 18-22 页，适合 8-12 分钟汇报。

| Section | Slides | Purpose |
|---|---:|---|
| 1. 汇报封面 | 1 | 从“开题申请答辩”改成“阶段成果汇报/会议汇报” |
| 2. 问题与目标 | 2-3 | 保留煤矿通风安全背景，但压缩长段落，突出“数值预警、图像识别、法规依据割裂” |
| 3. 总体方案 | 4-6 | 展示系统从法规知识、图谱/向量检索、多模态识别到前端报告的闭环 |
| 4. 已实现系统 | 7-11 | 展示 RAG 核心、Cypher 模板、多图识别、传感器融合、SSE 进度、用户/团队/统计/设置模块 |
| 5. 关键技术突破 | 12-15 | 讲清楚“法规可追溯”“图-文-数融合”“多图联合分析”“请求级模型配置” |
| 6. 阶段验证 | 16-18 | 放已通过的构建/接口/烟测结果和典型问答示例 |
| 7. 问题与计划 | 19-20 | 诚实说明真实图片准确率、生产部署、BYOK 密钥策略、数据质量仍需验证 |
| 8. 总结 | 21 | 一页结论：系统原型已从方案进入可交互验证阶段 |

## Slide-Level Rewrite Plan

| Original | Action | Reason |
|---|---|---|
| 1 封面 | 复用版式，改标题与会议属性 | 原封面视觉和身份信息可复用，但“开题申请答辩”不适合当前会议 |
| 2/5/8/13 提纲 | 合并为 1 页新目录 | 原 PPT 多个章节页适合答辩，不适合会议汇报节奏 |
| 3 背景 | 保留核心背景，压缩为问题链 | 会议听众需要快速知道系统价值，不需要大段综述 |
| 4 研究现状 | 改成“现有方法不足” | 从文献综述转为产品/系统问题陈述 |
| 6 理论意义 | 改成“研究目标与系统定位” | 让听众理解系统要完成的任务边界 |
| 7 创新点 | 保留并更新为当前实现创新 | 已从 Qwen2.5-VL 方案升级到 Qwen3.5-Omni + GraphRAG + 传感器融合 |
| 9-12 研究内容/技术路线 | 改为当前系统架构与数据流 | 原设想内容要落到已实现模块 |
| 14-15 实验设想 | 保留实验平台素材，补充当前 Web 系统截图和识别流程 | 证明不只是实验设想，已有可交互系统 |
| 16 困难与方法 | 改成“当前风险与下一步” | 会议汇报要体现判断力和后续路线 |
| 17 致谢 | 保留 | 只需改底部说明为会议汇报 |

## Visual Asset Plan

- 复用原 PPT 母版、主题、封面风格和章节页视觉。
- 复用开题目录 `images/` 下的实验平台、风机、风速传感器、技术路线图、架构图等素材。
- 新增当前系统截图：
  - 主对话页：文本问答 + Markdown 报告。
  - 多图片上传与预览。
  - 传感器数据录入/CSV 粘贴。
  - Agent 步骤/SSE 进度。
  - `/stats` 统计面板。
  - `/settings` 模型服务配置。
- 新增/重画 1 张核心架构图：
  - 输入：文本问题、现场图片/多图、传感器数据。
  - 知识层：Neo4j 图谱、Milvus 向量库、Concept 概念层、Cypher 模板。
  - 模型层：Qwen3.5-Omni 观察/分析、Qwen-Plus 生成。
  - 输出：隐患定性、法规依据、整改建议、可追溯报告。

## Execution Phases

| Phase | Status | Scope |
|---|---|---|
| 1 | complete | 读取原 PPT 文本结构、素材目录和当前项目状态 |
| 2 | complete | 安装并读取本地 `ppt-master` skill；确认使用 `template-fill-pptx` workflow |
| 3 | complete | 创建 `ppt-master` 项目目录：`sources/`、`analysis/`、`exports/`、`validation/` |
| 4 | complete | 用 `template_fill_pptx.py analyze` 抽取原 PPT 的 `slide_library.json` |
| 5 | complete | 根据 `slide_library.json` 写页面选择依据和 `fill_plan.json`，允许重复/重排原页面 |
| 6 | complete | 运行 `check-plan`，根据容量报告压缩标题/正文或调整页面选择 |
| 7 | complete | 运行 `apply` 生成新 PPTX，默认加淡入转场，嵌入演讲备注 |
| 8 | complete | 用 `ppt_to_md.py` 回读生成文件，验证页数、标题、正文和备注可读 |
| 9 | pending | 如需真实系统截图，另行启动 Django/Vite 并补充截图；`template-fill` v1 不直接替换图片，图片替换需作为后续直接 PPTX 操作 |
| 10 | pending | 给出最终 PPTX 路径、修改说明和剩余人工检查项 |

## Acceptance Criteria

- 生成新 PPTX，原 PPT 不被覆盖。
- 输出位于 `ppt-master` 项目目录的 `exports/` 下，文件名带时间戳。
- 汇报逻辑从“开题设想”变为“阶段成果汇报”。
- 至少体现当前系统的 6 个真实能力：GraphRAG、法规可追溯、多图识别、传感器融合、Web 前端交互、用户/团队/统计/模型配置。
- 技术路线页与当前代码事实一致，避免继续写 Qwen2.5-VL 或 FAISS 作为主方案。
- 每页文字密度降低，优先使用图、流程和短句。
- `check-plan` 容量风险已处理或明确接受；`ppt_to_md.py` 回读可看到关键标题与正文。
- 最终文件可在 PowerPoint 中打开，图片不丢失，版式不明显错位。

## Risks

| Risk | Mitigation |
|---|---|
| `ppt-master` 当前会话未动态加载 | 已安装到本地目录，本轮直接读取本地 workflow 执行；后续重启 Codex 后会作为正式 skill 出现 |
| `template-fill` v1 不支持图片替换 | 先完成可编辑文本版 PPT；真实系统截图如必须加入，则后续用直接 PPTX 操作或手动插入 |
| PowerShell 输出中文乱码 | 从 PPTX XML 和项目文档提取后人工整理中文，不直接复制乱码输出 |
| 当前系统截图需要运行服务 | 若服务未启动，先用已有素材和静态架构图；需要真实截图时再启动 Django/Vite |
| 原 PPT 是开题语气 | 全文统一改为“已实现/阶段完成/下一步验证”，避免前后矛盾 |
