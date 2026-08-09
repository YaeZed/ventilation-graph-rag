# Findings: Meeting PPT Report

## Skill Availability

- 用户提到可使用 `ppt-master`，但当前会话可用 skill 列表中没有该 skill。
- 用户补充 GitHub 地址：`https://github.com/hugohe3/ppt-master`。
- 已通过 `skill-installer` 安装到 `D:/codex-home/skills/ppt-master`。安装命令超时，但目录和 `SKILL.md`、`workflows/`、`scripts/`、`templates/` 均已落盘，`git ls-remote` 也确认仓库可访问。
- 当前会话不会动态刷新可用 skill 列表；本轮直接读取本地 `SKILL.md` 和 `workflows/template-fill-pptx.md` 执行。后续重启 Codex 后应能正式识别该 skill。
- `template-fill-pptx` 是最匹配本任务的 workflow：直接把原 PPTX 当作模板库，抽取 slide library，写 fill plan，再直接替换 OOXML 文本生成可编辑 PPTX。它明确不使用 `pptx_to_svg.py`、`pptx_template_import.py`、`finalize_svg.py` 或 `svg_to_pptx.py`。

## Original PPT Structure

- 原文件：`C:/Users/Administrator/Desktop/研究生期间材料/论文/毕业论文/开题/张明俊-开题答辩PPT.pptx`
- 文件大小约 1.5 MB，最后修改时间为 2026-03-01 16:08。
- 共 17 页。
- 原叙事是开题答辩：
  - 封面：数据与法规融合的煤矿通风系统安全隐患智能辨识方法研究。
  - 提纲：研究背景、理论意义、研究内容、实验设想。
  - 背景/现状：煤矿通风安全、监测预警、法规可追溯不足。
  - 理论意义/创新点：LLM、知识图谱、RAG、多模态融合。
  - 研究内容：系统开发、法规知识库、Qwen2.5-VL + RAG。
  - 技术路线：Python 处理法规、FAISS/Milvus、Qwen2.5-VL、Django + Vue3。
  - 实验设想：实验平台、传感器、图像和法规推理流程。
  - 困难与解决措施：多模态数据质量、RAG 精度、语义对齐。

## Existing Visual Assets

- `开题/images/` 下已有实验平台、风机、风速传感器、技术路线图、架构图、研究方法图等素材。
- 原 PPT 内部含 4 个 slide master、26 个 slide layout、13 个 media 资源，可复用原视觉体系。
- `开题/流程图/` 下有 `.eddx` 源文件：架构图、研究方法图。若后续需要重画架构图，可参考这些源图的主题和布局。

## Current System Facts To Reflect

- 当前系统已经不是单纯设想，已实现 Python RAG 核心、Django API/SSE 后端、Vue3 + TypeScript + Pinia 前端。
- 知识层包含 Neo4j 图谱、Milvus 向量索引、Concept 概念层和确定性 Cypher 模板。
- 图像链路已升级为 Qwen3.5-Omni，两阶段处理：初步观察、概念检索增强分析。
- 支持单图、多图联合分析，并保留每张图观察结果与跨图结论。
- 支持结构化传感器数据输入，并在生成层交叉验证图片证据、传感器数值和法规上下文。
- SSE 流式接口输出 `status/token/done/error`，图片、多图、传感器流程还输出 `step` 事件，用于前端 Agent 时间线。
- 用户模块已包含会话隔离、后端图片附件、PDF 导出、团队空间、团队统计、账户安全、模型服务配置。
- `/settings` 支持 DashScope/OpenAI/Ollama/自定义 OpenAI-compatible 模型配置，请求级覆盖 text/VL client。

## Content Corrections Needed

- 原 PPT 中 `Qwen2.5-VL` 应更新为当前实现中的 `Qwen3.5-Omni`。
- 原 PPT 中 `FAISS/Milvus` 应改为当前主实现：`Milvus` 向量库 + `Neo4j` 图谱 + `Cypher` 模板。
- 原 PPT 中“拟开发”“实验设想”页面应改成“已实现原型”“阶段验证”“下一步实验验证”。
- 原 PPT 的长段文字需要压缩；会议汇报应更多使用流程图、模块卡片、截图和短结论。

## `ppt-master` Workflow Notes

- 项目目录必须包含 `sources/`、`analysis/`、`exports/`、`validation/`。
- 先运行 `template_fill_pptx.py analyze` 生成 `slide_library.json`。
- 必须基于 slide library 做页面选择依据，不能按原顺序机械替换。
- `fill_plan.json` 是执行契约，source slide 可以重复、重排或省略。
- 运行 `check-plan` 后再 `apply`，容量风险优先通过压缩文案或换布局解决，不默认缩小字号。
- `apply` 会克隆选中原页面，保留原设计和可编辑文本框，并将 `notes` 写入 PowerPoint 演讲备注。
- 生成后用 `ppt_to_md.py` 回读验证。
- 当前 template-fill v1 不支持图片替换；若需要插入当前系统截图，需要在文本版生成后作为单独 PPTX 操作处理。
