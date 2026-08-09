# Findings: Project Final Report

## Source Documents

- `README.md`: current product capability, quick start, runtime services.
- `docs/architecture.md`: top-level data flow, RAG core modules, image/multi-image/sensor chain, web/user module.
- `docs/status.md`: completed work, verified commands, remaining risks.
- `docs/api.md`: REST/SSE/user/team/stats/model-test/vision-evaluation API contract.
- `docs/runbook.md`: environment, startup, validation, common troubleshooting.
- `docs/plan-user-module.md`: user module phases P1-P5/P4+ and acceptance records.
- `docs/plan-sensor-multiimage.md`: sensor and multi-image implementation scope.
- `docs/plan-model-config-deploy.md` and `docs/deployment-decision-matrix.md`: runtime model config and deployment boundary.

## Implemented Scope

- RAG core: Neo4j graph, Milvus vector index, GraphRAG, hybrid retrieval, Cypher templates, query routing, Markdown answer generation.
- Vision/multimodal: Qwen3.5-Omni initial observation, concept retrieval, concept-enhanced analysis, multi-image joint analysis, sensor-data fusion.
- Backend: Django REST/SSE chat APIs, image upload, stream events, model connection test, vision evaluation, user/team/stats/attachment APIs.
- Frontend: Vue3 + TypeScript + Pinia, chat workspace, multi-image queue, sensor input, Markdown renderer, Agent timeline, local/account persistence, stats/settings/team UI.
- User/team/security: session auth, CSRF, password validation, login throttling, security events, team roles, team conversation browsing.
- Runtime model config: DashScope/OpenAI/Ollama/custom OpenAI-compatible endpoint request-level override.
- Verified scope: Django checks, migrations, fake-pipeline smoke tests, Cypher/VL tests, frontend type-check/build according to `docs/status.md`.
- Project scale from local count: `agent` 32 files / 7260 lines, `web_backend` 26 files / 2351 lines, `frontend/src` 32 files / 8667 lines, `docs` 14 files / 1665 lines.

## Screenshot Placeholder Plan

- 主对话页整体截图：展示产品名称、侧边栏、输入区和消息流。
- 法规问答结果截图：展示风速/有害气体等 Markdown 表格报告。
- 图片或多图上传截图：展示图片缩略图和 Agent 步骤。
- 传感器数据输入/数图融合报告截图：展示传感器条目与融合结论。
- 统计看板截图：展示完成率、风险分布、趋势图。
- 设置页模型配置截图：展示模型服务商、text/vision endpoint 和测试连接。
- 团队空间截图：展示团队管理或团队对话只读浏览。
- 账号安全记录截图：展示 CSRF/安全事件/登录记录面板。
- 后端/数据服务运行截图：可选，展示 Django check、Docker services、Neo4j/Milvus 状态。

## Report Style

- Chinese.
- Student project final report.
- Product and engineering summary, not academic thesis.
