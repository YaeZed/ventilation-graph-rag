# 当前状态

更新日期：2026-05-22

## 已完成

- `ConnectionManager` 统一管理 Neo4j driver 和 Milvus client。
- RAG 模块支持外部连接注入，并避免关闭不属于自己的共享连接。
- 新增 4 类 Cypher 模板场景：
  - 井巷风速合规
  - 空气成分/有害气体
  - 局部通风机与风筒
  - 风门、风墙等通风设施
- 新增 `VentilationCypherTemplateEngine`。
- 新增 `VentilationVisionExtractor`，支持 Qwen3.5-Omni 初步观察、概念检索、概念增强图片分析。
- 新增 `VentilationConceptRetriever`，支持从 Neo4j/Milvus/内置兜底概念中检索通风概念定义卡片。
- `VentilationRAGPipeline.query()` 支持 `image_path`。
- 新增 Django 后端：
  - `POST /api/chat/`
  - `POST /api/chat/upload/`
  - `POST /api/chat/stream/`
- 新增 Vue3 前端：
  - 对话界面
  - 图片上传
  - SSE 流式输出
  - Markdown 渲染
  - 会话列表
- 图片流式辨识已显示 Agent 步骤：初步观察、概念检索、概念增强分析、规程模板匹配、报告生成。
- 新增真实图片识别精度验证功能：
  - `GET /api/chat/vision/scenes/` 返回可标注场景和字段 schema。
  - `POST /api/chat/vision/evaluate/` 批量评估图片识别结果。
  - 用户侧图片辨识统一走主会话窗口：上传图片后，问题/现场描述会与图片一起进入 Qwen3.5-Omni + RAG 链路。
- 已创建 Conda 环境 `ventilation-identify-system` 并安装依赖。

## 已验证

- `web_backend/manage.py check` 通过。
- `agent/rag_system/test_ventilation_cypher_templates.py` 通过。
- `agent/rag_system/test_ventilation_vision_extractor.py` 通过。
- Docker 中 Neo4j 和 Milvus 可连。
- Neo4j 当前可查询到图谱节点。
- Milvus 当前 collection 包含 `ventilation_safety`。
- CLI 示例问题“掘进中的岩巷最低风速要求是多少”返回 `>=0.15 m/s`。
- `POST /api/chat/` 返回 HTTP 200 和 Markdown 答案。
- `POST /api/chat/stream/` 返回 `status`、`token`、`done/error` SSE 事件。
- 图片 `POST /api/chat/stream/` 额外返回 `step` SSE 事件，前端可渲染处理进度。
- Vite 代理 `/api/chat/stream/` 到 Django 后端可正常流式返回 token。
- 视觉评估指标逻辑 `web_backend/chat/test_vision_evaluation.py` 通过 fake pipeline 烟测。
- `npm run build` 通过。

## 剩余风险

- 真实图片识别最终准确率仍取决于 Qwen3.5-Omni 服务、现场样图质量、概念知识覆盖度和用户提供的现场描述质量。
- 概念知识层目前可使用内置兜底概念；若要扩大覆盖面，需要执行 `agent/data_pipeline/build_concept_knowledge.py` 并确认 Neo4j `Concept` 与 Milvus `ventilation_concepts` 数据已入库。
- Celery/Redis 目前只是离线任务预留，在线问答链路不依赖 Celery。
- 生产部署尚未配置 ASGI/WSGI 服务、反向代理、静态资源托管和鉴权。
- 本地 `.env` 可能覆盖示例配置；排查时以当前 `.env` 和运行日志中的 Milvus collection 为准。
