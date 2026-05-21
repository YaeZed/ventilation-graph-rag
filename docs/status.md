# 当前状态

更新日期：2026-05-21

## 已完成

- `ConnectionManager` 统一管理 Neo4j driver 和 Milvus client。
- RAG 模块支持外部连接注入，并避免关闭不属于自己的共享连接。
- 新增 4 类 Cypher 模板场景：
  - 井巷风速合规
  - 空气成分/有害气体
  - 局部通风机与风筒
  - 风门、风墙等通风设施
- 新增 `VentilationCypherTemplateEngine`。
- 新增 `VentilationVisionExtractor`，支持 Qwen2.5-VL 两阶段图片理解。
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
- Vite 代理 `/api/chat/stream/` 到 Django 后端可正常流式返回 token。
- `npm run build` 通过。

## 剩余风险

- 真实图片识别精度仍需用 Qwen2.5-VL 服务和矿井现场样图验证。
- Celery/Redis 目前只是离线任务预留，在线问答链路不依赖 Celery。
- 生产部署尚未配置 ASGI/WSGI 服务、反向代理、静态资源托管和鉴权。
- `.env.example` 中的 Milvus collection 名称可能与现有运行数据不同；实际 RAG 配置以代码和 `.env` 为准。

