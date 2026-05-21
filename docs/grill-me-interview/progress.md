# Progress Log

## Session 2026-05-21

### Completed
- **[grill-me 访谈]** 完成 12 个决策点的深入讨论：
  1. 系统边界：单 repo
  2. VL 输出格式：结构化 + description
  3. Schema 设计：按场景分组
  4. 检索顺序：串行，图谱先向量后
  5. 图片入口路由：不需要
  6. VL 集成位置：Pipeline 前置步骤
  7. 场景分类：两阶段
  8. Cypher 生成：模板优先 + LLM 兜底
  9. 连接管理：单例 ConnectionManager
  10. 异步任务：SSE + Celery 仅离线
  11. 前端交互：对话式
  12. 实现优先级：Phase 1-5 顺序确定

- **[规划文件]** 创建 task_plan.md、findings.md、progress.md

- **[Phase 1: ConnectionManager 重构]** 已完成代码落地：
  - 新增 `agent/connection_manager.py`，以单例方式统一管理 Neo4j driver 和 Milvus client。
  - `VentilationDataPreparationModule`、`VentilationHybridRetrieval`、`VentilationGraphRAGRetrieval` 支持外部 Neo4j driver 注入，并保留独立自建连接能力。
  - `VentilationMilvusIndexConstruction` 支持外部 Milvus client 注入，并保留独立自建 client 能力。
  - `VentilationRAGPipeline` 改为从 `ConnectionManager` 获取 Neo4j/Milvus 连接并注入到各模块。
  - 验证：`python -m compileall` 通过；stub 验证确认共享连接不会被模块 `close()` 误关闭，且配置变化会释放旧连接。

- **[Phase 2: Cypher 模板系统]** 已完成代码落地：
  - 新增 `agent/rag_system/cypher_templates/scenes.json`，定义 4 个 VL 场景 schema：井巷风速、空气成分/有害气体、局部通风机与风筒、风门风墙等通风设施。
  - 新增 4 个参数化 Cypher 模板：`airflow_speed.cypher`、`air_quality.cypher`、`local_ventilation.cypher`、`ventilation_facility.cypher`。
  - 新增 `agent/rag_system/ventilation_cypher_templates.py`，负责模板加载、场景匹配、参数绑定、执行结果转 Document。
  - 验证：`test_ventilation_cypher_templates.py` 通过，确认模板匹配和 mock 执行结果包装正常。

- **[Phase 3: Qwen2.5-VL 集成]** 已完成代码落地：
  - 新增 `agent/rag_system/ventilation_vision_extractor.py`，实现 Stage 1 场景分类与 Stage 2 schema 驱动字段抽取。
  - `VentilationRAGPipeline.query()` 新增 `image_path` 参数；图片入口固定执行 `VL 抽取 → Cypher 模板检索 → 向量兜底 → 生成答案`。
  - CLI 新增 `--image` 参数。
  - 验证：`test_ventilation_vision_extractor.py` 使用 fake Qwen-VL client 通过，确认两阶段抽取和字段清洗正常。

- **[Phase 4: Django API + SSE]** 已完成代码落地：
  - 新增 `web_backend/` Django 项目结构。
  - 新增 API：`POST /api/chat/`（文字 JSON）、`POST /api/chat/upload/`（图片 multipart）、`POST /api/chat/stream/`（SSE 流式）。
  - 新增 `chat.pipeline_service`，Django 进程内懒加载单例 `VentilationRAGPipeline`。
  - 新增 Celery app 与 `chat.health_check` 占位任务，保持“仅离线任务预留”的架构决策。
  - `requirements.txt` 补充 `Django`、`celery`、`redis`。
  - 验证：`python -m compileall web_backend ...` 通过。

- **[Phase 5: Vue3 对话前端]** 已完成代码落地：
  - 新增 `frontend/`，参照 `D:\projects\personal_projects\mine-LLM-fronted` 的 Vue3 + Router + Pinia + MainLayout 结构。
  - 实现可折叠侧边栏、历史会话、新建会话、底部胶囊输入框、图片上传预览、SSE/普通响应切换。
  - `src/api/chat.ts` 对接 Django 端点：`/api/chat/`、`/api/chat/upload/`、`/api/chat/stream/`。
  - 视觉按 `frontend-design` 思路调整为克制工业调度台风格，服务于煤矿通风安全检查场景。

### Verification
- **[Python 环境]** 已创建 Conda 环境 `ventilation-identify-system`（Python 3.10.20），并通过 `D:\Miniconda\envs\ventilation-identify-system\python.exe -m pip install -r requirements.txt` 安装项目依赖。
- **[Django 运行验证]** `D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py check` 已通过：`System check identified no issues`。
- **[新增模块测试]** `test_ventilation_cypher_templates.py` 和 `test_ventilation_vision_extractor.py` 已在新环境通过。
- **[Docker 数据服务]** Neo4j、Milvus、etcd、MinIO 已通过 docker compose 启动；Neo4j 7687 与 Milvus 19530 端口可连，Neo4j 节点数 542，Milvus collection 为 `ventilation_safety`。
- **[真实 CLI 问答验证]** 已通过真实 Neo4j/Milvus 与本地 RAG 流程验证；示例问题“掘进中的岩巷最低风速要求是多少”返回 `>=0.15 m/s`，依据《煤矿安全规程》第一百五十七条表6。
- **[Django API 联调]** `POST /api/chat/` 返回 HTTP 200 且 `ok=True`，回答包含 `0.15`；`POST /api/chat/stream/` 返回 HTTP 200，SSE 事件包含 `status` 与连续 `token`。PowerShell 直接打印中文会出现显示层乱码，`unicode_escape` 验证真实响应字符串正常。
- **[前端构建验证]** 通过提升权限使用系统 Node `v20.19.5` 与 npm `10.8.2`；`npm install` 成功，`npm run build` 已通过 `vue-tsc --build` 和 `vite build`。修复了 `activeConversation` 在严格类型检查下可能为空的 store 类型问题。
- **[前端 SSE 卡住修复]** 用户反馈页面发送问题后长期停留在“正在生成...”。经直连 `127.0.0.1:8000/api/chat/stream/` 与 Vite 代理 `127.0.0.1:5173/api/chat/stream/` 验证，后端和代理均能正常返回 `status/token`；根因是前端 SSE token 更新落在非响应式消息对象上，页面没有刷新。已改为按 message id 回写 Pinia store 中的响应式消息对象，并加入 status 文案、120 秒请求超时和流结束兜底收尾；`npm run build` 通过，Vite dev server 已重启。
- **[前端 Markdown 渲染]** 模型输出为 Markdown，已新增 `MarkdownRenderer.vue`，使用 `markdown-it` 渲染助手消息并关闭原始 HTML；用户消息仍按纯文本展示。补充标题、列表、引用、表格、行内代码、代码块和链接样式；安装 `markdown-it` 与 `@types/markdown-it`，`npm run build` 通过，Vite dev server 已重启。
- **[neat-freak 文档收尾]** 已将 `AGENTS.md` 从过期计划改为当前 agent 速查规则；重写 `README.md` 为当前 B/S 系统快速启动；新增 `docs/README.md`、`docs/architecture.md`、`docs/api.md`、`docs/runbook.md`、`docs/status.md`。自检确认主文档无 “Django/Vue 待引入”“Node 阻塞”“未安装 Django” 等过期说法，关键命令通过 `manage.py check`、新增模块测试和 `npm run build` 验证。
- **[真实图片精度验证]** Python 依赖已恢复，代码路径已用 fake client 验证；仍需 Qwen-VL 服务可用和样图后执行精度测试。

### Next Steps
- 启动 Django 与 Vite 开发服务器，做浏览器端对话、SSE 渲染和图片上传体验验证。
- 接入真实 Qwen2.5-VL 服务和样图，验证图片隐患识别精度。
