# 当前状态

更新日期：2026-06-01

## 已完成

- `ConnectionManager` 统一管理 Neo4j driver 和 Milvus client。
- RAG 模块支持外部连接注入，并避免关闭不属于自己的共享连接。
- 新增 4 类 Cypher 模板场景：
  - 井巷风速合规
  - 空气成分/有害气体
  - 局部通风机与风筒
  - 风门、风墙等通风设施
- 新增 `VentilationCypherTemplateEngine`。
- 新增 `VentilationVisionExtractor`，支持 Qwen3.5-Omni 初步观察、概念检索、概念增强图片分析和多图片联合分析。
- 新增 `VentilationConceptRetriever`，支持从 Neo4j/Milvus/内置兜底概念中检索通风概念定义卡片。
- `VentilationRAGPipeline.query()` 支持 `image_path`、`image_paths` 和 `sensor_data`。
- 新增 Django 后端：
  - `POST /api/chat/`
  - `POST /api/chat/upload/`
  - `POST /api/chat/stream/`
- 新增 Vue3 前端：
  - 对话界面
  - 图片/多图片上传
  - 传感器数据手动录入与 CSV 粘贴解析
  - SSE 流式输出
  - Markdown 渲染
  - 会话列表
- 前端用户模块已完成：
  - Gemini 风格浅色侧边栏，支持展开/收缩。
  - 多会话新建、切换、重命名、删除、归档、恢复和搜索；搜索会对标题、风险等级、日期、消息正文做归一化加权匹配。
  - 会话三点菜单支持分享、归档、重命名、导出 PDF 和删除；PDF 导出会将助手 Markdown 渲染为排版后的 HTML 再打印。
  - 本地 `localStorage` 持久化会话、用户身份和偏好设置。
  - `/login` 和 `/register` 登录/注册页已接入 Django session 账号。
  - 登录用户会将当前账号作用域内的会话同步到 Django 后端 `ConversationRecord`，并同步昵称和偏好设置。
  - 账号本地缓存已按用户 ID 隔离；已有账号登录只加载该账号本地缓存和后端会话，不再自动继承游客或其他账号的本地会话；注册新账号时才迁移游客会话。
  - 登录用户图片附件已迁移到后端 `ConversationAttachment`；前端消息优先使用附件 URL/元数据，浏览器本地缓存可保留压缩预览兜底，刷新和 PDF 导出仍可显示图片。
  - P4 已新增团队空间：`Team`、`TeamMembership`、会话 `team` 归属、团队成员管理、团队统计范围。个人历史不会自动共享，只有显式分配到团队的会话进入团队统计。
  - P4+ 已新增团队会话浏览：对话菜单可显式分配团队；侧边栏“团队对话”可只读打开团队成员共享的会话。
  - P5 已新增生产账号安全基线：CSRF bootstrap、写请求 CSRF 校验、Django password validators、登录失败限流、`SecurityEvent` 审计记录和 `/settings` 账号安全记录。
  - P4+/P5 后续 UI 已完成对齐：团队归属子菜单固定显示在会话菜单右侧并避免悬停断层；账号安全记录使用五行滚动列表；团队名称支持内联编辑；设置页和统计页下拉框复用 `SettingsSelect.vue`。
  - 模型服务配置已接入运行时请求：`/settings` 可选择 DashScope、OpenAI、Ollama 或自定义 OpenAI-compatible 配置，访客配置保存在本地，登录用户配置保存在 `UserProfile.settings.modelConfig`；聊天 JSON、上传和 SSE 请求携带 `model_config`，后端对当前请求临时覆盖 text/VL client，执行完恢复默认 pipeline。
  - `/stats` 展示会话统计和 JSON 导出：游客使用本地统计，登录用户优先使用后端 `ConversationRecord` 聚合统计，并可切换个人/团队范围。
- 前端会话隔离已覆盖发送状态、SSE 回写、输入框草稿和待上传图片预览；一个会话生成中不应阻塞其他会话发起请求。
- 图片流式辨识已显示 Agent 步骤：初步观察、概念检索、概念增强分析、规程模板匹配、报告生成。
- `plan-sensor-multiimage` 已完成：前端可在主会话输入区同时添加多张现场图片和传感器数据；消息会保存 `images[]` 与 `sensorData`，PDF 导出包含多图和传感器表格；用户消息图片支持当前页弹层预览、长图边界约束、多图箭头/键盘切换；后端 `POST /api/chat/`、`/upload/`、`/stream/` 支持 `sensor_data`，multipart 支持重复 `images` 字段；RAG pipeline 会执行多图逐张观察、概念合并、联合分析、传感器增强检索和多模态融合报告生成。
- 新增真实图片识别精度验证功能：
  - `GET /api/chat/vision/scenes/` 返回可标注场景和字段 schema。
  - `POST /api/chat/vision/evaluate/` 批量评估图片识别结果。
  - 用户侧图片辨识统一走主会话窗口：上传图片后，问题/现场描述会与图片一起进入 Qwen3.5-Omni + RAG 链路。
- 已创建 Conda 环境 `ventilation-identify-system` 并安装依赖。

## 已验证

- `web_backend/manage.py check` 通过。
- `web_backend/manage.py makemigrations --check --dry-run` 通过，无遗漏 migration。
- 用户 API 烟测通过：注册、读取当前用户、同步会话、删除远端会话快照。
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
- 概念知识层脚本支持 Neo4j 已有 `Concept` 时跳过 LLM 生成并刷新 Milvus `ventilation_concepts`。
- `node node_modules/vue-tsc/bin/vue-tsc.js --build` 通过。
- `node node_modules/vite/bin/vite.js build` 通过。
- 修复账号切换串号和设置页同步按钮循环后，`vue-tsc --build` 与 `vite build` 再次通过。
- 修复 PDF 导出 Markdown 原文问题后，`vue-tsc --build` 与 `vite build` 再次通过。
- 优化侧边栏对话搜索和清空按钮样式后，`vue-tsc --build` 与 `vite build` 再次通过。
- 新增后端图片附件存储后，`web_backend/manage.py check`、`web_backend/manage.py makemigrations --check --dry-run`、附件上传/列表/删除烟测、`vue-tsc --build` 和 `vite build` 通过。
- 新增 P3 后端统计聚合后，`web_backend/manage.py check`、后端 stats summary/trends/hazards 烟测、`vue-tsc --build` 和 `vite build` 通过。
- 新增 P4 团队权限与统计后，`web_backend/manage.py check`、`makemigrations --check --dry-run`、`users.0003` 本地迁移、P4 团队 API 烟测、`vue-tsc --build` 和 `vite build` 通过。
- 新增 P5 账号安全后，`users.0004` 本地迁移、CSRF/弱密码/登录限流/安全事件/团队权限边界烟测、`web_backend/manage.py check`、`makemigrations --check --dry-run`、`vue-tsc --build` 和 `vite build` 通过。
- 新增 P4+ 团队会话浏览后，团队会话接口烟测通过：成员 B 归属团队的会话可被成员 A 读取，非成员返回 403；`web_backend/manage.py check`、`makemigrations --check --dry-run`、`vue-tsc --build` 和 `vite build` 通过。
- 完成 P4+/P5 UI 对齐后，`vue-tsc --build` 和 `vite build` 通过；最近一次验证覆盖 `SettingsSelect` 在 `/settings` 与 `/stats` 的复用，以及全局 `.page-header` 按钮样式收窄。
- 完成 `plan-sensor-multiimage` 后，Python 编译检查通过，`web_backend/manage.py check` 通过，Django chat 多模态端点 fake pipeline 烟测通过，`node node_modules/vue-tsc/bin/vue-tsc.js --build` 和 `node node_modules/vite/bin/vite.js build` 通过。
- 完成运行时模型配置后，Python 编译检查通过，`web_backend/manage.py check` 通过，Django chat `model_config` fake pipeline 烟测通过，pipeline 请求级覆盖/恢复烟测通过，`POST /api/chat/model/test/` fake OpenAI 烟测通过，`node node_modules/vue-tsc/bin/vue-tsc.js --build` 和 `node node_modules/vite/bin/vite.js build` 通过。

## 剩余风险

- 真实图片识别最终准确率仍取决于 Qwen3.5-Omni 服务、现场样图质量、概念知识覆盖度和用户提供的现场描述质量。
- 多图片联合分析和传感器交叉验证的最终质量仍取决于图片是否属于同一现场、传感器数值是否标注地点/时间、以及检索结果中是否包含对应规程阈值；生成层会在检索缺失时提示“当前检索结果未包含该参数阈值”。
- 概念知识层目前已支持脚本化构建/刷新；实际图片准确率仍需确认 Neo4j `Concept` 和 Milvus `ventilation_concepts` 的数据质量。
- Celery/Redis 目前只是离线任务预留，在线问答链路不依赖 Celery。
- 生产部署尚未配置 ASGI/WSGI 服务、反向代理、静态资源托管和正式数据库。
- 当前模型配置是请求级运行时覆盖，不是完整部署方案；生产环境仍需决定 API key 托管、审计、密钥加密/遮罩、部署环境变量和多实例锁/队列策略。公开部署必须避免普通用户请求回退到维护者 `.env` 模型密钥，优先采用 BYOK local-only 或服务端加密保存用户密钥。
- 本地 `.env` 可能覆盖示例配置；排查时以当前 `.env` 和运行日志中的 Milvus collection 为准。
- 账号模块已具备 CSRF、密码策略、登录限流和审计记录；仍使用 Django session 和 SQLite，生产部署前应迁移到正式数据库并按实际域名复核 `CSRF_TRUSTED_ORIGINS`、`SameSite`、`Secure`、CORS 和代理头。
- P4/P4+ 团队权限仍是轻量模型：支持 `owner/admin/member`、成员管理、团队统计、会话显式归属和团队会话只读浏览；尚未支持邀请链接、所有权转移、团队会话编辑审批或复杂组织层级。
- 游客图片消息仍以压缩 data URL 写入本地会话快照；登录用户远端同步使用后端附件引用，但本地账号缓存也可能保留压缩预览作为刷新兜底。生产部署前仍需把开发期本地 media 存储替换或约束为正式对象存储/静态资源策略。

## P2/P3 统计面板增强

- `/stats` 已从基础计数升级为本地统计看板。
- P2 数据来源是前端 Pinia store 中当前作用域的会话数据；P3 已增加后端账号级聚合。游客只统计游客本地会话，登录用户优先使用 `GET /api/users/stats/summary/?days=7` 返回的后端统计。
- 新增统计口径：
  - 会话数、归档数、消息总数、完成报告数。
  - 完成率：至少包含一份已完成助手报告的会话数 / 当前有效会话数。
  - 近 7 天活跃天数与每日趋势。
  - 风险等级分布与风险重点。
  - 场景分布。
- 风险等级前端使用 `Conversation.hazardLevel`，后端使用 `ConversationRecord.hazard_level`；空值归入 `未分级`，中文高/中/低风险和英文 high/medium/low 会归一到标准风险桶。
- P3 接口包括 `GET /api/users/stats/summary/`、`GET /api/users/stats/trends/`、`GET /api/users/stats/hazards/`。P4 起这些接口支持 `teamId` 参数，团队成员可查看该团队内显式分配会话的跨用户聚合统计。
