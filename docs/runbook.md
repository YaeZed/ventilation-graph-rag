# 运行手册

## 环境

推荐 Python 环境：

```powershell
conda create -y -n ventilation-identify-system python=3.10
conda activate ventilation-identify-system
D:\Miniconda\envs\ventilation-identify-system\python.exe -m pip install -r requirements.txt
```

前端需要 Node `^20.19.0 || >=22.12.0` 和 npm。

```powershell
cd frontend
npm install
```

## 环境变量

复制 `.env.example` 为 `.env`，至少配置：

```ini
DASHSCOPE_API_KEY=sk-...
LLM_MODEL=qwen-plus
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=160722yaesakura
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

可选：

```ini
QWEN_VL_MODEL=qwen3.5-omni-plus
DJANGO_DEBUG=1
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
DJANGO_CSRF_TRUSTED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
DJANGO_SESSION_COOKIE_AGE=604800
ACCOUNT_LOGIN_FAILURE_LIMIT=5
ACCOUNT_LOGIN_LOCKOUT_SECONDS=300
VENTILATION_PIPELINE_FORCE_REBUILD=0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

## 启动数据服务

```powershell
docker compose up -d
docker compose ps
```

预期：

- Neo4j 监听 `7474` 和 `7687`
- Milvus 监听 `19530`
- etcd 和 MinIO 正常运行

## 验证后端

```powershell
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py check
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py migrate
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\test_ventilation_cypher_templates.py
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\test_ventilation_vision_extractor.py
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\chat\test_vision_evaluation.py
```

真实 RAG CLI：

```powershell
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\ventilation_rag_pipeline.py -q "掘进中的岩巷最低风速要求是多少" --top-k 3
```

预期答案应包含 `0.15 m/s`。

## 启动 Web

终端 1：

```powershell
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py runserver 127.0.0.1:8000 --noreload
```

终端 2：

```powershell
cd frontend
npm run dev
```

打开 `http://127.0.0.1:5173`。

本地开发时建议让 Django 后端和 Vite 前端分别占用一个前台终端。Codex 桌面环境里，Django 后端进程如果被后台启动可能会被会话清理；需要稳定调试接口时，直接在 PowerShell 中运行上面的 `runserver` 命令并保持窗口打开。

## 构建概念知识层

图片链路会优先检索 Neo4j `Concept` 和 Milvus `ventilation_concepts` 中的概念卡片。构建命令：

```powershell
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\data_pipeline\build_concept_knowledge.py
```

脚本行为：

- 如果 Neo4j 没有 `Concept` 节点，会从图谱条文/参数/设施提取候选概念，调用 Qwen 生成定义、视觉线索、识别特征，并写入 Neo4j 与 Milvus。
- 如果 Neo4j 已有 `Concept` 节点，会跳过 LLM 生成，直接从 Neo4j 读取概念并刷新 Milvus 的 `ventilation_concepts` 集合。
- `--force` 会删除旧 `Concept` 节点并重新生成，会重新消耗 DashScope 额度。

Neo4j 验证：

```cypher
MATCH (c:Concept) RETURN count(c) AS total;
MATCH (c:Concept) RETURN c.name, c.definition, c.visual_clues LIMIT 10;
```

## 前端构建

```powershell
cd frontend
npm run build
```

该命令同时执行 `vue-tsc --build` 和 `vite build`。

## 前端用户模块冒烟

打开 `http://127.0.0.1:5173/chat` 后检查：

1. 左侧侧边栏可展开/收缩；收缩态只显示图标，展开态显示新建、搜索、统计入口、未归档对话列表、归档、设置和用户头像。
2. 新建 2 个对话，分别输入不同问题；切换 URL `/chat/:conversationId` 时消息、发送状态和输入草稿互不串线。
3. 刷新浏览器后，会话、昵称、偏好设置仍存在；若上传过图片，预览和导出内容应优先保留。
4. 搜索会按标题、场景、风险等级、日期和消息内容过滤未归档会话。
5. 会话三点菜单可重命名、归档、导出 PDF、删除；归档区在列表底部，可展开/收起，点击归档项会恢复该会话。
6. `/stats` 显示会话统计并可导出 JSON；游客模式显示“本地统计”，登录后应显示“个人后端统计”，选择团队后应显示“团队统计”。后端加载失败时回退本地统计。重点检查完成率圆环、风险等级分布、近 7 天趋势和场景分布。`/settings` 可修改昵称、默认流式响应、Agent 步骤展开偏好和 temperature。
7. 访问 `/register` 创建账号后回到 `/chat`；游客本地会话应迁移到新账号并同步到后端。访问 `/login` 登录已有账号时，只应恢复该账号自己的会话、昵称和偏好设置，不应继承游客或其他账号的会话。
8. `/settings` 在登录后显示账号同步状态，可手动“立即同步”或退出登录；退出后回到本地模式。
9. 登录状态下上传图片后，消息应立即显示图片；点击缩略图应在当前页面中央放大预览而不是打开新页面；刷新页面后仍能显示。浏览器 `localStorage` 可以保存压缩后的消息 `images[]` 预览兜底，但不应保存原始大图；同步到后端的会话快照应只保留附件 URL/元数据。
10. 登录后在 `/settings` 创建团队并按用户名添加成员；回到侧边栏对话三点菜单，通过“归属团队”子菜单把当前会话切到团队。子菜单应固定显示在主菜单右侧，移动鼠标到子菜单时不应消失。成员账号刷新后，“最近对话”下方的“团队对话”应能看到并只读打开这条会话，副标题应显示团队名和放入者身份名。再到 `/stats` 选择该团队，团队统计应包含显式分配到团队的会话。非团队成员访问带 `teamId` 的统计接口和团队会话接口应返回 403。
11. `/settings` 登录后应显示“账号安全”记录；列表高度约为 5 条记录并可滚动。连续输错同一账号密码 5 次后，登录接口应返回 429 并提示稍后再试。
12. `/settings` 的团队选择、成员角色选择和 `/stats` 的统计范围选择应使用同一套浅色下拉样式；鼠标悬停和选中态不应出现系统蓝色或白字空白。
13. 在 `/chat` 输入区点击 `+` 连续选择 2-3 张图片，缩略图队列应显示在输入框上方；点击“数”录入风速/瓦斯等传感器数据或粘贴 CSV，消息发送后应显示多图网格和传感器数据条。SSE 步骤应包含多图观察/联合分析或传感器核对；报告应包含图片证据、传感器数据和规程依据的交叉验证。登录状态下刷新后，多图应优先使用附件 URL/元数据恢复，也可以使用本地压缩预览兜底；游客模式保留压缩 data URL。

## 常见问题

### 页面一直显示“正在生成...”

先确认后端流是否返回 token：

```powershell
curl -N -X POST http://127.0.0.1:8000/api/chat/stream/ -H "Content-Type: application/json" -d "{\"question\":\"矿井有害气体最高允许浓度范围是什么\"}"
```

如果后端有 token，但页面不更新，检查 `frontend/src/stores/chat.ts` 是否通过 message id 更新 Pinia 中的响应式消息对象。

### PowerShell 中文乱码

Windows PowerShell 可能把中文或 CO₂ 这类字符显示成乱码。优先用 UTF-8 输出或 Python `unicode_escape` 验证真实响应，不要直接判断后端返回损坏。

### Node 被拒绝访问

Codex 沙箱中可能无法直接执行 WindowsApps 下的 `node.exe`。用系统 Node/npm 或在获批的外部 shell 中运行 `npm install`、`npm run build`、`npm run dev`。

### 第一次请求较慢

Django pipeline 是懒加载。第一次请求会初始化 RAG、连接 Neo4j/Milvus、加载嵌入模型和索引，后续请求复用同一实例。

### 多个会话互相影响

前端应按 `conversationId` 隔离发送状态、SSE 回写、输入框草稿和待上传图片预览。若出现 A 会话生成中导致 B 会话不能发送，或切换会话后回复写错窗口，优先检查 `frontend/src/stores/chat.ts` 的 `sendingByConversation` 和回调里的 `conversationId`，以及 `frontend/src/views/HomeView.vue` 的 `drafts`。

### 刷新后历史对话或图片丢失

未登录时，会话、设置和简易用户身份保存在浏览器 `localStorage`，key 为 `ventilation-graph-rag:user-module:v2:guest`。登录用户按账号 ID 隔离保存到 `ventilation-graph-rag:user-module:v2:user:<userId>`，并同步到 Django `users.ConversationRecord`。已有账号登录不会自动继承游客或其他账号的会话；注册新账号时才迁移游客会话。游客上传图片会压缩为 data URL 保存；登录用户上传图片会写入后端 `ConversationAttachment`。浏览器本地缓存可以保留压缩后的消息 `images[]` 预览，用于刷新后兜底恢复；远端同步会剥离 data URL，只保存附件 URL/元数据。若原图附件上传失败，前端会用压缩预览重试上传，失败后仍保留本地预览，保证本次辨识能继续。

如果刷新后图片仍丢失，按顺序检查：附件上传接口是否返回 200；图片是否超过 8MB 且压缩预览重试是否成功；Django `/media/` 是否可访问；`ConversationAttachment.messageClientId` 是否和用户消息 id 匹配；当前账号的 localStorage key 是否为 `ventilation-graph-rag:user-module:v2:user:<userId>`。

### 登录后同步失败

先确认数据库迁移已经执行：

```powershell
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py migrate
```

再检查 `/api/users/me/` 是否返回当前用户。开发期账号使用 Django session，同源 Vite 代理会携带 cookie；写请求必须先访问 `/api/users/auth/csrf/` 并携带 `X-CSRFToken`。若改成跨域部署，需要把前端源加入 `DJANGO_CSRF_TRUSTED_ORIGINS`，并重新评估 `SameSite`、`Secure` 和 CORS 策略。

### 登录后统计不更新

登录用户的 `/stats` 会调用 `GET /api/users/stats/summary/?days=7`；团队范围会额外携带 `teamId`。先确认 `/settings` 手动同步成功，再用浏览器网络面板或 Django 日志检查该接口是否返回 `ok: true`。统计聚合来自后端 `ConversationRecord`，如果前端刚生成报告但还未同步到后端，统计页会短暂显示旧数据或回退本地统计。团队统计只包含显式分配到该团队的会话，不会自动包含个人空间历史记录。

### 团队接口返回 403

先检查当前登录账号是否在 `GET /api/users/teams/` 返回列表内。`teamId` 统计、成员列表、会话分配都要求当前用户已加入团队；添加/调整成员需要 `owner` 或 `admin`；删除团队只能由 `owner` 执行。

### 团队成员看不到团队对话

先确认创建者已经通过对话三点菜单的“归属团队”把会话显式分配到团队。团队成员登录后会在刷新团队列表时加载 `GET /api/users/teams/<teamId>/conversations/`；这些记录显示在侧边栏“团队对话”下拉中。别人的团队会话是只读浏览，不会出现在“最近对话”的个人列表里。

### 归档对话找不到

归档对话不会显示在未归档对话列表中。展开侧边栏底部的“已归档”区域，点击归档项即可恢复到普通会话列表。

### 概念构建写入 Milvus 报错

当前 `pymilvus` 版本需要用 `client.prepare_index_params()` 创建 index 参数，不能传普通 dict。若从 Neo4j 读出的概念字段是列表或字典，脚本会先归一化为字符串再向量化；不要把 `_as_text()` / `_as_list()` 去掉。

### 图片请求很久没有完整报告

图片链路会先调用 Qwen3.5-Omni 做观察，再检索概念、复核图片、检索规程并生成报告。前端应显示 `vision_observe`、`concept_search`、`vision_analyze`、`cypher_match`、`generating` 等步骤；如果只停在某一步，优先检查 DashScope 额度、模型名 `QWEN_VL_MODEL=qwen3.5-omni-plus`、后端日志和 SSE `error` 事件。

### DashScope 免费额度或请求中断

如果返回 `AllocationQuota.FreeTierOnly`，说明当前 DashScope 账号免费额度耗尽，需要在管理控制台关闭 free-tier-only 限制或更换可用 key。浏览器出现 `BodyStreamBuffer was aborted` 多半是长时间图片请求被前端或代理中断；当前前端图片请求和 SSE 已使用更长超时，仍可通过后端日志确认真实异常。
