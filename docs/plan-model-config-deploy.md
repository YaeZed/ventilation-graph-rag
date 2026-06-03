# Plan: 自定义模型配置 + 系统上线

## Context

系统当前硬编码 DashScope API 和模型名（qwen-plus / qwen3.5-omni-plus）。用户希望能切换模型厂商（如 OpenAI、本地部署的 Ollama/vLLM），自定义 API endpoint 和 key。同时考虑将系统从本地开发环境部署到线上。

## Part 1: 自定义模型配置

### 功能设计

在 `/settings` 页面新增"模型配置"卡片：

```
┌─ 模型配置 ─────────────────────────────────────┐
│                                                │
│  预设方案    [DashScope (通义千问)    ▼]        │
│              OpenAI (GPT-4o)                   │
│              Ollama (本地部署)                  │
│              自定义                            │
│                                                │
│  ── 视觉模型 ──                                │
│  模型名称    [qwen3.5-omni-plus               ]│
│  API Endpoint [https://dashscope.aliyuncs.... ]│
│  API Key     [sk-••••••••              ] [显示]│
│                                               │
│  ── 文本生成模型 ──                            │
│  模型名称    [qwen-plus                       ]│
│  API Endpoint [https://dashscope.aliyuncs.... ]│
│  API Key     [同视觉模型 ▼]                    │
│                                               │
│  ── 嵌入模型（高级，默认不改）──                 │
│  本地路径    [models/bge-small-zh-v1.5        ]│
│                                              │
│  [测试连接]  [恢复默认]  [保存]                │
└──────────────────────────────────────────────┘
```

### 预设方案

```typescript
const MODEL_PRESETS = {
  dashscope: {
    name: 'DashScope (通义千问)',
    vlModel: 'qwen3.5-omni-plus',
    vlEndpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    textModel: 'qwen-plus',
    textEndpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  },
  openai: {
    name: 'OpenAI',
    vlModel: 'gpt-4o',
    vlEndpoint: 'https://api.openai.com/v1',
    textModel: 'gpt-4o',
    textEndpoint: 'https://api.openai.com/v1',
  },
  ollama: {
    name: 'Ollama (本地)',
    vlModel: 'llava:latest',
    vlEndpoint: 'http://localhost:11434/v1',
    textModel: 'qwen2.5:latest',
    textEndpoint: 'http://localhost:11434/v1',
  },
  custom: { name: '自定义', vlModel: '', vlEndpoint: '', textModel: '', textEndpoint: '' }
}
```

### 数据存储

- **localStorage**（guest）：模型配置跟随现有用户模块作用域 key，写入 `settings.modelConfig`
- **Django UserProfile**（logged-in）：模型配置作为 `UserProfile.settings.modelConfig` 同步，不单独建 `model_config` 字段
- sensor_data 和 image 走现有 storage key 体系

### 生效机制

Pipeline 初始化仍从 `.env`/ConnectionManager/config 读取开发默认值。前端每次聊天请求携带 `model_config`，Django 侧在当前请求范围内临时覆盖 text/VL OpenAI-compatible client 和模型名，执行完成后恢复默认对象。

数据流：
```
用户保存配置 → store.setModelConfig()
                  ├─ localStorage scoped settings.modelConfig（立即生效，下次刷新可用）
                  └─ API: PATCH /api/users/profile/ {settings: {modelConfig: {...}}}
                          ↓
                  chat/upload/stream 请求携带 model_config
                          ↓
                  VentilationRAGPipeline 请求级覆盖 text/VL client
```

### 密钥策略（部署前必须解决）

当前开发态允许在用户未填写 key 时回退到 `.env` 中的 `DASHSCOPE_API_KEY`，方便本地调试默认 DashScope 模型。这个行为不能直接带到公开部署环境。

部署版本必须采用 BYOK（Bring Your Own Key）或明确的后端租户密钥策略：

- 普通用户请求必须使用该用户在 `/settings` 提交的 text/vision API key；未配置时前端应提示“请先配置模型密钥”，后端也应拒绝执行模型调用。
- 项目维护者的 `.env` 密钥只能用于本地开发、管理员自测或受控的内部演示，不能作为公开用户请求的默认兜底。
- 登录用户密钥如果同步到后端，必须加密存储并在 UI/API 响应中遮罩；更保守的方案是只在浏览器本地保存密钥，请求时随 `model_config` 发送，不写入后端持久化。
- 部署前需要选择一种策略并实现：`BYOK local-only`、`BYOK encrypted server-side` 或 `admin-managed tenant key`。默认推荐 `BYOK local-only`，因为它最直接避免用户消耗维护者额度。

### 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `frontend/src/stores/chat.ts` | 新增 | modelConfig state + preset/validate/save actions |
| `frontend/src/views/SettingsView.vue` | 改造 | 新增模型配置卡片 |
| `frontend/src/api/chat.ts` | 改造 | chat/upload/stream/model-test 请求携带 model_config |
| `web_backend/users/views.py` | 改造 | 归一化并保存 `settings.modelConfig` |
| `web_backend/chat/views.py` | 改造 | 解析 `model_config`，提供连接测试接口 |
| `agent/rag_system/ventilation_rag_pipeline.py` | 微调 | 支持请求级 model/client 注入和恢复 |

---

## Part 2: 系统上线

### 现状分析

| 组件 | 当前 | 线上方案 |
|------|------|---------|
| Neo4j | Docker 本地 | Neo4j AuraDB Free（云端图数据库，免费 1GB） 或 云服务器自建 |
| Milvus | Docker 本地 | Zilliz Cloud Free（免费 1GB） 或 轻量替代 FAISS |
| Django | runserver | Gunicorn + 云服务器 / Railway / Render |
| 静态前端 | Vite dev | `npm run build` → nginx 静态文件 或 同一 Django 进程 serve |
| Embedding 模型 | 本地 BGE | 保留本地加载，需确保服务器有足够内存 |

### 推荐方案：单服务器 + 云数据库

```
┌─ 阿里云 / 腾讯云 ECS (2C4G) ──────────────────┐
│                                                │
│  nginx :80/:443                                 │
│    ├── /api/*  → Gunicorn (Django) :8000        │
│    ├── /media/* → 静态文件目录                   │
│    └── /*     → Vue dist 静态文件                │
│                                                │
│  外部依赖:                                      │
│    Neo4j AuraDB (云端)                          │
│    Zilliz Cloud 或 本地 FAISS (嵌入向量)         │
│    DashScope/OpenAI API (自定义配置)             │
└────────────────────────────────────────────────┘
```

### 上线步骤

**Step 1: 生产配置**

- Django `DEBUG=False`，`SECRET_KEY` 从环境变量读取
- `ALLOWED_HOSTS` 配置域名
- nginx 配置 HTTPS（Let's Encrypt）
- 静态文件收集：`collectstatic` → nginx alias

**Step 1.5: 用户密钥与额度边界**

- 禁止公开环境把普通用户请求自动回退到维护者的 `DASHSCOPE_API_KEY`。
- 未配置用户 key 时，`/api/chat/`、`/api/chat/upload/`、`/api/chat/stream/` 和 `/api/chat/model/test/` 应返回可操作错误，不应调用默认模型。
- 如果选择后端保存用户 key，需要增加加密字段、密钥轮换、遮罩更新和安全审计；如果选择 local-only，需要避免 profile sync 把 API key 写入 `UserProfile.settings`。
- 运维文档必须写清楚：`.env` 中的模型 key 只用于本地开发或管理员受控演示，不能作为公开用户额度池。

**Step 2: 数据库上云**

- Neo4j AuraDB Free 创建实例，获取连接 URI → 配置到 .env
- 向量存储：轻量方案用 FAISS（python 库，无需外部服务）。当前 Milvus 改为 FAISS 或继续用 Milvus/Zilliz

**Step 3: Docker Compose 线上版**

抽一个 `docker-compose.prod.yml`，只包含 nginx + Django（Neo4j/Milvus 外部化）。

**Step 4: CI/CD**

简单方案：GitHub Actions → SSH 到服务器 → `git pull` → `docker-compose -f docker-compose.prod.yml up -d --build`

### 前端构建配置

Vite 生产构建时需要区分 API 地址：
- 开发：Vite proxy → `http://127.0.0.1:8000`
- 生产：前端静态文件和 Django 同域，`/api/` 走 nginx proxy

不需要改代码，nginx 反代即可。

### 修改文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `web_backend/ventilation_web/settings.py` | 改造 | 生产环境配置分离（DEBUG, ALLOWED_HOSTS, STATIC_ROOT） |
| `docker-compose.prod.yml` | **新建** | 生产环境编排（nginx + Django） |
| `nginx.conf` | **新建** | nginx 反代配置 |
| `Dockerfile` | **新建/改造** | Django 应用容器化 |
| `.github/workflows/deploy.yml` | **新建** | CI/CD 部署流水线 |
| `scripts/start.sh` | **新建** | 一键生产启动脚本 |

---

## 实施顺序

```
Part 1: 模型配置 (前后端)
    ↓
Part 2 Step 1: 生产配置抽象 (settings / nginx / Dockerfile)
    ↓
Part 2 Step 1.5: 用户密钥策略 (BYOK / 加密存储 / 禁止维护者 key 兜底)
    ↓
Part 2 Step 2: 向量存储可选化 (FAISS 作为轻量替代)
    ↓
Part 2 Step 3: 部署脚本 + CI/CD
```

Part 1 和 Part 2 相对独立，可以并行但建议先做 Part 1（用户能用自己 key 调试），再做 Part 2。部署时不要把项目维护者 key 当作普通用户默认 key；公开环境必须先完成 Step 1.5。

## 验证方式

**模型配置**：
- 切换预设到 OpenAI，填入 key，测试连接通过
- 发送一次辨识请求，确认走的是 OpenAI 模型
- 刷新页面，配置仍在

**上线**：
- `docker-compose -f docker-compose.prod.yml up` 在本机验证
- nginx 正确代理 /api/ 和静态文件
- HTTPS 证书正常
- 未填写用户 API key 时，公开环境不会消耗维护者 `.env` 密钥，而是提示用户先配置自己的模型服务
- 填写用户自己的 key 后，测试连接和真实辨识请求都走该用户 key
