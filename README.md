# 煤矿通风隐患智能辨识系统

本项目基于《煤矿安全规程》通风相关知识，构建 Neo4j 知识图谱、Milvus 向量索引和 GraphRAG 问答流水线，并提供 Django API 与 Vue3 对话前端。系统支持文字问答、SSE 流式输出，以及基于 Qwen3.5-Omni 的现场图片隐患辨识入口。

## 当前能力

- 图谱 + 向量混合检索，支持多跳 GraphRAG 推理。
- Cypher 模板优先检索，适配通风场景的结构化字段。
- Qwen3.5-Omni 图片理解：先初步观察并列出不确定概念，再检索通风概念定义，最后结合概念卡片完成场景、字段和风险分析。
- Django REST/SSE API，包含开发期账号、用户资料和会话同步接口。
- Vue3 + TypeScript + Pinia 前端，支持图片上传、流式回复、Agent 步骤展示、Markdown 渲染、多会话管理、搜索、归档、登录/注册、导出和会话级输入草稿隔离。
- 前端用户层本地优先：未登录时使用浏览器 localStorage 保存会话、偏好设置和简易用户身份；登录用户按账号隔离本地缓存并同步到 Django 后端。已有账号登录只加载该账号数据，注册新账号时才迁移游客会话。上传图片会压缩为 data URL 用于刷新后预览和报告导出，容量不足时自动降级为仅保存文本记录。
- 图片辨识统一在会话窗口完成：上传图片后，用户描述会和图像一起进入 Qwen3.5-Omni + RAG 辨识链路。

## 目录速览

```text
ventilation-graph-rag/
├── agent/
│   ├── data_pipeline/          # 规程知识抽取、CSV 构建、Neo4j 入库
│   ├── rag_system/             # RAG 检索、生成、Cypher 模板、VL/概念检索集成
│   └── connection_manager.py    # Neo4j/Milvus 共享连接
├── web_backend/                 # Django API + SSE
├── frontend/                    # Vue3 对话前端
├── docs/                        # 架构、API、运行手册、状态记录
├── cypher/                      # Neo4j 导入脚本和 CSV 映射目录
├── docker-compose.yml           # Neo4j + Milvus + etcd + MinIO
└── requirements.txt
```

## 快速启动

### 1. Python 环境

推荐使用项目当前验证过的环境名：

```powershell
conda create -y -n ventilation-identify-system python=3.10
conda activate ventilation-identify-system
D:\Miniconda\envs\ventilation-identify-system\python.exe -m pip install -r requirements.txt
```

### 2. 环境变量

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

图片识别可额外配置：

```ini
QWEN_VL_MODEL=qwen3.5-omni-plus
```

### 3. 启动数据库

```powershell
docker compose up -d
docker compose ps
```

服务端口：

- Neo4j HTTP: `http://127.0.0.1:7474`
- Neo4j Bolt: `bolt://127.0.0.1:7687`
- Milvus: `127.0.0.1:19530`
- MinIO Console: `http://127.0.0.1:9001`

### 4. 后端验证和启动

```powershell
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py check
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py migrate
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\manage.py runserver 127.0.0.1:8000 --noreload
```

### 5. 前端启动

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5173`，默认路由会进入 `/chat`。左侧 Gemini 风格侧边栏支持新建/切换/搜索/重命名/归档/删除对话，统计页在 `/stats`，偏好设置在 `/settings`，账号入口在 `/login` 和 `/register`。

图片辨识无需进入单独页面；在主会话窗口点击 `+` 上传图片，再输入现场描述或检查重点后发送即可。

### 6. 概念知识层

如需让图片链路使用更完整的概念卡片，构建或刷新 `Concept` 节点与 `ventilation_concepts` 向量集合：

```powershell
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\data_pipeline\build_concept_knowledge.py
```

如果 Neo4j 已经有 `Concept` 节点，脚本会跳过 LLM 生成，直接从 Neo4j 读取并刷新 Milvus，避免重复消耗 DashScope 额度。

## 常用验证

```powershell
# Cypher 模板测试
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\test_ventilation_cypher_templates.py

# VL 抽取逻辑测试（fake client，不依赖真实 Qwen-VL）
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\test_ventilation_vision_extractor.py

# 真实图片评估指标测试（fake pipeline，不依赖真实 Qwen-VL）
D:\Miniconda\envs\ventilation-identify-system\python.exe web_backend\chat\test_vision_evaluation.py

# 概念构建脚本语法检查
D:\Miniconda\envs\ventilation-identify-system\python.exe -m py_compile agent\data_pipeline\build_concept_knowledge.py

# 真实 CLI 问答
D:\Miniconda\envs\ventilation-identify-system\python.exe agent\rag_system\ventilation_rag_pipeline.py -q "掘进中的岩巷最低风速要求是多少" --top-k 3

# 前端生产构建
cd frontend
npm run build
```

## API 简例

```powershell
curl -X POST http://127.0.0.1:8000/api/chat/ `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"掘进中的岩巷最低风速要求是多少\",\"top_k\":3}"
```

流式接口：

```powershell
curl -N -X POST http://127.0.0.1:8000/api/chat/stream/ `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"矿井有害气体最高允许浓度范围是什么\",\"top_k\":5}"
```

更多接口说明见 [docs/api.md](docs/api.md)。

## 文档

- [docs/architecture.md](docs/architecture.md)：系统架构和数据流
- [docs/api.md](docs/api.md)：REST/SSE API
- [docs/runbook.md](docs/runbook.md)：运行、验证、排障
- [docs/status.md](docs/status.md)：当前状态和剩余风险
- [docs/grill-me-interview/](docs/grill-me-interview/)：设计访谈、阶段计划和开发日志
