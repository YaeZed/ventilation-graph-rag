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

## 前端构建

```powershell
cd frontend
npm run build
```

该命令同时执行 `vue-tsc --build` 和 `vite build`。

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

### 图片请求很久没有完整报告

图片链路会先调用 Qwen3.5-Omni 做观察，再检索概念、复核图片、检索规程并生成报告。前端应显示 `vision_observe`、`concept_search`、`vision_analyze`、`cypher_match`、`generating` 等步骤；如果只停在某一步，优先检查 DashScope 额度、模型名 `QWEN_VL_MODEL=qwen3.5-omni-plus`、后端日志和 SSE `error` 事件。

### DashScope 免费额度或请求中断

如果返回 `AllocationQuota.FreeTierOnly`，说明当前 DashScope 账号免费额度耗尽，需要在管理控制台关闭 free-tier-only 限制或更换可用 key。浏览器出现 `BodyStreamBuffer was aborted` 多半是长时间图片请求被前端或代理中断；当前前端图片请求和 SSE 已使用更长超时，仍可通过后端日志确认真实异常。
