# 系统架构

## 目标

本项目将《煤矿安全规程》通风相关知识构建为 Neo4j 图谱和 Milvus 向量索引，并通过 RAG、Cypher 模板和 Qwen2.5-VL 支持文字问答与现场图片隐患辨识。

## 顶层数据流

```mermaid
flowchart LR
  A["煤矿安全规程 / CSV"] --> B["Neo4j 知识图谱"]
  B --> C["DataPreparation: Neo4j -> Documents"]
  C --> D["MilvusIndex: BGE embedding -> Milvus"]
  Q["文字问题"] --> R["QueryRouter"]
  IMG["现场图片"] --> VL["Qwen2.5-VL 两阶段抽取"]
  VL --> T["Cypher 模板检索"]
  R --> H["Hybrid Retrieval"]
  R --> G["GraphRAG Retrieval"]
  T --> GEN["Generation: Qwen-Plus Markdown answer"]
  H --> GEN
  G --> GEN
  GEN --> API["Django REST / SSE"]
  API --> UI["Vue3 前端 Markdown 渲染"]
```

## 主要目录

| 路径 | 职责 |
|---|---|
| `agent/data_pipeline/` | Word/CSV 知识抽取与 Neo4j 入库流水线 |
| `agent/rag_system/` | 检索、路由、生成、VL 集成、Cypher 模板 |
| `agent/connection_manager.py` | Neo4j/Milvus 共享连接单例 |
| `web_backend/` | Django API、SSE、上传图片临时文件处理、Celery 占位 |
| `frontend/` | Vue3 + TypeScript + Pinia 对话前端 |
| `docs/grill-me-interview/` | 设计访谈和阶段计划记录 |

## RAG 核心模块

| 模块 | 当前职责 |
|---|---|
| `VentilationRAGPipeline` | 统一初始化连接、索引、检索、生成和图片入口 |
| `VentilationDataPreparationModule` | 从 Neo4j 读取图谱内容并转换为 LangChain `Document` |
| `VentilationMilvusIndexConstruction` | 将文档嵌入并写入 Milvus collection |
| `VentilationHybridRetrieval` | Milvus 向量检索 + Neo4j 图关键词检索 |
| `VentilationGraphRAGRetrieval` | LLM 意图解析 + 多跳图遍历检索 |
| `VentilationQueryRouter` | 根据问题复杂度选择 hybrid / graph / combined |
| `VentilationCypherTemplateEngine` | 根据结构化字段匹配和执行确定性 Cypher 模板 |
| `VentilationVisionExtractor` | Qwen2.5-VL 场景分类和结构化字段抽取 |
| `VentilationGeneration` | 根据检索文档生成 Markdown 格式回答 |

## 图片识别链路

图片请求固定走串行流程：

1. Django 保存上传图片到 `web_backend/media/` 临时文件。
2. `VentilationVisionExtractor` Stage 1 判断场景。
3. Stage 2 按场景 schema 抽取结构化字段和自然语言 description。
4. `VentilationCypherTemplateEngine` 优先用字段执行参数化 Cypher。
5. 文档不足时用 description 走 hybrid 检索兜底。
6. 生成模块输出 Markdown 答案。
7. Django 删除临时图片文件。

## Web 层

Django 不在 app 启动时立即加载 RAG，而是通过 `web_backend/chat/pipeline_service.py` 懒加载单例 `VentilationRAGPipeline`。第一次问答会触发初始化，后续请求复用同一 pipeline。

前端用 Vite 代理 `/api` 到 `127.0.0.1:8000`。助手消息使用 `markdown-it` 渲染 Markdown，并关闭原始 HTML。

