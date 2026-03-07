# 矿井通风安全规程 GraphRAG 系统 (Ventilation Safety GraphRAG)

本项目是一个基于《煤矿安全规程》（通风部分）构建的 **GraphRAG（图谱+向量增强检索生成）** 智能问答系统。它将非结构化的规程文本解析为 Neo4j 知识图谱，并结合 Milvus 向量数据库与大语言模型 (Qwen-Plus)，实现对复杂通风安全标准的精准提问与多跳推理回答。

## 🛠️ 环境要求

- Python 3.9+
- Docker & Docker Compose
- [Neo4j](https://neo4j.com/) (图数据库)
- [Milvus](https://milvus.io/) (向量数据库)
- 阿里云 DashScope API Key (用于调用通义千问模型)

---

## 🚀 快速启动指南 (部署到新电脑)

### 1. 获取代码与依赖安装

首先克隆本仓库到本地，并进入项目目录：

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
```

接着，建议创建一个虚拟环境来安装依赖包：

```bash
conda create -n graph-rag python=3.10
conda activate graph-rag
pip install -r requirements.txt
```

_(注意：请确保 `requirements.txt` 中包含 `neo4j`, `pymilvus`, `openai`, `python-dotenv`, `python-docx` 等核心依赖)_

### 2. 启动底层数据库

系统运行依赖 Neo4j 和 Milvus，请确保 Docker Desktop 已经启动。

**启动 Neo4j:**
在项目根目录运行 docker-compose：

```bash
docker-compose up -d neo4j
```

_启动后可访问 `http://localhost:7474`，默认用户名/密码在代码中配置为 `neo4j` / `160722yaesakura`。_

**启动 Milvus:**
因为本地 `docker-compose.yml` 中未包含 Milvus，所以你需要使用 Milvus 官方提供的单机版或利用 Docker Desktop 的方式把 Milvus 启动起来，并确保 `19530` 端口开启即可。

### 3. 配置环境变量

在项目根目录下创建一个名为 `.env` 的文件，填入你的通义千问 API 密钥：

```ini
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. 知识抽取与图谱构建 (首次运行必需)

因为新电脑的数据库是空的，你需要先对《煤矿安全规程》Word 文档进行大模型智能化信息抽取，并组装写入 Neo4j。

```bash
cd agent/data_pipeline

# 该脚本将调用大模型抽取实体与逻辑，把非结构化的条款变成图节点和关系
python run_ventilation_agent.py

# 请根据你的业务逻辑和导入脚本，确保所有节点成功导入 Neo4j 数据库。
```

### 5. 向量索引构建与 RAG 系统启动

图数据入库完成后，我们需要把图节点关联的内容打包成富文档并建立 Milvus 向量索引，同时启动问答系统进行测试。

```bash
cd ../rag_system
# 带有 --build-index 标志，强制从 Neo4j 抽取数据并构建/更新 Milvus 索引
python ventilation_rag_pipeline.py --build-index
```

当终端提示就绪后，你就可以直接在控制台输入问题，体验你的通风规程问答系统了。

---

## 💻 日常使用

完成首次数据入库和建索引后，之后常规启动问答系统，只需保证 Docker 环境里的图库和向量库是运行状态：

```bash
cd agent/rag_system
python ventilation_rag_pipeline.py
```

_(已存在向量索引的情况下，不需要加 `--build-index` 参数)_

## 📂 核心项目结构简述

- `agent/data_pipeline/`: **知识入库与图谱构建**流水线，负责文档解析 (Word)、智能信息抽取与图数据库入库。
- `agent/rag_system/`: **检索与生成系统**，包含图检索、混合检索、多跳 GraphRAG 推理、智能路由等检索增强能力。
