# Plan: VL 辨识增强 + 概念知识层 + Agent 式交互

## Context

用户完成了 ventilation-graph-rag 的 5 个 Phase 开发。实际使用中发现：

1. **VL 认不出核心隐患** — 根本原因是模型不知道"串联通风"等专业概念的定义和识别特征，RAG 知识库只有规程条文，缺少概念解释
2. **回答死板** — 四段式 prompt 约束过严，temperature=0.1，模型没有思考空间
3. **交互不透明** — 用户只看到"正在生成..."，看不到 Agent 处理过程

## 核心思路

给 VL 模型一个"现场学习"的能力：第一次观察图片后，用不确定的概念去查知识库，学到了定义后再重新判断。同时补建概念知识层，让知识库有东西可查。

## 新增数据流

```
图片 → Pass 1 VL (初始观察，不做最终判断)
         │
         ▼
      输出: 看到什么 + 不确定哪些概念 + 初步场景猜测
         │
         ▼
   ConceptRetriever: 拿不确定概念去 Neo4j/Milvus 查定义
   (精确匹配 Concept 节点 + 向量语义搜索 definition 及 visual_clues)
         │
         ▼
   Pass 2 VL (带着概念定义重新分析图片)
         │
         ▼
      输出: scene + structured_fields + key_observations + primary_hazard + risk_level
         │
         ▼
   Cypher 模板精确检索 → 向量兜底 → LLM 灵活生成报告
```

## 修改方案

### Phase A: 构建概念知识层（新增）

新建 `agent/data_pipeline/build_concept_knowledge.py`

**Step 1: 提取概念词表**
- 从 Neo4j 现有节点提取：Parameter.name, Requirement.name, Facility.name
- LLM 扫描所有 Article.content，提取通风专业术语
- 去重后得到概念清单（预计 30-50 个核心概念）

**Step 2: LLM 生成概念定义**
- 对每个概念，调 Qwen-Plus 生成结构化定义：
  ```json
  {
    "name": "串联通风",
    "aliases": ["串联", "一条龙通风"],
    "definition": "...",
    "identification_features": "如何从现场辨别该概念",
    "visual_clues": "图片中可见的特征线索",
    "typical_scenarios": "典型场景描述",
    "hazard_significance": "为什么这是隐患",
    "related_regulation_articles": ["第一百三十三条"]
  }
  ```
- `visual_clues` 是关键——这是 VL 模型用来对比图片的输入
- `identification_features` 是判别依据

**Step 3: 入库**
- Neo4j: 新建 `Concept` 节点，属性包含所有字段，RELATES_TO → Article
- Milvus: 将 name+definition+visual_clues+identification_features 拼接做 embedding，存入 collection `ventilation_concepts`

**Step 4: 创建概念检索工具**
新建 `agent/rag_system/ventilation_concept_retriever.py`
- `search_concepts(queries, extra_text, top_k)` → 从 Neo4j 精确匹配 + Milvus 语义搜索
- 返回结构化的概念卡片列表，供 VL prompt 拼接使用

### Phase B: VL 两轮识别（改造 vision_extractor）

改造 `agent/rag_system/ventilation_vision_extractor.py`

**Pass 1 — 初始观察**（temperature=0.3，给思考空间）：
```
你是一位矿井通风安全检查员，正在查看井下现场照片。

请仔细观察图片，回答以下问题（不用急于下结论）：
1. 你看到了哪些通风设施、设备和环境特征？
2. 图片中哪些现象让你觉得可能存在安全隐患？
3. 有哪些通风专业概念你觉得需要更多定义才能确认？（比如"循环风"、"串联通风"等）
4. 初步猜测属于哪种场景？

返回 JSON：
{
  "raw_observations": "...",
  "uncertain_concepts": ["概念1", "概念2"],
  "preliminary_scene": "...",
  "preliminary_concern": "..."
}
```

**系统中间步骤 — 概念检索**：
- 拿 `uncertain_concepts` 调 ConceptRetriever
- 搜不到的 fallback：用 raw_observations 做向量检索，找语义相近的概念
- 组装"概念参考卡片"注入 Pass 2

**Pass 2 — 带知识重判**（temperature=0.25）：
```
你已获得以下通风专业概念的定义：

【概念参考卡片】
{concept_cards}

现在请带着这些知识，重新分析图片，按场景 schema 提取结构化字段：
（后续同原 Stage 2 prompt + 新增 primary_hazard / key_observations / risk_level）
```

**VisionExtractionResult 扩展**：
- 新增 `raw_observations: str`
- 新增 `uncertain_concepts: List[str]`
- 新增 `concepts_retrieved: List[Dict]` — 检索到的概念定义
- 新增 `key_observations: List[str]`
- 新增 `primary_hazard: str` — 主要隐患判断
- 新增 `risk_level: str` — "正常"/"需要注意"/"疑似隐患"/"明确隐患"

### Phase C: 生成灵活化（改造 generation + pipeline）

改造 `agent/rag_system/ventilation_generation.py`

新增 `generate_image_answer()` 和 `generate_image_answer_stream()`：

prompt 结构：
```
你是一位经验丰富的矿井通风安全检查员，你善于结合图片分析结果和规程要求，做出专业的隐患判断。

【图片分析结果】
- 识别场景：{scene_name}（置信度：{confidence}）
- 风险等级：{risk_level}
- 主要隐患判断：{primary_hazard}
- 关键观察：
  {key_observations 列表}
- 结构化参数：{structured_fields}
- 参考的概念定义：{concept_retrieved 摘要}

【参考规程内容】
{context}

【用户提问】
{question}

请你对图片中的通风安全状况做全面的分析判断。

约束：
- 规程依据必须来自【参考规程内容】，不得编造条款编号或数值
- 如果某个分析点超出了当前检索范围，可以在"补充观察"中标注为"基于现场经验的初步判断"
- 展示你的推理过程，不要直接跳到结论

输出格式参考（自由发挥，不必逐字照搬）：
- **推理过程**：从图片观察到合规性判断的完整思考链
- **合规性结论**：明确判定结果
- **规程依据**：引用条款和数值（必须标注来源）
- **整改建议**：可操作的具体措施
- **补充观察**（可选）：超出检索范围但值得关注的细节
```

关键区别：
- temperature: 0.35（图片查询专用，比文本查询 0.1 高，给思考空间）
- 约束从"禁止推测"改为"区分事实依据和补充观察"
- 输出格式是"参考"而非"模板"（加了"自由发挥，不必逐字照搬"）
- 注入概念定义为上下文

### Phase D: Agent 步骤展示（改造 pipeline + SSE + 前端）

**Pipeline** (`ventilation_rag_pipeline.py`)：
`_query_with_image()` 改为流式版本，yield 步骤事件：

```
step: vision_observe        → "正在初步观察图片..."
step: vision_observe_done   → "识别到 N 个不确定概念：串联通风、...", data: {uncertain_concepts, raw_observations}
step: concept_search        → "正在检索通风概念定义..."
step: concept_search_done   → "检索到 3 个概念定义", data: {concept_count}
step: vision_analyze        → "正在结合概念定义深度分析图片..."
step: vision_analyze_done   → "场景：局部通风机与风筒 | 风险：疑似隐患", data: {scene_name, risk_level, primary_hazard}
step: cypher_match          → "正在匹配规程模板..."
step: cypher_match_done     → "匹配到 N 条相关规程条文", data: {doc_count}
step: generating            → "正在生成辨识报告..."
token: ...                  → 流式文本
done: ...                   → 完成
```

**Django SSE** (`chat/views.py`)：
透传所有 step 事件，不做过滤。

**前端** (`HomeView.vue` + `main.css`)：
- 新增 agent 步骤展示区，渲染在消息卡片内
- 步骤显示为带图标的时间线：观察 → 学习概念 → 重新分析 → 匹配规程 → 生成报告
- 当前步骤有脉冲动画，已完成步骤收起
- 中间结果（场景名、概念名、匹配条文数）显示在对应步骤旁
- 生成开始后步骤区自动折叠为一行摘要，可点击展开

## 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `agent/data_pipeline/build_concept_knowledge.py` | **新建** | 概念词典构建脚本 |
| `agent/rag_system/ventilation_concept_retriever.py` | **新建** | 概念检索工具（Neo4j+Milvus） |
| `agent/rag_system/ventilation_vision_extractor.py` | 重写 | 两轮 VL + 概念检索集成 |
| `agent/rag_system/ventilation_generation.py` | 新增方法 | 图片专用灵活生成 prompt |
| `agent/rag_system/ventilation_rag_pipeline.py` | 改造 | 新流程 + 步骤事件 |
| `web_backend/chat/views.py` | 微调 | SSE 透传 step 事件 |
| `frontend/src/views/HomeView.vue` | 新增组件 | Agent 步骤时间线 |
| `frontend/src/assets/main.css` | 新增样式 | 步骤动画 |
| `web_backend/chat/vision_evaluation.py` | 适配 | 新字段适配 |

## 执行顺序

```
Phase A: 概念知识层构建 (数据基础)
    ↓
Phase B: VL 两轮识别 (依赖 A 的概念检索工具)
    ↓
Phase C: 生成灵活化 (依赖 B 的输出字段)
    ↓
Phase D: Agent 步骤展示 (依赖 C 的流式输出)
```

## 验证方式

1. **概念词典**：`python build_concept_knowledge.py`，检查 Neo4j 中新增 Concept 节点数量（预期 30-50），抽查 5 个概念的定义质量
2. **VL 两轮**：`python ventilation_rag_pipeline.py -q "检查隐患" --image test.jpg`，观察 Pass 1 的 uncertain_concepts 是否合理，Pass 2 是否用了检索到的定义
3. **生成灵活度**：同一张图片问两次，检查回复结构是否不再完全一致，是否有推理过程
4. **SSE 步骤**：`curl -N ... /api/chat/stream/ ...` 观察事件流中 step 事件
5. **前端**：`npm run dev` 上传图片，观察时间线展示
