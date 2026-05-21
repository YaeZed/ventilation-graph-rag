# Findings: 系统设计访谈记录

## 论文信息

- **题目**: 数据与法规融合的煤矿通风系统安全隐患智能辨识方法研究
- **学位**: 工学硕士（辽宁工程技术大学 安全科学与工程学院）
- **方向**: 安全技术及工程
- **类型**: 应用基础研究

## 现有系统状态

ventilation-graph-rag 已实现：
- **数据管道**: 通风.docx → LLM 实体抽取 → CSV → Neo4j 导入
- **知识图谱**: Article / Parameter / Requirement / Facility / Location 五类节点，CONSTRAINS / SPECIFIES / APPLIES_TO / INVOLVES_FACILITY / REFERENCES 关系
- **向量索引**: BGE-small-zh-v1.5 嵌入 + Milvus 向量库
- **检索**: 三路由分发（Hybrid / Graph RAG / Combined）+ 递归上下文增强
- **生成**: Qwen-Plus + 四段式约束 prompt（结论/依据/解析/建议）
- **状态**: 命令行交互可正常运行

## 系统架构目标（开题报告规划）

- Qwen2.5-VL 多模态感知 → 图片中提取通风设施状态
- 2025 版《煤矿安全规程》通风一章 RAG 知识库 → 法规对标
- Django+Celery+Redis 后端 + Vue3+TypeScript 前端
- B/S 架构，对话式交互

## 关键设计决策

### 决策 1: 系统边界 → 单 repo
ventilation-graph-rag 内的 pipeline 代码直接扩展为 Web 服务，不拆多 repo。
**理由**: 硕士论文场景，一条链路从数据到前端打通，答辩演示清晰。

### 决策 2: VL 输出格式 → 结构化为主 + 自然语言兜底
Qwen2.5-VL 输出 JSON：`{structured: {...}, description: "..."}`
- structured: 设备类型、距壁距离、风筒状态等可量化字段
- description: 自然语言场景描述，用于向量检索

### 决策 3: Schema → 按场景分组，从规程反向提取
从规程条文提取 3-4 个典型场景，每个场景 4-6 个字段：
- 局部通风机安装（距壁距离、风机间距、风筒完整性、循环风区域）
- 掘进工作面通风（风筒距工作面距离、风量、风速、瓦斯浓度）
- 风门/风墙设施（设施类型、密闭性、连锁状态）
- 通用 description 字段兜底未预见信息

### 决策 4: 检索顺序 → 串行，图谱先，向量后
流程图：
```
图片 → VL 提取 → structured → Cypher 精确查图谱（确定性）
              → description → Milvus 向量检索（兜底）
```
**理由**: 结构化字段确定性高，图谱结果可为向量检索提供剪枝。当前系统两条路互斥调用而非协作，串行避免了冲突消解。

### 决策 5: 图片入口不做路由
图片提取的结构化字段是确定性信息，直接走固定串行流程。
文字查询保留现有 Router。

### 决策 6: VL 集成位置 → Pipeline 前置步骤
Qwen2.5-VL 作为 `VentilationRAGPipeline` 的可选前置步骤。
`query(image_path=...)` 参数控制，VL 提取结果替换原始 query 走检索链路。

### 决策 7: 场景分类 → 两阶段
Stage 1: VL 做场景分类（给定枚举选项）
Stage 2: 按分类结果加载对应子 schema，再调 VL 提取结构化字段
**理由**: 精度优于一步到位，每个阶段任务更聚焦。

### 决策 8: Cypher 生成 → 模板优先 + LLM 兜底
- 每个场景预写 Cypher 模板，结构化字段直接参数绑定
- 字段不完整时走 LLM 生成，同时注入模板作为 few-shot 参考
- 模板文件放在 `agent/rag_system/cypher_templates/`

### 决策 9: 连接管理 → 单例 ConnectionManager
全局一个 ConnectionManager 管理 Neo4j driver + Milvus client。
改为所有模块接收外部传入的 driver，不各自 new。
**理由**: Neo4j/Milvus Python driver 自带线程安全连接池，多模块各自 driver 是浪费且不利于管理。

### 决策 10: 异步 → SSE 流式 + Celery 仅离线任务
Django view 直接 SSE 推送进度，不引入 Celery 处理在线请求。
**理由**: 流式推送让用户看到 Agent 工作状态，体验优于等待转圈。Celery 增加 Redis broker + worker 复杂度，当前阶段不需要。

### 决策 11: 前端 → 对话式交互
一个聊天窗口，统一处理图片上传和文字提问。
所有历史对话可见。

## 实现优先级 & 依赖链

```
Phase 1 (ConnectionManager) → Phase 2 (Cypher 模板) → Phase 3 (VL 集成)
                                                              ↓
Phase 5 (Vue3 前端) ← Phase 4 (Django API + SSE)
```

Priority 顺序的决策逻辑：
1. ConnectionManager 是地基，所有模块依赖它
2. Cypher 模板是 VL 提取的上游依赖（模板字段 = VL schema）
3. VL 集成是技术风险最高环节，需在前端前验证
4. Django API 依赖 1-3 完成
5. Vue3 在 API 稳定后开发效率最高

## 现有代码需要改动的关键文件

| 文件 | 需要的改动 |
|------|-----------|
| `agent/rag_system/ventilation_rag_pipeline.py` | 从 ConnectionManager 取连接；query() 增加 image_path 参数；集成 VisionExtractor |
| `agent/rag_system/ventilation_hybrid_retrieval.py` | 构造函数接受外部 neo4j_driver |
| `agent/rag_system/ventilation_graph_rag_retrieval.py` | 构造函数接受外部 neo4j_driver |
| `agent/rag_system/ventilation_data_preparation.py` | 构造函数接受外部 neo4j_driver |
| `agent/rag_system/ventilation_generation.py` | prompt 已符合要求，生成结论时增加图片相关上下文 |
| 新建 `agent/connection_manager.py` | 单例，管理 Neo4j + Milvus 连接 |
| 新建 `agent/rag_system/ventilation_vision_extractor.py` | 两阶段 VL 提取 |
| 新建 `agent/rag_system/cypher_templates/` | 场景模板文件 |
| 新建 Django 项目 | 后端 API + SSE |
| 新建 Vue3 项目 | 前端对话界面 |

## Notes

- Parameter 节点中的 value_min/value_max 均为从规程提取的精确数值，Cypher 精确匹配可行
- 现有 prompt 模板（ventilation_generation.py）已定义四段式输出，可直接复用
- 系统入口双轨：图片→固定串行流程 / 文字→保留现有 Router
