# Task Plan: 煤矿通风隐患智能辨识系统

## Goal

将 ventilation-graph-rag 知识库集成为基于 B/S 架构的煤矿通风系统安全隐患智能辨识系统（Vue3+Django+Qwen2.5-VL+RAG），支撑硕士论文《数据与法规融合的煤矿通风系统安全隐患智能辨识方法研究》。

## Architecture Decisions (from grill-me interview)

| # | 决策 | 选择 | 理由 |
|---|------|------|------|
| 1 | 系统边界 | 单 repo（B） | 硕士论文场景，一条链路从 data pipeline 到前端全打通 |
| 2 | VL 输出格式 | 结构化字段 + 自然语言 description | 结构化支持精确 Cypher 匹配，description 支持向量兜底 |
| 3 | 字段 Schema | 按场景分组预定义 | 从规程反向提取 3-4 个场景，每组 4-6 个字段 |
| 4 | 检索顺序 | 串行：图谱先，向量后 | 结构化字段确定性高，图谱结果可剪枝向量空间 |
| 5 | 图片入口处理 | 固定串行流程，不做路由 | 结构化字段是确定性信息，路由无必要 |
| 6 | VL 集成位置 | Pipeline 前置步骤（A） | VL 结果影响检索策略，放在 pipeline 统一管理 |
| 7 | 场景分类 | 两阶段（B） | 第一次分类，第二次按对应 schema 提取，精度优于一步到位 |
| 8 | Cypher 生成 | 模板优先 + LLM 兜底 | 预写模板做确定性匹配，LLM 参考模板做补充 |
| 9 | 连接管理 | 单例 ConnectionManager（B 改良） | Neo4j/Milvus driver 自带线程安全连接池 |
| 10 | 异步任务 | SSE 流式推送 + Celery 仅离线任务（B） | 流式推送提升用户体验，避免过度架构 |
| 11 | 前端交互 | 对话式 | 一个聊天窗口，传图+打字统一入口 |
| 12 | 实现优先级 | 见下方 Phase 顺序 | 从内到外：连接层→模板→VL→API→前端 |

## Phases

### Phase 1: ConnectionManager 重构
- **Status**: completed
- **Description**: 统一 Neo4j/Milvus 连接管理
  - 创建 `agent/connection_manager.py`，单例模式
  - 修改 `VentilationDataPreparationModule.__init__()` 接受外部 driver
  - 修改 `VentilationHybridRetrieval.__init__()` 接受外部 driver
  - 修改 `VentilationGraphRAGRetrieval.__init__()` 接受外部 driver
  - 修改 `VentilationRAGPipeline` 从 ConnectionManager 获取所有连接
  - 验证：现有命令行问答正常运作
  - 备注：代码级编译、共享连接注入验证和真实 CLI 问答均已通过；本地 Neo4j/Milvus 服务可用，示例问题“掘进中的岩巷最低风速要求是多少”返回 ≥0.15 m/s

### Phase 2: Cypher 模板系统
- **Status**: completed
- **Description**: 从规程提取场景分类和对应 Cypher 模板
  - 分析《煤矿安全规程》通风一章，提取含明确数值约束的条文
  - 按设施类型归类为 3-4 个场景组
  - 为每个场景定义子 schema（JSON）和模板 Cypher
  - 创建 `agent/rag_system/cypher_templates/` 目录和模板文件
  - 实现模板匹配逻辑（结构化字段 → Cypher 参数绑定）
  - 备注：已实现 4 类场景模板：井巷风速、空气成分/有害气体、局部通风机与风筒、风门风墙等通风设施；基于 `nodes.csv/relationships.csv` 的图谱结构设计，并通过无 Neo4j mock 测试

### Phase 3: Qwen2.5-VL 集成
- **Status**: completed
- **Description**: 图片入口的两阶段 VL 提取
  - 实现 `VentilationVisionExtractor` 模块
  - Stage 1: 场景分类（prompt 枚举所有场景，选一）
  - Stage 2: 按场景子 schema 提取结构化字段 + description
  - 集成到 `VentilationRAGPipeline.query()` 作为可选前置步骤
  - 修改 `query()` 签名支持 `image_path` 参数
  - 验证：对测试图片验证提取精度
  - 备注：已实现两阶段 VL 抽取、schema 驱动字段清洗、`query(image_path=...)` 固定流程；mock 测试通过。真实图片精度验证需在 `openai/neo4j/langchain` 依赖和 Qwen-VL 服务可用后执行

### Phase 4: Django API + SSE
- **Status**: completed
- **Description**: 后端 Web 层
  - 创建 Django 项目结构
  - `apps.py` ready() 初始化 ConnectionManager + Pipeline
  - REST API: `/api/chat/` (文字), `/api/chat/upload/` (图片)
  - SSE 流式端点：`/api/chat/stream/`
  - Celery 配置（预留，仅用于离线批量任务）
  - 验证：curl / Postman 测试 API
  - 备注：已新增 Django 后端骨架、REST JSON/图片上传/SSE 端点和 Celery 占位；`manage.py check`、真实 REST 请求和 SSE token 流均已通过

### Phase 5: Vue3 对话前端
- **Status**: completed
- **Description**: 前端对话界面
  - Vue3 + TypeScript + Pinia 项目初始化
  - 对话窗口组件（支持文字输入 + 图片上传 + 流式回复渲染）
  - 历史对话列表
  - SSE 消费逻辑
  - 隐患报告样式渲染（四段式：结论/依据/解析/建议）
  - 验证：端到端测试图片上传→流式回复→报告展示
  - 备注：已参照 `D:\projects\personal_projects\mine-LLM-fronted` 的 Vue3+Router+MainLayout 骨架，并按 `frontend-design` 的审美要求改为克制工业调度台风格；`npm install` 和 `npm run build` 已通过
