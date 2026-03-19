"""
矿井通风安全规程 - 混合检索模块

集成向量检索 (Milvus) 和知识图谱检索 (Neo4j)，提供双层检索 (Dual-level Retrieval) 能力：
1. 实体级检索 (Entity-level)：通过关键词锁定具体的条款、设施、参数节点。
2. 主题级检索 (Topic-level)：通过全局关键词搜索相关的安全要求和逻辑关联。
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
from langchain_core.documents import Document

# 导入本地通风专用模块
from ventilation_graph_indexing import VentilationGraphIndexingModule, EntityKeyValue

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'entity' or 'topic' or 'vector'
    metadata: Dict[str, Any]

class VentilationHybridRetrieval:
    """通风安全规程混合检索模块"""

    def __init__(self, config, data_module, milvus_module, llm_client):
        self.config = config
        self.data_module = data_module
        self.milvus_module = milvus_module
        self.llm_client = llm_client
        
        # 初始化图索引模块
        self.graph_indexing = VentilationGraphIndexingModule(config, llm_client)
        self.graph_indexed = False
        
        # Neo4j 连接
        uri = getattr(config, "neo4j_uri", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        user = getattr(config, "neo4j_user", os.getenv("NEO4J_USER", "neo4j"))
        password = getattr(config, "neo4j_password", os.getenv("NEO4J_PASSWORD", "password"))
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        logger.info("通风混合检索模块初始化完成")

    def initialize(self, chunks: Optional[List[Document]] = None):
        """初始化检索器，构建图索引"""
        if not self.graph_indexed:
            self._build_graph_index()
            self.graph_indexed = True
            logger.info("图索引构建完成")

    def _build_graph_index(self):
        """调用图索引模块构建通风规程索引"""
        articles = getattr(self.data_module, 'articles', [])
        parameters = getattr(self.data_module, 'parameters', [])
        requirements = getattr(self.data_module, 'requirements', [])
        facilities = getattr(self.data_module, 'facilities', [])
        locations = getattr(self.data_module, 'locations', [])
        
        # 调用重构后的独立方法
        self.graph_indexing.create_entity_key_values(
            articles, parameters, requirements, facilities, locations
        )
        
        # 处理关系
        relationships = getattr(self.data_module, 'relationships', [])
        self.graph_indexing.create_relation_key_values(relationships)
        
        # 去重
        self.graph_indexing.deduplicate_entities_and_relations()

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """执行混合检索：结合向量搜索和双层图搜索"""
        self.initialize()
        
        # 1. 向量搜索（增强版）
        vector_docs = self.vector_search_enhanced(query, top_k)
        
        # 2. 图谱搜索（双层）
        graph_docs = self.dual_level_retrieval(query, top_k)
        
        # 3. 合并并去重
        all_docs = vector_docs + graph_docs
        seen_content = set()
        unique_docs = []
        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        
        # 4. 风速查询强制注入第一百五十七条（表6）
        # 原因：第一百五十七条正文仅一句话，向量相似度低，但它是所有风速合规判断的核心依据
        WIND_SPEED_TRIGGERS = ["风速", "允许速度", "最低风速", "最高风速", "合规", "违规"]
        if any(kw in query for kw in WIND_SPEED_TRIGGERS):
            already_has_157 = any(
                "第一百五十七条" in (doc.metadata.get("article_name", "") + doc.page_content)
                for doc in unique_docs
            )
            if not already_has_157:
                injected = self._fetch_article_157()
                if injected:
                    unique_docs = injected + unique_docs
                    logger.info("风速查询：已强制注入第一百五十七条（表6）到检索结果")

        # 5. 递归增强：为所有命中的条款自动丰富表格和关联引用摘要
        unique_docs = self._enrich_documents_with_recursive_context(unique_docs)
        
        return unique_docs[:top_k * 2]

    def _fetch_article_157(self) -> List[Document]:
        """
        专用方法：从 Neo4j 直查第一百五十七条（表6 井巷中的允许风流速度），
        构建含完整参数表的 Document，用于风速查询时强制注入。
        """
        try:
            with self.driver.session() as session:
                cypher = """
                MATCH (a:Article)
                WHERE a.name = '第一百五十七条' OR a.node_id CONTAINS '157'
                WITH a
                OPTIONAL MATCH (a)-[:CONSTRAINS]->(p:Parameter)
                OPTIONAL MATCH (p)-[:APPLIES_TO]->(l:Location)
                RETURN a.node_id AS node_id, a.name AS name,
                       a.content AS content, a.title AS title,
                       collect(DISTINCT {name: p.name, min: p.value_min, max: p.value_max,
                                         unit: p.unit, location: l.name}) AS params
                LIMIT 1
                """
                res = session.run(cypher)
                record = res.single()
                if not record:
                    return []

                full_text = (
                    f"【第一百五十七条】{record['title'] or ''}\n"
                    f"{record['content'] or '井巷中的风流速度应当符合表6要求。'}"
                )

                params = [p for p in record["params"] if p.get("name")]
                if params:
                    table_md = "\n\n### [规程附件：技术参数对照表（表6 井巷中的允许风流速度）]\n| 参数名称 | 适用地点 | 最小值 | 最大值 | 单位 |\n| :--- | :--- | :--- | :--- | :--- |\n"
                    for p in params:
                        min_val = p['min'] if p['min'] is not None else "-"
                        max_val = p['max'] if p['max'] is not None else "-"
                        loc_val = p.get('location') or "-"
                        table_md += f"| {p['name']} | {loc_val} | {min_val} | {max_val} | {p.get('unit') or '-'} |\n"
                    full_text += table_md

                return [Document(
                    page_content=full_text,
                    metadata={
                        "node_id": record["node_id"],
                        "article_name": "第一百五十七条",
                        "retrieval_level": "mandatory_inject",
                    }
                )]
        except Exception as e:
            logger.error(f"强制注入第一百五十七条失败: {e}")
            return []

    def _enrich_documents_with_recursive_context(self, docs: List[Document]) -> List[Document]:
        """
        关键优化：为命中条款自动组合 Neo4j 中的 [参数表] 和 [1-hop 关联条款摘要]。

        node_id 解析优先级（由高到低）：
          1. metadata['parent_id']    —— chunk 文档的父条款 ID (art_xxx)
          2. metadata['article_name'] —— 条款中文名（第一百xx条）
          3. metadata['node_id']      —— 直接条款节点 ID
        """
        enriched_docs = []
        for doc in docs:
            # ── 幂等检查：已增强过的文档跳过 ──────────────────
            if "[规程附件：技术参数对照表]" in doc.page_content:
                enriched_docs.append(doc)
                continue

            # ── 定位 Neo4j 中对应的 Article 节点 ID ──────────
            meta = doc.metadata
            lookup_id = (
                meta.get("parent_id")       # chunk 的父条款 art_xxx
                or meta.get("article_name") # 中文条款名如"第一百五十七条"
                or meta.get("node_id")      # 直接使用 node_id
            )
            if not lookup_id:
                enriched_docs.append(doc)
                continue

            try:
                with self.driver.session() as session:
                    cypher = """
                    MATCH (a:Article)
                    WHERE a.node_id = $nid OR a.name = $nid
                    WITH a
                    OPTIONAL MATCH (a)-[:CONSTRAINS]->(p:Parameter)
                    OPTIONAL MATCH (p)-[:APPLIES_TO]->(l:Location)
                    OPTIONAL MATCH (a)-[:RELATED_TO|REFERENCES]-(ref:Article)
                    RETURN a.node_id AS node_id, a.name AS name,
                           collect(DISTINCT {name: p.name, min: p.value_min, max: p.value_max,
                                            unit: p.unit, location: l.name}) AS params,
                           collect(DISTINCT {name: ref.name, content: ref.content}) AS related_docs
                    """
                    res = session.run(cypher, {"nid": lookup_id})
                    record = res.single()
                    if record:
                        full_text = doc.page_content

                        # A. 附加技术参数对照表（含适用地点列）
                        params = [p for p in record["params"] if p.get("name")]
                        if params:
                            table_md = "\n\n### [规程附件：技术参数对照表]\n| 参数名称 | 适用地点 | 最小值 | 最大值 | 单位 |\n| :--- | :--- | :--- | :--- | :--- |\n"
                            for p in params:
                                min_val = p['min'] if p['min'] is not None else "-"
                                max_val = p['max'] if p['max'] is not None else "-"
                                loc_val = p.get('location') or "-"
                                table_md += f"| {p['name']} | {loc_val} | {min_val} | {max_val} | {p.get('unit') or '-'} |\n"
                            full_text += table_md

                        # B. 附加关联引用条款摘要（防止 content 为 None）
                        related = [r for r in record["related_docs"] if r.get("name")]
                        if related:
                            ref_section = "\n\n### [关联引用条款摘要]\n"
                            for r in related:
                                c = r.get('content') or ''
                                summary = c[:200] + "..." if len(c) > 200 else c
                                ref_section += f"- **{r['name']}**: {summary}\n"
                            full_text += ref_section

                        doc.page_content = full_text
                        logger.info(f"已为混合检索命中的条款 {lookup_id} 执行递归语境增强")
            except Exception as e:
                logger.error(f"混合检索递归增强失败 ({lookup_id}): {e}")
            enriched_docs.append(doc)
        return enriched_docs

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """向量检索并补全通风邻居信息"""
        try:
            # 使用重构后的 Milvus 模块执行相似度搜索
            vector_res = self.milvus_module.similarity_search(query, k=top_k*2)
            docs = []
            for res in vector_res:
                content = res.get("text", "")
                metadata = res.get("metadata", {})
                node_id = metadata.get("node_id")
                
                # 加载 Neo4j 邻居信息增强上下文
                if node_id:
                    nbs = self._get_node_neighbors(node_id)
                    if nbs:
                        content += f"\n[关联参考]: {', '.join(nbs)}"
                
                # 统一元数据字段，确保生成模块能识别
                metadata["article_name"] = metadata.get("article_name") or metadata.get("name") or "未知条款"
                
                docs.append(Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "score": res.get("score", 0.0),
                        "retrieval_level": "vector"
                    }
                ))
            return docs[:top_k]
        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """图谱版：合并实体级和主题级检索结果"""
        entity_kw, topic_kw = self.extract_query_keywords(query)
        
        entity_res = self.entity_level_retrieval(entity_kw, top_k)
        topic_res  = self.topic_level_retrieval(topic_kw, top_k)
        
        all_res = entity_res + topic_res
        seen = set()
        unique = []
        for r in sorted(all_res, key=lambda x: x.relevance_score, reverse=True):
            if r.node_id not in seen:
                seen.add(r.node_id)
                unique.append(r)
        
        docs = []
        for r in unique[:top_k]:
            meta = r.metadata.copy()
            meta["article_name"] = meta.get("name") or meta.get("article_name") or "未知条款"
            meta["node_id"] = r.node_id
            meta["retrieval_level"] = r.retrieval_level
            docs.append(Document(page_content=r.content, metadata=meta))
        return docs

    def entity_level_retrieval(self, keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """实体级检索：直接通过索引键查找本地 KV 存储"""
        results = []
        for kw in keywords:
            # 从图索引中获取实体
            entities = self.graph_indexing.get_entities_by_key(kw)
            for ent in entities:
                results.append(RetrievalResult(
                    content=ent.value_content,
                    node_id=ent.metadata["node_id"],
                    node_type=ent.entity_type,
                    relevance_score=1.0, # 关键词精确匹配
                    retrieval_level="entity",
                    metadata=ent.metadata
                ))
        return results[:top_k]

    def topic_level_retrieval(self, keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """主题级检索：通过关系键查找本地 KV 存储"""
        results = []
        for kw in keywords:
            relations = self.graph_indexing.get_relations_by_key(kw)
            for rel in relations:
                results.append(RetrievalResult(
                    content=rel.value_content,
                    node_id=rel.relation_id,
                    node_type="Relation",
                    relevance_score=0.8,
                    retrieval_level="topic",
                    metadata=rel.metadata
                ))
        return results[:top_k]

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """使用 LLM 提取通风安全领域的检索关键词"""
        prompt = f"""作为矿井通风专家，从以下用户问题中提取检索关键词：
问题：{query}

请返回 JSON 格式：
{{
  "entity_keywords": ["设备/条款名"],
  "topic_keywords": ["安全主题/风险/要求类型"]
}}"""
        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.config, 'llm_model', 'qwen-plus'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            # 处理可能的 Markdown 格式
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            data = json.loads(content)
            return data.get("entity_keywords", []), data.get("topic_keywords", [])
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return [query], [query]

    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """Neo4j 查询邻居节点名称 - 显式使用通风领域标签"""
        try:
            with self.driver.session() as session:
                # 显式使用 node_id 属性，避免基类的 nodeId 错误
                result = session.run("""
                    MATCH (n)-[]-(nb)
                    WHERE n.node_id = $nid
                    RETURN nb.name AS name
                    LIMIT $limit
                """, {"nid": node_id, "limit": max_neighbors})
                return [r["name"] for r in result if r["name"]]
        except Exception as e:
            logger.error(f"获取邻居失败: {e}")
            return []

    def close(self):
        """资源释放"""
        if self.driver:
            self.driver.close()
        logger.info("混合检索连接已关闭")
