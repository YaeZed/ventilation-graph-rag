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
    """通风安全规程混合检索模块 - 独立版"""

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
        
        return unique_docs[:top_k * 2]

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
