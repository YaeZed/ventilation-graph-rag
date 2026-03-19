"""
矿井通风安全规程 - 图 RAG 检索模块

完全独立的图 RAG 检索模块，执行复杂的图结构推理和多步路径查询。
支持对通风参数、设施安装要求、巷道合规性进行深度图搜索。
"""

import json
import logging
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """查询类型枚举"""
    ENTITY_RELATION = "entity_relation"  # 实体关系查询
    MULTI_HOP = "multi_hop"              # 多跳查询
    SUBGRAPH = "subgraph"                # 子图查询
    PATH_FINDING = "path_finding"        # 路径查找
    CLUSTERING = "clustering"            # 聚类查询

@dataclass
class GraphQuery:
    """图查询结构"""
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None
    relation_types: List[str] = None
    max_depth: int = 2
    max_nodes: int = 50
    constraints: Dict[str, Any] = None

@dataclass
class GraphPath:
    """图路径结构"""
    nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    path_length: int
    relevance_score: float
    path_type: str

@dataclass
class KnowledgeSubgraph:
    """知识子图结构"""
    central_nodes: List[Dict[str, Any]]
    connected_nodes: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    graph_metrics: Dict[str, float]
    reasoning_chains: List[str]

class VentilationGraphRAGRetrieval:
    """
    通风安全规程图 RAG 检索系统 
    核心能力：
    1. 通风规程意图理解：识别规程条款、设施、参数间的关联意图
    2. 多跳规程链条追踪：解决“A故障导致B要求生效”这类逻辑链
    3. 结构化子图提取：获取针对某一地点或设施的完整安全知识网络
    """
    
    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None
        
        # 缓存系统
        self.entity_cache = {}
        self.relation_cache = {}
        
        # 连接准备
        self._init_connection()

    def _init_connection(self):
        """初始化 Neo4j 连接"""
        try:
            uri = getattr(self.config, "neo4j_uri", "")
            user = getattr(self.config, "neo4j_user", "")
            pwd = getattr(self.config, "neo4j_password", "")
            self.driver = GraphDatabase.driver(uri, auth=(user, pwd))
            logger.info("图 RAG 检索 Neo4j 连接成功")
        except Exception as e:
            logger.error(f"图 RAG Neo4j 连接失败: {e}")

    def initialize(self):
        """构建初始化图索引"""
        if not self.driver: return
        self._build_graph_index()

    def _build_graph_index(self):
        """构建通风领域节点索引"""
        logger.info("构建通风规程图结构索引...")
        try:
            with self.driver.session() as session:
                # 使用通用的 node_id 属性
                entity_query = """
                MATCH (n)
                WHERE n.node_id IS NOT NULL
                WITH n, COUNT { (n)--() } AS degree
                RETURN labels(n) AS node_labels, n.node_id AS node_id,
                       n.name AS name,
                       COALESCE(n.title, n.content, '') AS category,
                       degree
                ORDER BY degree DESC
                LIMIT 1000
                """
                result = session.run(entity_query)
                for record in result:
                    nid = record["node_id"]
                    self.entity_cache[nid] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"]
                    }
                
                logger.info(f"图索引加载成功: {len(self.entity_cache)} 个实体")
        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def understand_graph_query(self, query: str) -> GraphQuery:
        """解析用户问题的通风规程图检索意图"""
        prompt = f"""你是矿井通风安全专家，请分析以下查询并映射到通风规程图谱结构。

已知图谱 Schema：
- 节点类型：Article(条款), Parameter(数值参数), Requirement(安全要求), Facility(通风设施), Location(适用地点)
- 主要关系：CONSTRAINS, APPLIES_TO, SPECIFIES, INVOLVES_FACILITY, REFERENCES, RELATED_TO

查询：{query}

请识别以下字段并返回 JSON：
1. query_type: entity_relation, multi_hop, subgraph, path_finding
2. source_entities: 图中存在的具体节点名（如"第一百条"、"局部通风机"）
3. target_entities: 针对路径查找的终点，通常为空 []
4. relation_types: 相关的关系名
5. max_depth: 遍历深度 (1-3)
6. constraints: 属性过滤条件

返回 JSON 示例：
{{
  "query_type": "subgraph",
  "source_entities": ["局部通风机"],
  "target_entities": [],
  "relation_types": ["INVOLVES_FACILITY", "SPECIFIES"],
  "max_depth": 2,
  "constraints": {{}}
}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.config, 'llm_model', 'qwen-plus'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"): content = content[7:-3].strip()
            result = json.loads(content)
            
            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2)
            )
        except Exception as e:
            logger.error(f"理解图查询意图失败: {e}")
            return GraphQuery(query_type=QueryType.SUBGRAPH, source_entities=[query])

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """图 RAG 主搜索接口"""
        if not self.driver: return []
        
        # 1. 意图理解
        graph_query = self.understand_graph_query(query)
        
        # 2. 遍历执行
        results = []
        if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING, QueryType.ENTITY_RELATION]:
            paths = self._execute_multi_hop(graph_query)
            results = self._paths_to_documents(paths)
        elif graph_query.query_type == QueryType.SUBGRAPH:
            subgraph = self._extract_subgraph(graph_query)
            results = self._subgraph_to_documents(subgraph)
            
        return results[:top_k]

    def _execute_multi_hop(self, graph_query: GraphQuery) -> List[GraphPath]:
        """多跳图遍历核心逻辑"""
        paths = []
        try:
            with self.driver.session() as session:
                depth = graph_query.max_depth
                # 模糊匹配节点名，并向外拓展找寻 Article 节点
                cypher = f"""
                UNWIND $source_entities AS sname
                MATCH (s) WHERE s.name CONTAINS sname OR s.node_id = sname
                MATCH p = (s)-[*0..{depth}]-(t)
                WHERE (t:Article OR labels(t)[0] = 'Article' OR t.node_id STARTS WITH 'art_')
                WITH p, length(p) AS len, nodes(p) AS ns, relationships(p) AS rs
                ORDER BY len ASC
                LIMIT 20
                RETURN ns, rs, len
                """
                res = session.run(cypher, {"source_entities": graph_query.source_entities})
                for record in res:
                    paths.append(GraphPath(
                        nodes=[dict(n) for n in record["ns"]],
                        relationships=[dict(r) for r in record["rs"]],
                        path_length=record["len"],
                        relevance_score=1.0/(record["len"] + 1),
                        path_type="multi_hop"
                    ))
                
                if not paths:
                    logger.warning(f"直接路径未找到条款，尝试利用关联关系进行 2 跳外推...")
                    fallback_cypher = """
                    UNWIND $source_entities AS sname
                    MATCH (s) WHERE s.name CONTAINS sname OR s.node_id = sname
                    MATCH (s)-[*1..2]-(a:Article)
                    RETURN DISTINCT a.node_id AS nid LIMIT 5
                    """
                    f_res = session.run(fallback_cypher, {"source_entities": graph_query.source_entities})
                    nids = [r["nid"] for r in f_res]
                    if nids:
                        logger.info(f"回退查找成功，找到关联条款 ID: {nids}")
                        # 构造虚拟路径以触发内容回查
                        for nid in nids:
                            paths.append(GraphPath(nodes=[{"node_id": nid}], relationships=[], path_length=1, relevance_score=0.5, path_type="fallback"))
                            
            logger.info(f"多跳遍历完成，初步路径数: {len(paths)}")
        except Exception as e:
            logger.error(f"执行多跳遍历解析失败: {e}")
        return paths

    def _extract_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """子图提取逻辑（支持多实体合并）"""
        central_nodes = []
        connected_nodes = []
        relationships = []
        try:
            with self.driver.session() as session:
                cypher = f"""
                UNWIND $source_entities AS sname
                MATCH (s) WHERE s.name CONTAINS sname OR s.node_id = sname
                MATCH (s)-[r*1..{graph_query.max_depth}]-(n)
                RETURN s, collect(DISTINCT n) AS neighbors, collect(DISTINCT r) AS rels
                """
                result = session.run(cypher, {"source_entities": graph_query.source_entities})
                # 遍历所有记录（支持多个匹配实体），而非只取第一条
                for record in result:
                    central_nodes.append(dict(record["s"]))
                    connected_nodes.extend([dict(n) for n in record["neighbors"]])
                    relationships.extend([dict(r) for r in record["rels"]])
        except Exception as e:
            logger.error(f"子图提取解析失败: {e}")
        return KnowledgeSubgraph(central_nodes, connected_nodes, relationships, {}, [])

    def _fetch_article_content(self, node_ids: list, max_articles: int = 5) -> list:
        """从 Neo4j 批量回查条款详情，并执行 1-hop 递归增强（表格 + 关联条款）"""
        if not node_ids: return []
        docs = []
        try:
            with self.driver.session() as session:
                # 强化查询：主条款 + 结构化参数 + 关联条款摘要
                cypher = """
                UNWIND $ids AS nid
                MATCH (a:Article) 
                WHERE a.node_id = nid OR a.name = nid OR a.name CONTAINS nid
                WITH DISTINCT a
                OPTIONAL MATCH (a)-[:CONSTRAINS]->(p:Parameter)
                OPTIONAL MATCH (p)-[:APPLIES_TO]->(l:Location)
                OPTIONAL MATCH (a)-[:RELATED_TO|REFERENCES]-(ref:Article)
                RETURN a.node_id AS node_id, a.name AS name, a.title AS title, a.content AS content,
                       collect(DISTINCT {name: p.name, min: p.value_min, max: p.value_max,
                                         unit: p.unit, location: l.name}) AS params,
                       collect(DISTINCT {name: ref.name, content: ref.content}) AS related_docs
                LIMIT $limit
                """
                res = session.run(cypher, {"ids": list(node_ids), "limit": max_articles})
                for record in res:
                    main_content = record["content"]
                    if not main_content: continue
                    
                    full_text = f"【定位条款：{record['name']}】\n# {record['title']}\n\n{main_content}"
                    
                    # 1. 动态重组参数表格（含适用地点列）
                    params = [p for p in record["params"] if p.get("name")]
                    if params:
                        table_md = "\n\n### [规程附件：技术参数对照表]\n| 参数名称 | 适用地点 | 最小值 | 最大值 | 单位 |\n| :--- | :--- | :--- | :--- | :--- |\n"
                        for p in params:
                            min_val = p['min'] if p['min'] is not None else "-"
                            max_val = p['max'] if p['max'] is not None else "-"
                            loc_val = p.get('location') or "-"
                            table_md += f"| {p['name']} | {loc_val} | {min_val} | {max_val} | {p.get('unit') or '-'} |\n"
                        full_text += table_md
                    
                    # 2. 递归组合 1-hop 关联条款摘要
                    related = [r for r in record["related_docs"] if r.get("name")]
                    if related:
                        ref_section = "\n\n### [关联引用条款参考]\n"
                        for r in related:
                            c = r.get('content') or ''
                            summary = c[:200] + "..." if len(c) > 200 else c
                            ref_section += f"- **{r['name']}**: {summary}\n"
                        full_text += ref_section
                    
                    docs.append(Document(
                        page_content=full_text,
                        metadata={
                            "node_id": record["node_id"],
                            "article_name": record["name"],
                            "retrieval_level": "graph_rag_enriched"
                        }
                    ))
        except Exception as e:
            logger.error(f"递归回查条款失败: {e}")
        return docs

    def _paths_to_documents(self, paths: List[GraphPath]) -> List[Document]:
        """将路径转换为具体条款文档"""
        article_ids = set()
        for path in paths:
            for node in path.nodes:
                # 强化版 ID 提取
                nid = node.get("node_id") or node.get("nodeId")
                if not nid and "properties" in node:
                    nid = node["properties"].get("node_id") or node["properties"].get("nodeId")
                
                if not nid: continue

                # 判定规则：
                # 1. 直接是 Article
                # 2. ID 以 art_ 开头
                # 3. ID 格式为 PAR_第一百八十条-M1 等，需提取中间的条款名
                labels = node.get("labels", [])
                if not labels and "labels" in node: labels = node["labels"]
                
                if "Article" in labels or nid.startswith("art_"):
                    article_ids.add(nid)
                elif "_" in nid and ("PAR_" in nid or "FAC_" in nid or "REQ_" in nid):
                    # 尝试从 PAR_第一百八十条-xxx 中提取出“第一百八十条”作为回查线索
                    # 或者如果是 node_id 模式，找到它所属的 article 节点
                    try:
                        parts = nid.split('_')
                        if len(parts) >= 2:
                            term = parts[1].split('-')[0]
                            # 我们需要将这个 term 转回真正的 node_id (art_xxx)
                            # 最稳妥的方法是利用它去查一次
                            article_ids.add(f"art_{term}")
                    except: pass
        
        if not article_ids:
            logger.warning(f"未能从 {len(paths)} 条路径中解析出任何 Article ID。")
        else:
            logger.info(f"解析到待回查条款 ID: {article_ids}")
            
        return self._fetch_article_content(list(article_ids))

    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph) -> List[Document]:
        """将子图转换为具体条款文档"""
        article_ids = set()
        all_nodes = subgraph.central_nodes + subgraph.connected_nodes
        for node in all_nodes:
            nid = node.get("node_id") or node.get("nodeId")
            if not nid and "properties" in node:
                nid = node["properties"].get("node_id") or node["properties"].get("nodeId")
            
            if nid and nid.startswith("art_"):
                article_ids.add(nid)
        return self._fetch_article_content(list(article_ids))

    def close(self):
        if self.driver: self.driver.close()
