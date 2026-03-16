"""
矿井通风安全规程 - 图数据准备模块

数据加载和文档构建逻辑
适配通风规程知识图谱的节点模式（Article / Parameter / Requirement / Facility / Location）。
"""

import sys
import os
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_core.documents import Document



logger = logging.getLogger(__name__)

@dataclass
# 装饰器：@dataclass 可以自动为类添加 __init__、__repr__ 等方法
class GraphNode:
    """图节点数据结构"""
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]

class VentilationDataPreparationModule:
    """
    通风安全规程数据准备模块
    覆盖以下方法：
        load_graph_data()         → 加载 Article / Parameter / Requirement / Facility / Location
        build_article_documents() → 构建条款文档（一条款一文档）
        get_statistics()          → 统计信息
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        # 复用父类连接逻辑，但修改为通风领域的字段名
        # super().__init__(uri, user, password, database)
       
        """
        初始化图数据库连接
        
        Args:
            uri: Neo4j连接URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None

        # 通风领域专用数据容器
        self.articles: List[GraphNode]    = []  # 存放所有"条款"节点
        self.parameters: List[GraphNode]  = []  # 存放所有"安全指标"节点
        self.requirements: List[GraphNode] = [] # 存放所有"安全要求"节点
        self.facilities: List[GraphNode]  = []  # 存放所有"通风设施"节点
        self.locations: List[GraphNode]   = []  # 存放所有"适用地点"节点


        # 兼容性存根：让 hybrid_retrieval._build_graph_index() 不报 AttributeError
        # （父类 cooking 版本用这几个字段，通风版本里直接清空即可）
        self.recipes        = []
        self.ingredients    = []
        self.cooking_steps  = []

        self.documents: List[Document] = []
        self.chunks: List[Document] = []


        self._connect()
        
    def _connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                database=self.database
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri}")
            
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")
                    
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")
    # ──────────────────────────────────────────────────────────
    # 1. 加载图数据
    # ──────────────────────────────────────────────────────────
    def load_graph_data(self) -> Dict[str, Any]:
        """从 Neo4j 加载通风规程图数据"""
        logger.info("正在从 Neo4j 加载通风规程图数据...")

        with self.driver.session() as session:
        # 它创建了一个会话（Session），with 语句确保查询完成后，连接会被正确关闭，不会占用资源。

            # ── 条款 ──────────────────────────────────────────
            result = session.run("""
                MATCH (a:Article)
                RETURN a.node_id AS node_id, labels(a) AS labels,
                       a.name AS name, properties(a) AS props
                ORDER BY a.node_id
            """)
            self.articles = [
                GraphNode(
                    node_id=r["node_id"], labels=r["labels"],
                    name=r["name"], properties=dict(r["props"])
                ) for r in result
            ]
            logger.info(f"加载了 {len(self.articles)} 个条款节点")

            # ── 安全指标 ─────────────────────────────────────
            result = session.run("""
                MATCH (p:Parameter)
                RETURN p.node_id AS node_id, labels(p) AS labels,
                       p.name AS name, properties(p) AS props
                ORDER BY p.node_id
            """)
            self.parameters = [
                GraphNode(
                    node_id=r["node_id"], labels=r["labels"],
                    name=r["name"], properties=dict(r["props"])
                ) for r in result
            ]
            logger.info(f"加载了 {len(self.parameters)} 个指标节点")

            # ── 安全要求 ─────────────────────────────────────
            result = session.run("""
                MATCH (req:Requirement)
                RETURN req.node_id AS node_id, labels(req) AS labels,
                       req.name AS name, properties(req) AS props
                ORDER BY req.node_id
            """)
            self.requirements = [
                GraphNode(
                    node_id=r["node_id"], labels=r["labels"],
                    name=r["name"], properties=dict(r["props"])
                ) for r in result
            ]
            logger.info(f"加载了 {len(self.requirements)} 个要求节点")

            # ── 通风设施 ─────────────────────────────────────
            result = session.run("""
                MATCH (f:Facility)
                RETURN f.node_id AS node_id, labels(f) AS labels,
                       f.name AS name, properties(f) AS props
                ORDER BY f.node_id
            """)
            self.facilities = [
                GraphNode(
                    node_id=r["node_id"], labels=r["labels"],
                    name=r["name"], properties=dict(r["props"])
                ) for r in result
            ]
            logger.info(f"加载了 {len(self.facilities)} 个设施节点")

            # ── 适用地点 ─────────────────────────────────────
            result = session.run("""
                MATCH (l:Location)
                RETURN l.node_id AS node_id, labels(l) AS labels,
                       l.name AS name, properties(l) AS props
                ORDER BY l.node_id
            """)
            self.locations = [
                GraphNode(
                    node_id=r["node_id"], labels=r["labels"],
                    name=r["name"], properties=dict(r["props"])
                ) for r in result
            ]
            logger.info(f"加载了 {len(self.locations)} 个地点节点")

        return {
            'articles': len(self.articles),
            'parameters': len(self.parameters),
            'requirements': len(self.requirements),
            'facilities': len(self.facilities),
            'locations': len(self.locations),
        }

    # ──────────────────────────────────────────────────────────
    # 2. 构建条款文档（核心方法）
    # ──────────────────────────────────────────────────────────
    def build_article_documents(self) -> List[Document]:
        """
        为每条通风规程条款构建一个 Document，内容包括：
            - 条款原文
            - 数值指标（名称 / 上下限 / 单位 / 适用地点）
            - 安全要求（类型 / 内容 / 涉及设施）
            - 相关条款引用
        """
        logger.info("正在构建通风规程条款文档...")
        documents = []

        with self.driver.session() as session:
            for article in self.articles:
                try:
                    art_id = article.node_id
                    art_name = article.name
                    props = article.properties

                    # ── 数值指标 ─────────────────────────────
                    params_result = session.run("""
                        MATCH (a:Article {node_id: $id})-[:CONSTRAINS]->(p:Parameter)
                        OPTIONAL MATCH (p)-[:APPLIES_TO]->(l:Location)
                        RETURN p.name AS name,
                               p.value_min AS v_min, p.value_max AS v_max,
                               p.unit AS unit, l.name AS location
                        ORDER BY p.node_id
                    """, id=art_id)

                    param_lines = []
                    for r in params_result:
                        line = f"- 【指标】{r['name']}"
                        if r['v_min'] is not None and r['v_min'] != '':
                            # 下限
                            line += f"：≥{r['v_min']}"
                        if r['v_max'] is not None and r['v_max'] != '':
                            # 上限
                            line += f"，≤{r['v_max']}"
                        if r['unit']:
                            # 单位
                            line += f" {r['unit']}"
                        if r['location']:
                            # 地点
                            line += f"（{r['location']}）"
                        param_lines.append(line)

                    # ── 安全要求 ─────────────────────────────
                    reqs_result = session.run("""
                        MATCH (a:Article {node_id: $id})-[:SPECIFIES]->(req:Requirement)
                        WITH req ORDER BY req.node_id
                        OPTIONAL MATCH (req)-[:INVOLVES_FACILITY]->(f:Facility)
                        RETURN req.name AS req_type,
                               req.content AS content,
                               collect(f.name) AS facilities
                    """, id=art_id)

                    req_lines = []
                    seen_contents = set()
                    for r in reqs_result:
                        content = r['content'] or ''
                        # 去重
                        if content in seen_contents:
                            continue
                        seen_contents.add(content)
                        line = f"- 【{r['req_type']}】{content}"
                        facs = [f for f in r['facilities'] if f]
                        if facs:
                            line += f"\n  涉及设施：{'、'.join(facs)}"
                        req_lines.append(line)

                    # ── 相关条款 ─────────────────────────────
                    refs_result = session.run("""
                        MATCH (a:Article {node_id: $id})-[:REFERENCES]->(b:Article)
                        RETURN b.name AS ref_name
                        UNION
                        MATCH (a:Article {node_id: $id})-[:RELATED_TO]-(b:Article)
                        RETURN b.name AS ref_name
                        ORDER BY ref_name
                    """, id=art_id)
                    related = [r['ref_name'] for r in refs_result]

                    # ── 构建文档正文 ─────────────────────────
                    parts = [
                        f"# {art_name}",
                        f"## 条款主题\n{props.get('title', '')}",
                        f"## 条款原文\n{props.get('content', '')}",
                    ]

                    if param_lines:
                        parts.append("## 数值指标\n" + "\n".join(param_lines))

                    if req_lines:
                        parts.append("## 安全要求\n" + "\n".join(req_lines))

                    if related:
                        parts.append("## 关联条款\n" + "、".join(related))

                    full_content = "\n\n".join(parts)

                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "node_id":          art_id,
                            "article_name":     art_name,
                            "article_title":    props.get("title", ""),
                            "node_type":        "Article",
                            "param_count":      len(param_lines),
                            "req_count":        len(req_lines),
                            "related_articles": related,
                            "doc_type":         "article",
                            "content_length":   len(full_content),
                        }
                    )
                    documents.append(doc)

                except Exception as e:
                    logger.warning(f"构建条款文档失败 {art_name}: {e}")
                    continue

        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个条款文档")
        return documents

    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        对文档进行分块处理
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块后的文档列表
        """
        logger.info(f"正在进行文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")
        
        if not self.documents:
            raise ValueError("请先构建文档")
        
        chunks = []
        chunk_id = 0
        
        for doc in self.documents:
            content = doc.page_content
            
            # 简单的按长度分块
            if len(content) <= chunk_size:
                # 内容较短，不需要分块
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                        "parent_id": doc.metadata["node_id"],
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "doc_type": "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # 按章节分块（基于标题）
                sections = content.split('\n## ')
                if len(sections) <= 1:
                    # 没有二级标题，按长度强制分块
                    total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1
                    
                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end = min(start + chunk_size, len(content))
                        
                        chunk_content = content[start:end]
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                else:
                    # 按章节分块
                    total_chunks = len(sections)
                    for i, section in enumerate(sections):
                        if i == 0:
                            # 第一个部分包含标题
                            chunk_content = section
                        else:
                            # 其他部分添加章节标题
                            chunk_content = f"## {section}"
                        
                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk",
                                "section_title": section.split('\n')[0] if i > 0 else "主标题"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1
        
        self.chunks = chunks
        logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks
    # ──────────────────────────────────────────────────────────
    # 3. 统计信息（覆盖父类，使用通风领域字段）
    # ──────────────────────────────────────────────────────────
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_articles':     len(self.articles),
            'total_parameters':   len(self.parameters),
            'total_requirements': len(self.requirements),
            'total_facilities':   len(self.facilities),
            'total_locations':    len(self.locations),
            'total_documents':    len(self.documents),
            'total_chunks':       len(self.chunks),
        }
        if self.documents:
            stats['avg_content_length'] = (
                sum(d.metadata.get('content_length', 0) for d in self.documents)
                / len(self.documents)
            )
        return stats

    def __del__(self):
        """析构函数，确保关闭连接"""
        self.close() 