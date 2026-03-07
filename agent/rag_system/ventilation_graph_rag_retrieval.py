"""
矿井通风安全规程 - 图 RAG 检索模块（深度优化版）

继承 GraphRAGRetrieval，覆盖：
  1. understand_graph_query() - 将图 Schema 说明从烹饪改为通风规程
  2. _build_graph_index()     - 用 node_id 替代 nodeId 属性
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval, GraphQuery, QueryType

logger = logging.getLogger(__name__)


class VentilationGraphRAGRetrieval(GraphRAGRetrieval):
    """
    通风安全规程图 RAG 检索系统

    重写的核心方法：
        understand_graph_query()  - 通风规程 Schema 的查询意图理解
        _build_graph_index()      - 用 node_id 而非 nodeId 属性
    """

    # ──────────────────────────────────────────────────────────
    # 1. 理解查询意图：通风规程图 Schema
    # ──────────────────────────────────────────────────────────
    def understand_graph_query(self, query: str) -> GraphQuery:
        """
        将自然语言问题映射到通风规程知识图谱结构上
        使用通风领域正确的节点类型和关系类型
        """
        prompt = f"""
你是图数据库专家，请分析以下查询并映射到**矿井通风安全规程知识图谱**的结构上。

已知图谱 Schema：
- 节点类型：
  - Article（条款）：name（如"第一百七十七条"）、title（主题）、content（原文）
  - Parameter（数值指标）：name、value_min、value_max、unit
  - Requirement（安全要求）：name（要求类型）、content（要求内容）
  - Facility（通风设施）：name（如"主要通风机"、"局部通风机"、"风门"）
  - Location（适用地点）：name（如"掘进工作面"、"回风巷"、"采煤工作面"）

- 主要关系：
  - (Article)-[:CONSTRAINS]->(Parameter)     条款规定数值参数
  - (Parameter)-[:APPLIES_TO]->(Location)    参数适用于某地点
  - (Article)-[:SPECIFIES]->(Requirement)    条款规定安全要求
  - (Requirement)-[:INVOLVES_FACILITY]->(Facility) 要求涉及某设施
  - (Article)-[:RELATED_TO]-(Article)        条款间关联
  - (Article)-[:REFERENCES]->(Article)       条款引用其他条款

查询：{query}

请识别：
1. query_type（查询类型）：
   - entity_relation: 询问特定实体的直接信息（如：局部通风机的要求？）
   - multi_hop: 需要多跳推理（如：主通风机失电后的备用通风机切换程序）
   - subgraph: 需要完整子图（如：掘进工作面相关的所有通风要求）
   - path_finding: 路径查找（如：从设施到条款的关联路径）
   - clustering: 聚类相似（如：有哪些条款都涉及主要通风机）

2. source_entities（源实体，必须是图中可能存在的节点名称）：
   - 优先选择：具体设施名、条款编号、地点名、指标名
   - 不要放抽象概念，如"违规"、"安全"等

3. target_entities：只在需要限制路径终点时填写，否则为 []

4. relation_types：本次推理优先考虑的关系类型列表
   例如：["CONSTRAINS", "APPLIES_TO"] 或 ["SPECIFIES", "INVOLVES_FACILITY"]

5. max_depth：建议的图遍历深度（1-3 整数）

6. constraints（属性级约束，用于后处理过滤）：
   例如：{{"location": ["掘进工作面"], "facility_type": ["主要通风机"]}}

示例1：
查询："局部通风机的安装要求有哪些？"
{{
  "query_type": "subgraph",
  "source_entities": ["局部通风机"],
  "target_entities": [],
  "relation_types": ["INVOLVES_FACILITY", "SPECIFIES"],
  "max_depth": 2,
  "constraints": {{}}
}}

示例2：
查询："掘进工作面风速不达标时应该怎么处理？"
{{
  "query_type": "multi_hop",
  "source_entities": ["掘进工作面", "最低风速"],
  "target_entities": [],
  "relation_types": ["APPLIES_TO", "CONSTRAINS", "SPECIFIES"],
  "max_depth": 3,
  "constraints": {{"location": ["掘进工作面"]}}
}}

请严格返回合法 JSON 对象，不包含多余说明：
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()
            # 去除可能的 markdown 代码块包裹
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)

            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50,
            )

        except Exception as e:
            logger.error(f"查询意图理解失败（降级为 subgraph）: {e}")
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2,
            )

    # ──────────────────────────────────────────────────────────
    # 2. 图索引构建：用 node_id 替代 nodeId
    # ──────────────────────────────────────────────────────────
    def _build_graph_index(self):
        """通风规程版本：构建图结构索引（entity_cache / relation_cache）"""
        logger.info("构建通风规程图结构索引...")
        try:
            with self.driver.session() as session:
                # 用 node_id 属性（通风版）替代原来的 nodeId
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
                        "labels":   record["node_labels"],
                        "name":     record["name"],
                        "category": record["category"],
                        "degree":   record["degree"],
                    }

                # 关系类型统计（不变）
                rel_query = """
                MATCH ()-[r]->()
                RETURN type(r) AS rel_type, count(r) AS frequency
                ORDER BY frequency DESC
                """
                result = session.run(rel_query)
                for record in result:
                    self.relation_cache[record["rel_type"]] = record["frequency"]

                logger.info(f"通风规程图索引完成: "
                            f"{len(self.entity_cache)} 个实体, "
                            f"{len(self.relation_cache)} 个关系类型")

        except Exception as e:
            logger.error(f"构建图结构索引失败: {e}")

    # ──────────────────────────────────────────────────────────
    # 3. 核心修复：从图路径/子图回查完整条款内容
    # ──────────────────────────────────────────────────────────
    def _fetch_article_content(self, node_ids: list, max_articles: int = 5) -> list:
        """
        按 Article node_id 列表从 Neo4j 回查完整条款内容
        （包含数值指标、安全要求、涉及设施）
        返回 list[Document]
        """
        from langchain_core.documents import Document

        if not self.driver or not node_ids:
            return []

        docs = []
        try:
            with self.driver.session() as session:
                art_query = """
                UNWIND $ids AS nid
                MATCH (a:Article {node_id: nid})
                OPTIONAL MATCH (a)-[:CONSTRAINS]->(p:Parameter)
                OPTIONAL MATCH (p)-[:APPLIES_TO]->(loc:Location)
                OPTIONAL MATCH (a)-[:SPECIFIES]->(req:Requirement)
                OPTIONAL MATCH (req)-[:INVOLVES_FACILITY]->(fac:Facility)
                OPTIONAL MATCH (a)-[:REFERENCES]->(ref_a:Article)
                WITH a,
                     collect(DISTINCT {
                         name: p.name,
                         value_min: p.value_min,
                         value_max: p.value_max,
                         unit: p.unit,
                         location: loc.name
                     }) AS params,
                     collect(DISTINCT {
                         type: req.name,
                         content: req.content,
                         facility: fac.name
                     }) AS reqs,
                     collect(DISTINCT {
                         name: ref_a.name,
                         content: ref_a.content
                     })[0..3] AS references
                RETURN a.node_id AS node_id,
                       a.name    AS name,
                       a.title   AS title,
                       a.content AS content,
                       params, reqs, references
                """
                result = session.run(art_query, {"ids": list(node_ids)[:max_articles]})

                for record in result:
                    parts = []

                    # 条款头部
                    art_name  = record["name"] or ""
                    art_title = record["title"] or ""
                    parts.append(f"# {art_name}")
                    if art_title:
                        parts.append(f"## 主题：{art_title}")

                    # 条款原文
                    if record.get("content"):
                        parts.append(f"\n## 条款原文\n{record['content']}")

                    # 数值指标
                    params = [p for p in record["params"]
                              if p.get("name") and p["name"] is not None]
                    if params:
                        parts.append("\n## 数值指标")
                        for p in params:
                            line = f"- 【指标】{p['name']}"
                            if p.get("value_min") is not None:
                                line += f"：≥{p['value_min']}"
                            if p.get("value_max") is not None:
                                line += f"（上限 {p['value_max']}）"
                            if p.get("unit"):
                                line += f" {p['unit']}"
                            if p.get("location"):
                                line += f"（适用：{p['location']}）"
                            parts.append(line)

                    # 安全要求
                    reqs = [r for r in record["reqs"]
                            if r.get("type") and r["type"] is not None]
                    if reqs:
                        parts.append("\n## 安全要求")
                        seen = set()
                        for r in reqs:
                            req_key = r["type"]
                            if req_key in seen:
                                continue
                            seen.add(req_key)
                            line = f"- 【要求】{r['type']}"
                            if r.get("content"):
                                line += f"：{r['content'][:150]}"
                            if r.get("facility"):
                                line += f"（涉及设施：{r['facility']}）"
                            parts.append(line)

                    # 补充引用
                    if record.get("references"):
                        for ref in record["references"]:
                            if ref.get("name") and ref.get("content"):
                                parts.append(f"\n## 【补充引用】{ref['name']}\n{ref['content'][:200]}...")

                    full_text = "\n".join(parts)
                    docs.append(Document(
                        page_content=full_text,
                        metadata={
                            "node_id":   record["node_id"] or "",
                            "node_type": "Article",
                            "name":      art_name,
                            "title":     art_title,
                            "search_type": "graph_enriched",
                            "relevance_score": 0.9,
                        },
                    ))

        except Exception as e:
            logger.error(f"回查条款内容失败: {e}")

        return docs

    def _paths_to_documents(self, paths, query: str) -> list:
        """
        覆盖父类：从路径节点中收集 Article node_id，
        回查 Neo4j 取完整条款内容，而不只是路径字符串描述
        """
        from langchain_core.documents import Document

        # 收集路径里所有 Article 节点的 node_id
        article_ids = []
        seen = set()
        for path in paths:
            for node in path.nodes:
                labels = node.get("labels", [])
                nid    = node.get("id") or node.get("properties", {}).get("node_id")
                # 优先用 properties 里的 node_id
                if not nid:
                    nid = node.get("properties", {}).get("node_id")
                if nid and nid not in seen:
                    seen.add(nid)
                    if "Article" in labels:
                        article_ids.append(nid)

        # 如果没有直接找到 Article，尝试查询相连的 Article 节点
        if not article_ids and seen:
            logger.info(f"路径中未直接找到 Article 节点，尝试根据收集到的 {len(seen)} 个非条款节点查询相关联的条款")
            try:
                with self.driver.session() as session:
                    res = session.run("""
                        UNWIND $nids AS nid
                        MATCH (n {node_id: nid})
                        MATCH (n)-[*]-(a:Article)
                        RETURN DISTINCT a.node_id AS art_id LIMIT 5
                    """, {"nids": list(seen)})
                    article_ids = [r["art_id"] for r in res]
            except Exception as e:
                logger.error(f"拓展查找相连 Article 失败: {e}")

        if article_ids:
            logger.info(f"图路径关联中找到 {len(article_ids)} 个 Article 节点，回查完整内容")
            return self._fetch_article_content(article_ids)

        # 降级：返回路径字符串描述（原父类行为）
        logger.warning("路径及其周边均未找到 Article 节点，返回结构描述")
        return super()._paths_to_documents(paths, query)

    def _subgraph_to_documents(self, subgraph, reasoning_chains: list, query: str) -> list:
        """
        覆盖父类：从子图中收集 Article node_id，
        回查 Neo4j 取完整条款内容
        """
        # 收集子图里所有 Article 节点的 node_id
        article_ids = []
        seen = set()

        all_nodes = list(subgraph.central_nodes) + list(subgraph.connected_nodes)
        for node in all_nodes:
            labels = node.get("labels", [])
            # node_id 可能存储在不同字段
            nid = (node.get("node_id")
                   or node.get("nodeId")
                   or node.get("properties", {}).get("node_id"))
            is_article = ("Article" in labels
                          or node.get("node_id", "").startswith("art_"))
            if nid and is_article and nid not in seen:
                seen.add(nid)
                article_ids.append(nid)
            elif nid and nid not in seen:
                seen.add(nid)

        if not article_ids and seen:
            logger.info(f"子图未直接包含 Article 节点，尝试根据 {len(seen)} 个关联节点回查条款")
            try:
                with self.driver.session() as session:
                    res = session.run("""
                        UNWIND $nids AS nid
                        MATCH (n {node_id: nid})
                        MATCH (n)-[*]-(a:Article)
                        RETURN DISTINCT a.node_id AS art_id LIMIT 5
                    """, {"nids": list(seen)})
                    article_ids = [r["art_id"] for r in res]
            except Exception as e:
                logger.error(f"拓展查找相连 Article 失败: {e}")

        if article_ids:
            logger.info(f"子图中找到/关联了 {len(article_ids)} 个 Article 节点，回查完整内容")
            return self._fetch_article_content(article_ids)

        # 降级：返回子图摘要描述（原父类行为）
        logger.warning("子图及其周边均未找到 Article 节点，返回摘要描述")
        return super()._subgraph_to_documents(subgraph, reasoning_chains, query)
