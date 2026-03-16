"""
矿井通风安全规程 - 混合检索模块（深度优化版）

继承 HybridRetrievalModule，覆盖所有烹饪领域相关的方法：
  1. _build_graph_index()        → 用 Article/Facility/Parameter 构建实体键值对
  2. extract_query_keywords()    → 通风安全领域的 LLM 关键词提取 Prompt
  3. _neo4j_entity_level_search()→ 查询 Article/Facility 节点，替代 Recipe 查询
  4. _neo4j_topic_level_search() → 按类型/内容检索 Requirement，替代 r.category 查询
  5. _generate_ventilation_relation_keys() → 为通风规程关系类型生成主题索引键
"""

import sys
import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from rag_modules.hybrid_retrieval import HybridRetrievalModule, RetrievalResult
from rag_modules.graph_indexing import EntityKeyValue

logger = logging.getLogger(__name__)


class VentilationHybridRetrieval(HybridRetrievalModule):
    """
    通风安全规程混合检索模块

    重写的核心方法：
        _build_graph_index()         - 用 articles/facilities/parameters 填充图索引
        extract_query_keywords()     - 通风领域关键词提取
        _neo4j_entity_level_search() - 查 Article/Facility 节点
        _neo4j_topic_level_search()  - 查 Requirement 节点按类型/内容
    """

    # ──────────────────────────────────────────────────────────
    # 1. 图索引构建：用通风领域实体替代烹饪实体
    # ──────────────────────────────────────────────────────────
    def _build_graph_index(self):
        """构建通风规程实体图索引"""
        if self.graph_indexed:
            return

        logger.info("开始构建通风规程图索引...")

        try:
            # ── 用 articles/facilities 填充实体 KV 存储 ──────
            articles   = getattr(self.data_module, 'articles', [])
            facilities = getattr(self.data_module, 'facilities', [])
            parameters = getattr(self.data_module, 'parameters', [])

            gi = self.graph_indexing  # GraphIndexingModule 实例

            # 条款 → 实体键值对
            for art in articles:
                art_id   = art.node_id
                art_name = art.name or f"条款_{art_id}"
                props    = getattr(art, 'properties', {})

                content_parts = [f"条款编号: {art_name}"]
                if props.get('title'):
                    content_parts.append(f"主题: {props['title']}")
                if props.get('content'):
                    content_parts.append(f"原文节选: {props['content'][:200]}")

                kv = EntityKeyValue(
                    entity_name=art_name,
                    index_keys=[art_name, props.get('title', art_name)],
                    value_content='\n'.join(content_parts),
                    entity_type="Article",
                    metadata={"node_id": art_id, "properties": props},
                )
                gi.entity_kv_store[art_id] = kv
                gi.key_to_entities[art_name].append(art_id)
                if props.get('title'):
                    gi.key_to_entities[props['title']].append(art_id)

            # 设施 → 实体键值对
            for fac in facilities:
                fac_id   = fac.node_id
                fac_name = fac.name or f"设施_{fac_id}"
                props    = getattr(fac, 'properties', {})

                content_parts = [f"通风设施: {fac_name}"]
                if props.get('description'):
                    content_parts.append(f"说明: {props['description']}")

                kv = EntityKeyValue(
                    entity_name=fac_name,
                    index_keys=[fac_name],
                    value_content='\n'.join(content_parts),
                    entity_type="Facility",
                    metadata={"node_id": fac_id, "properties": props},
                )
                gi.entity_kv_store[fac_id] = kv
                gi.key_to_entities[fac_name].append(fac_id)

            # 指标 → 实体键值对
            for param in parameters:
                param_id   = param.node_id
                param_name = param.name or f"指标_{param_id}"
                props      = getattr(param, 'properties', {})

                parts = [f"安全指标: {param_name}"]
                if props.get('value_min') is not None:
                    parts.append(f"下限: {props['value_min']}")
                if props.get('value_max') is not None:
                    parts.append(f"上限: {props['value_max']}")
                if props.get('unit'):
                    parts.append(f"单位: {props['unit']}")

                kv = EntityKeyValue(
                    entity_name=param_name,
                    index_keys=[param_name],
                    value_content='\n'.join(parts),
                    entity_type="Parameter",
                    metadata={"node_id": param_id, "properties": props},
                )
                gi.entity_kv_store[param_id] = kv
                gi.key_to_entities[param_name].append(param_id)

            # ── 提取图关系，构建关系 KV ──────────────────────
            relationships = self._extract_relationships_from_graph()

            # 覆盖关系主题键生成
            self._build_ventilation_relation_kvs(relationships)

            # 去重
            gi.deduplicate_entities_and_relations()

            self.graph_indexed = True
            stats = gi.get_statistics()
            logger.info(f"通风规程图索引构建完成: "
                        f"{stats['total_entities']} 个实体, "
                        f"{stats['total_relations']} 个关系")

        except Exception as e:
            logger.error(f"构建通风规程图索引失败: {e}", exc_info=True)

    def _build_ventilation_relation_kvs(self, relationships: List[Tuple[str, str, str]]):
        """为通风规程关系类型构建带主题键的关系 KV"""
        from rag_modules.graph_indexing import RelationKeyValue

        gi = self.graph_indexing
        for i, (src_id, rel_type, tgt_id) in enumerate(relationships):
            src_kv = gi.entity_kv_store.get(src_id)
            tgt_kv = gi.entity_kv_store.get(tgt_id)
            if not src_kv or not tgt_kv:
                continue

            # 通风领域主题键映射
            topic_keys = {
                "CONSTRAINS":        ["数值限制", "参数要求", "安全指标"],
                "APPLIES_TO":        ["适用地点", "适用场所"],
                "SPECIFIES":         ["安全要求", "规程条款"],
                "INVOLVES_FACILITY": ["通风设施", "机械设备", "安全设施"],
                "RELATED_TO":        ["关联条款", "相关规程"],
                "REFERENCES":        ["引用条款", "参考条款"],
            }.get(rel_type, [rel_type])

            index_keys = list(set([
                rel_type,
                src_kv.entity_name,
                tgt_kv.entity_name,
                *topic_keys,
            ]))

            rel_id = f"rel_{i}_{src_id}_{tgt_id}"
            rel_kv = RelationKeyValue(
                relation_id=rel_id,
                index_keys=index_keys,
                value_content=(
                    f"关系类型: {rel_type}\n"
                    f"源: {src_kv.entity_name} ({src_kv.entity_type})\n"
                    f"目标: {tgt_kv.entity_name} ({tgt_kv.entity_type})"
                ),
                relation_type=rel_type,
                source_entity=src_id,
                target_entity=tgt_id,
                metadata={
                    "source_name": src_kv.entity_name,
                    "target_name": tgt_kv.entity_name,
                },
            )
            gi.relation_kv_store[rel_id] = rel_kv
            for key in index_keys:
                gi.key_to_relations[key].append(rel_id)

    # ──────────────────────────────────────────────────────────
    # 2. 关键词提取：通风安全领域版本
    # ──────────────────────────────────────────────────────────
    def extract_query_keywords(self, query: str):
        """
        通风安全领域版本的双层关键词提取
          - 实体级：具体设施名、条款号、指标名、地点名
          - 主题级：安全主题（通风、瓦斯、风速等）
        """
        import json

        prompt = f"""
你是矿井通风安全专家，请分析以下查询并提取关键词，分为两个层次：

查询：{query}

提取规则：
1. 实体级关键词：具体的设施名称、条款编号、技术指标名、地点名称等有形实体
   - 例如：主要通风机、局部通风机、瓦斯浓度、回风巷、第一百七十七条
   - 数值型参数也算实体：0.25m/s、1%瓦斯浓度

2. 主题级关键词：抽象的安全主题、通风类别、适用场所类型等
   - 例如：通风、瓦斯管理、风速要求、掘进工作面、采煤工作面
   - 排除动作词：检查、符合、要求、规定等纯动词

示例：
查询："掘进工作面最低风速要求是多少"
{{
    "entity_keywords": ["最低风速", "掘进工作面"],
    "topic_keywords": ["风速要求", "通风参数", "掘进"]
}}

查询："主要通风机和备用通风机的切换规定"
{{
    "entity_keywords": ["主要通风机", "备用通风机"],
    "topic_keywords": ["通风机管理", "切换规定", "通风设施"]
}}

请严格按照JSON格式返回，不要包含多余文字：
{{
    "entity_keywords": ["实体1", "实体2", ...],
    "topic_keywords": ["主题1", "主题2", ...]
}}
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            result = json.loads(response.choices[0].message.content.strip())
            entity_kw = result.get("entity_keywords", [])
            topic_kw  = result.get("topic_keywords", [])
            logger.info(f"通风规程关键词提取 - 实体级: {entity_kw}, 主题级: {topic_kw}")
            return entity_kw, topic_kw

        except Exception as e:
            logger.error(f"关键词提取失败，降级处理: {e}")
            words = query.split()
            return words[:3], words[3:6] if len(words) > 3 else words

    # ──────────────────────────────────────────────────────────
    # 3. Neo4j 实体级补充检索：覆盖 Recipe fulltext 查询
    # ──────────────────────────────────────────────────────────
    def _neo4j_entity_level_search(
        self, keywords: List[str], limit: int
    ) -> List[RetrievalResult]:
        """通风规程版本：按关键词模糊搜索 Article / Facility 节点，并扩展引用的条款"""
        results = []
        try:
            with self.driver.session() as session:
                # ── 查条款（按名称或标题匹配），并拉取引用的条款内容 ──────────────
                article_query = """
                UNWIND $keywords AS kw
                MATCH (a:Article)
                WHERE a.name CONTAINS kw
                   OR a.title CONTAINS kw
                   OR a.content CONTAINS kw
                OPTIONAL MATCH (a)-[:CONSTRAINS]->(p:Parameter)
                WITH a, kw, collect(p.name)[0..3] AS params
                OPTIONAL MATCH (a)-[:REFERENCES]->(ref_a:Article)
                WITH a, kw, params, collect({name: ref_a.name, content: ref_a.content})[0..3] AS references
                RETURN a.node_id AS node_id, a.name AS name,
                       a.title AS title, a.content AS content,
                       params, references, kw AS matched_kw
                ORDER BY a.name
                LIMIT $limit
                """
                for record in session.run(article_query,
                                          {"keywords": keywords, "limit": limit}):
                    parts = [f"条款: {record['name']}"]
                    if record["title"]:
                        parts.append(f"主题: {record['title']}")
                    if record["content"]:
                        parts.append(f"原文节选: {record['content'][:300]}")
                    if record["params"]:
                        parts.append(f"关联指标: {'、'.join(p for p in record['params'] if p)}")
                    if record["references"]:
                        for ref in record["references"]:
                            if ref["name"] and ref["content"]:
                                parts.append(f"\n【补充引用】{ref['name']}: {ref['content'][:200]}...")

                    results.append(RetrievalResult(
                        content='\n'.join(parts),
                        node_id=record["node_id"] or "",
                        node_type="Article",
                        relevance_score=0.8,
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "title": record["title"],
                            "matched_keyword": record["matched_kw"],
                            "source": "neo4j_article",
                        },
                    ))

                # ── 查设施（按名称匹配）──────────────────────
                fac_query = """
                UNWIND $keywords AS kw
                MATCH (f:Facility)
                WHERE f.name CONTAINS kw
                OPTIONAL MATCH (a:Article)-[:SPECIFIES]->(:Requirement)
                              -[:INVOLVES_FACILITY]->(f)
                WITH f, kw, collect(DISTINCT a.name)[0..3] AS articles
                RETURN f.node_id AS node_id, f.name AS name,
                       articles, kw AS matched_kw
                ORDER BY f.name
                LIMIT $limit
                """
                for record in session.run(fac_query,
                                          {"keywords": keywords, "limit": limit}):
                    parts = [f"通风设施: {record['name']}"]
                    if record["articles"]:
                        parts.append(f"相关条款: {'、'.join(a for a in record['articles'] if a)}")

                    results.append(RetrievalResult(
                        content='\n'.join(parts),
                        node_id=record["node_id"] or "",
                        node_type="Facility",
                        relevance_score=0.7,
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "matched_keyword": record["matched_kw"],
                            "source": "neo4j_facility",
                        },
                    ))

        except Exception as e:
            logger.error(f"Neo4j 实体级检索失败: {e}")

        return results[:limit]

    # ──────────────────────────────────────────────────────────
    # 4. Neo4j 主题级补充检索：覆盖 Recipe.category 查询
    # ──────────────────────────────────────────────────────────
    def _neo4j_topic_level_search(
        self, keywords: List[str], limit: int
    ) -> List[RetrievalResult]:
        """通风规程版本：按安全要求类型 / 内容关键词搜索 Requirement + 关联 Article，并拉取引用"""
        results = []
        try:
            with self.driver.session() as session:
                query = """
                UNWIND $keywords AS kw
                MATCH (a:Article)-[:SPECIFIES]->(req:Requirement)
                WHERE req.name CONTAINS kw
                   OR req.content CONTAINS kw
                   OR a.title CONTAINS kw
                   OR a.content CONTAINS kw
                WITH a, req, kw
                OPTIONAL MATCH (req)-[:INVOLVES_FACILITY]->(f:Facility)
                WITH a, req, kw, collect(f.name)[0..3] AS facilities
                OPTIONAL MATCH (a)-[:REFERENCES]->(ref_a:Article)
                WITH a, req, kw, facilities, collect({name: ref_a.name, content: ref_a.content})[0..3] AS references
                RETURN a.node_id AS art_id, a.name AS art_name,
                       a.title AS art_title,
                       req.name AS req_type, req.content AS req_content,
                       facilities, references, kw AS matched_kw
                ORDER BY a.name
                LIMIT $limit
                """
                for record in session.run(query,
                                          {"keywords": keywords, "limit": limit}):
                    parts = [
                        f"条款: {record['art_name']}",
                        f"主题: {record['art_title'] or ''}",
                        f"要求类型: {record['req_type']}",
                    ]
                    if record["req_content"]:
                        parts.append(f"要求内容: {record['req_content'][:300]}")
                    if record["facilities"]:
                        parts.append(f"涉及设施: {'、'.join(f for f in record['facilities'] if f)}")
                    if record["references"]:
                        for ref in record["references"]:
                            if ref["name"] and ref["content"]:
                                parts.append(f"\n【补充引用】{ref['name']}: {ref['content'][:200]}...")

                    results.append(RetrievalResult(
                        content='\n'.join(parts),
                        node_id=record["art_id"] or "",
                        node_type="Article",
                        relevance_score=0.75,
                        retrieval_level="topic",
                        metadata={
                            "art_name": record["art_name"],
                            "art_title": record["art_title"],
                            "req_type": record["req_type"],
                            "matched_keyword": record["matched_kw"],
                            "source": "neo4j_requirement",
                        },
                    ))

        except Exception as e:
            logger.error(f"Neo4j 主题级检索失败: {e}")

        return results[:limit]

    # ──────────────────────────────────────────────────────────
    # 5. 获取邻居：用 node_id 属性（通风版）替代 nodeId
    # ──────────────────────────────────────────────────────────
    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """通风规程版本：用 node_id 而非 nodeId 属性查邻居"""
        try:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (n {node_id: $nid})-[]-(nb)
                    RETURN nb.name AS name
                    LIMIT $limit
                """, {"nid": node_id, "limit": max_neighbors})
                return [r["name"] for r in result if r["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []

    # ──────────────────────────────────────────────────────────
    # 6. 统一检索逻辑：解决字段名不匹配问题
    # ──────────────────────────────────────────────────────────
    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """通风版：合并实体级和主题级检索结果"""
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
            # 补全字段映射，防止上层代码报错
            meta["article_name"] = meta.get("name") or meta.get("art_name") or "未知条款"
            meta["node_id"] = r.node_id
            docs.append(Document(page_content=r.content, metadata=meta))
        return docs

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """向量检索并补全通风邻居信息"""
        try:
            vector_res = self.milvus_module.similarity_search(query, k=top_k*2)
            docs = []
            for res in vector_res:
                content = res.get("text", "")
                metadata = res.get("metadata", {})
                node_id = metadata.get("node_id")
                
                # 加载邻居信息
                if node_id:
                    nbs = self._get_node_neighbors(node_id)
                    if nbs:
                        content += f"\n相关参考: {', '.join(nbs)}"
                
                docs.append(Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "score": res.get("score", 0.0),
                        "search_type": "vector_enhanced"
                    }
                ))
            return docs[:top_k]
        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []

    def close(self):
        """关闭资源连接"""
        if self.driver:
            from neo4j import exceptions
            try:
                self.driver.close()
            except:
                pass
            logger.info("Neo4j连接已关闭")
