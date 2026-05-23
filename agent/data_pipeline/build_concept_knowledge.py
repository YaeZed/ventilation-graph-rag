"""
Build ventilation concept knowledge layer from existing Neo4j data + LLM enrichment.

Steps:
  1. Extract concept names from Neo4j nodes (Parameter, Requirement, Facility)
     and LLM scan of Article.content
  2. Generate structured definitions via Qwen-Plus (definition, visual_clues,
     identification_features, etc.)
  3. Store Concept nodes in Neo4j with RELATES_TO edges to Article
  4. Vectorize concepts and insert into Milvus

Usage:
    python build_concept_knowledge.py              # build only if no Concept nodes exist
    python build_concept_knowledge.py --force      # delete existing Concept nodes and rebuild
"""

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from connection_manager import ConnectionManager

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("concept_builder")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY") or ""
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

CONCEPT_PROMPT_TEMPLATE = """你是一位煤矿通风安全专家，正在编写《矿井通风安全概念词典》。

请为以下通风专业概念生成结构化的解释条目：

概念名称：{concept_name}
相关规程条文：{article_context}

请返回严格 JSON（不要 Markdown 标记）：
{{
  "name": "概念名称",
  "aliases": ["别名1", "别名2"],
  "definition": "用一段话精确定义该概念（2-4句）",
  "identification_features": "从现场检查角度，如何辨别该现象或设施状态（列出3-5条可操作的判别标准）",
  "visual_clues": "从图片/视频中可以观察到的视觉特征（颜色、位置、空间关系、设备排列等），为图像识别模型提供线索",
  "typical_scenarios": "该概念通常出现在哪些作业场景中",
  "hazard_significance": "为什么这是安全隐患，违反了什么原则，可能导致什么后果",
  "related_regulation_articles": ["第一百三十三条"]
}}

注意：
- visual_clues 要具体到图片能看到的特征，不要说"查阅规程"之类的抽象话
- identification_features 要是巡检员可以直接对照检查的清单式标准
- 如果某个字段不确定，填 null 不要编造"""

ARTICLE_SCAN_PROMPT = """你是一位煤矿通风安全专家。请从以下《煤矿安全规程》通风章节条文中，提取所有专业概念和术语。

条文内容：
{article_text}

请返回严格 JSON：
{{
  "concepts": [
    {{
      "name": "概念名称",
      "context": "条文中的相关描述（直接引用原文）",
      "article_name": "条款名"
    }}
  ]
}}

注意：
- 只提取通风安全领域的专业术语（如"循环风"、"串联通风"、"风电闭锁"等）
- 不要提取通用词汇
- 每个概念标注出处条款"""


@dataclass
class ConceptEntry:
    name: str
    aliases: List[str]
    definition: str
    identification_features: str
    visual_clues: str
    typical_scenarios: str
    hazard_significance: str
    related_regulation_articles: List[str]


class ConceptKnowledgeBuilder:
    def __init__(self, force: bool = False):
        self.force = force
        self.llm = OpenAI(
            api_key=DASHSCOPE_API_KEY or "sk-dummy",
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.cm = ConnectionManager.get_instance()
        self.driver = self.cm.get_neo4j_driver(verify=True)
        self.embedder = SentenceTransformer(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "bge-small-zh-v1.5")
        )

    def run(self) -> int:
        logger.info("=== 通风概念知识层构建 ===")

        if self.force:
            self._clear_existing_concepts()

        existing = self._count_concepts()
        if existing > 0:
            entries = self._load_concepts_from_neo4j()
            logger.info("Concept 节点已存在 %s 个，跳过 LLM 生成，刷新 Milvus 向量集合", existing)
            self._store_to_milvus(entries)
            return existing

        concept_names = self._extract_concept_names()
        logger.info("提取到 %s 个候选概念", len(concept_names))

        entries = self._generate_definitions(concept_names)
        logger.info("成功生成 %s 个概念定义", len(entries))

        self._store_to_neo4j(entries)
        self._store_to_milvus(entries)
        logger.info("概念知识层构建完成")
        return len(entries)

    def _clear_existing_concepts(self):
        with self.driver.session() as session:
            result = session.run("MATCH (c:Concept) DETACH DELETE c RETURN count(c) AS deleted")
            record = result.single()
            if record and record["deleted"]:
                logger.info("已删除 %s 个旧 Concept 节点", record["deleted"])

    def _count_concepts(self) -> int:
        with self.driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN count(c) AS cnt")
            record = result.single()
            return record["cnt"] if record else 0

    def _load_concepts_from_neo4j(self) -> List[ConceptEntry]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (c:Concept)
                RETURN c.name AS name,
                       c.aliases AS aliases,
                       c.definition AS definition,
                       c.identification_features AS identification_features,
                       c.visual_clues AS visual_clues,
                       c.typical_scenarios AS typical_scenarios,
                       c.hazard_significance AS hazard_significance,
                       c.related_regulation_articles AS related_regulation_articles
                ORDER BY c.name
                """
            )
            entries = []
            for record in result:
                entries.append(ConceptEntry(
                    name=self._as_text(record["name"]),
                    aliases=self._as_list(record["aliases"]),
                    definition=self._as_text(record["definition"]),
                    identification_features=self._as_text(record["identification_features"]),
                    visual_clues=self._as_text(record["visual_clues"]),
                    typical_scenarios=self._as_text(record["typical_scenarios"]),
                    hazard_significance=self._as_text(record["hazard_significance"]),
                    related_regulation_articles=self._as_list(record["related_regulation_articles"]),
                ))
            return entries

    def _extract_concept_names(self) -> Dict[str, Dict[str, Any]]:
        """Collect concept names from existing nodes and LLM article scan."""
        collected: Dict[str, Dict[str, Any]] = {}

        with self.driver.session() as session:
            for label, field in [("Parameter", "name"), ("Requirement", "name"), ("Facility", "name")]:
                result = session.run(
                    f"MATCH (n:{label}) WHERE n.{field} IS NOT NULL RETURN n.{field} AS name"
                )
                for record in result:
                    name = str(record["name"]).strip()
                    if name and len(name) <= 20:
                        collected.setdefault(name, {"sources": []})["sources"].append(label)

            articles_result = session.run(
                "MATCH (a:Article) RETURN a.name AS name, a.content AS content ORDER BY a.node_id"
            )
            articles = [{"name": r["name"], "content": r["content"] or ""} for r in articles_result]

        for i in range(0, len(articles), 3):
            batch = articles[i : i + 3]
            batch_text = "\n\n".join(
                f"【{a['name']}】\n{a['content'][:1200]}" for a in batch
            )
            try:
                response = self.llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": ARTICLE_SCAN_PROMPT.format(article_text=batch_text)}],
                    temperature=0.1,
                )
                data = self._parse_json(response.choices[0].message.content)
                for item in data.get("concepts", []):
                    name = str(item.get("name", "")).strip()
                    if name and len(name) <= 20:
                        entry = collected.setdefault(name, {"sources": [], "articles": []})
                        entry.setdefault("articles", []).append(
                            {"article_name": item.get("article_name"), "context": item.get("context")}
                        )
            except Exception as exc:
                logger.warning("条文扫描失败: %s", exc)
            time.sleep(0.3)

        logger.info("从 Neo4j 节点提取 %s 个概念名 + LLM 扫描补充", sum(1 for v in collected.values() if v.get("sources")))
        return collected

    def _generate_definitions(self, concept_names: Dict[str, Dict[str, Any]]) -> List[ConceptEntry]:
        entries: List[ConceptEntry] = []
        names = list(concept_names.keys())

        for name in names:
            info = concept_names[name]
            article_context = ""
            for art in info.get("articles", [])[:2]:
                article_context += f"\n【{art.get('article_name', '')}】{art.get('context', '')}"

            try:
                response = self.llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{
                        "role": "user",
                        "content": CONCEPT_PROMPT_TEMPLATE.format(
                            concept_name=name,
                            article_context=article_context or "暂无直接条文引用",
                        ),
                    }],
                    temperature=0.2,
                )
                data = self._parse_json(response.choices[0].message.content)
                entries.append(ConceptEntry(
                    name=self._as_text(data.get("name", name)),
                    aliases=self._as_list(data.get("aliases")),
                    definition=self._as_text(data.get("definition")),
                    identification_features=self._as_text(data.get("identification_features")),
                    visual_clues=self._as_text(data.get("visual_clues")),
                    typical_scenarios=self._as_text(data.get("typical_scenarios")),
                    hazard_significance=self._as_text(data.get("hazard_significance")),
                    related_regulation_articles=self._as_list(data.get("related_regulation_articles")),
                ))
                logger.info("  ✓ %s", name)
            except Exception as exc:
                logger.warning("  ✗ %s: %s", name, exc)
            time.sleep(0.2)

        return entries

    def _store_to_neo4j(self, entries: List[ConceptEntry]):
        with self.driver.session() as session:
            for entry in entries:
                try:
                    session.run(
                        """
                        MERGE (c:Concept {name: $name})
                        SET c.aliases = $aliases,
                            c.definition = $definition,
                            c.identification_features = $identification_features,
                            c.visual_clues = $visual_clues,
                            c.typical_scenarios = $typical_scenarios,
                            c.hazard_significance = $hazard_significance,
                            c.related_regulation_articles = $related_articles
                        """,
                        name=entry.name,
                        aliases=entry.aliases,
                        definition=entry.definition,
                        identification_features=entry.identification_features,
                        visual_clues=entry.visual_clues,
                        typical_scenarios=entry.typical_scenarios,
                        hazard_significance=entry.hazard_significance,
                        related_articles=entry.related_regulation_articles,
                    )
                    for article_name in entry.related_regulation_articles:
                        session.run(
                            """
                            MATCH (c:Concept {name: $concept_name})
                            MATCH (a:Article)
                            WHERE a.name = $article_name OR a.name CONTAINS $article_name
                            MERGE (c)-[:RELATES_TO]->(a)
                            """,
                            concept_name=entry.name,
                            article_name=article_name,
                        )
                except Exception as exc:
                    logger.warning("Neo4j 存储失败 %s: %s", entry.name, exc)

        logger.info("已存储 %s 个 Concept 节点到 Neo4j", len(entries))

    def _store_to_milvus(self, entries: List[ConceptEntry]):
        collection_name = "ventilation_concepts"
        dim = 512

        client = self.cm.get_milvus_client()

        has_collection = False
        try:
            collections = getattr(client, "list_collections", lambda: [])()
            has_collection = collection_name in collections
        except Exception:
            pass

        if has_collection:
            logger.info("Milvus collection %s 已存在，删除后重建以避免重复向量", collection_name)
            client.drop_collection(collection_name)
            has_collection = False

        if not has_collection:
            from pymilvus import DataType, CollectionSchema, FieldSchema

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="definition", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="visual_clues", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="identification_features", dtype=DataType.VARCHAR, max_length=2048),
            ]
            schema = CollectionSchema(fields, description="Ventilation safety concept dictionary")
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )
            client.load_collection(collection_name)

        data = []
        for i, entry in enumerate(entries):
            text_for_embedding = " ".join([
                entry.name,
                entry.definition,
                entry.visual_clues,
                entry.identification_features,
                " ".join(entry.aliases),
            ])
            vector = self.embedder.encode(text_for_embedding, normalize_embeddings=True).tolist()
            data.append({
                "id": f"concept_{i:04d}",
                "vector": vector,
                "name": entry.name,
                "definition": entry.definition[:4000],
                "visual_clues": entry.visual_clues[:2000],
                "identification_features": entry.identification_features[:2000],
            })

        if data:
            client.insert(collection_name=collection_name, data=data)
            try:
                client.flush(collection_name)
            except Exception:
                pass
            client.load_collection(collection_name)
            logger.info("已向量化并存入 Milvus (%s 个概念)", len(data))

    def _as_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple, set)):
            return "\n".join(self._as_text(item) for item in value if item is not None)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _as_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, (list, tuple, set)):
            return [self._as_text(item) for item in value if self._as_text(item)]
        return [self._as_text(value)]

    def _parse_json(self, content: str) -> Dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build ventilation concept knowledge layer")
    parser.add_argument("--force", action="store_true", help="Delete existing Concept nodes and rebuild")
    args = parser.parse_args()

    builder = ConceptKnowledgeBuilder(force=args.force)
    count = builder.run()
    print(f"\n概念知识层构建完成: {count} 个概念")


if __name__ == "__main__":
    main()
