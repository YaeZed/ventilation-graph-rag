"""
Cypher template retrieval for structured ventilation scene fields.

This module is the deterministic bridge between future Qwen-VL structured
outputs and the Neo4j knowledge graph. It loads predefined scene schemas and
parameterized Cypher files from ``cypher_templates/``.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CypherScene:
    id: str
    name: str
    query_file: str
    aliases: List[str]
    keywords: List[str]
    schema: Dict[str, Dict[str, Any]]
    query: str


@dataclass
class TemplateMatch:
    scene: CypherScene
    score: int
    reasons: List[str]


class VentilationCypherTemplateEngine:
    """Load, match, and execute parameterized ventilation Cypher templates."""

    def __init__(self, template_dir: Optional[str] = None, neo4j_database: str = "neo4j"):
        self.template_dir = template_dir or os.path.join(os.path.dirname(__file__), "cypher_templates")
        self.neo4j_database = neo4j_database
        self.scenes = self._load_scenes()

    def _load_scenes(self) -> Dict[str, CypherScene]:
        scenes_path = os.path.join(self.template_dir, "scenes.json")
        with open(scenes_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        scenes: Dict[str, CypherScene] = {}
        for raw in payload.get("scenes", []):
            query_path = os.path.join(self.template_dir, raw["query_file"])
            with open(query_path, "r", encoding="utf-8") as f:
                query = f.read().strip()

            scene = CypherScene(
                id=raw["id"],
                name=raw["name"],
                query_file=raw["query_file"],
                aliases=raw.get("aliases", []),
                keywords=raw.get("keywords", []),
                schema=raw.get("schema", {}),
                query=query,
            )
            scenes[scene.id] = scene

        logger.info("已加载 %s 个通风 Cypher 场景模板", len(scenes))
        return scenes

    def get_scene(self, scene_id: str) -> Optional[CypherScene]:
        scene_key = self._normalize_scene_id(scene_id)
        if scene_key in self.scenes:
            return self.scenes[scene_key]

        for scene in self.scenes.values():
            if scene_id in scene.aliases or scene_id == scene.name:
                return scene
        return None

    def list_scene_schemas(self) -> List[Dict[str, Any]]:
        """Return compact scene metadata for VL prompt construction."""
        return [
            {
                "id": scene.id,
                "name": scene.name,
                "aliases": scene.aliases,
                "keywords": scene.keywords,
                "schema": scene.schema,
            }
            for scene in self.scenes.values()
        ]

    def match_scene(self, structured_fields: Dict[str, Any], text: str = "") -> Optional[TemplateMatch]:
        """Pick the best template using scene id, populated fields, and keywords."""
        explicit_scene = structured_fields.get("scene") or structured_fields.get("scene_id")
        if explicit_scene:
            scene = self.get_scene(str(explicit_scene))
            if scene:
                return TemplateMatch(scene=scene, score=100, reasons=["explicit_scene"])

        haystack = self._build_haystack(structured_fields, text)
        best: Optional[TemplateMatch] = None

        for scene in self.scenes.values():
            score = 0
            reasons: List[str] = []

            for alias in scene.aliases:
                if alias and alias in haystack:
                    score += 12
                    reasons.append(f"alias:{alias}")

            for keyword in scene.keywords:
                if keyword and keyword in haystack:
                    score += 5
                    reasons.append(f"keyword:{keyword}")

            populated_schema_fields = [
                key for key in scene.schema
                if self._has_value(structured_fields.get(key))
            ]
            if populated_schema_fields:
                score += len(populated_schema_fields) * 3
                reasons.append("fields:" + ",".join(populated_schema_fields))

            if best is None or score > best.score:
                best = TemplateMatch(scene=scene, score=score, reasons=reasons)

        if best and best.score > 0:
            return best
        return None

    def build_params(
        self,
        scene: CypherScene,
        structured_fields: Dict[str, Any],
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Bind only known schema fields and common Cypher params."""
        params = {field: None for field in scene.schema}
        for field in scene.schema:
            params[field] = self._normalize_value(structured_fields.get(field))

        params["limit"] = int(limit)
        return params

    def execute(
        self,
        driver,
        structured_fields: Dict[str, Any],
        scene_id: Optional[str] = None,
        text: str = "",
        top_k: int = 5,
    ) -> Tuple[List[Document], Optional[TemplateMatch]]:
        """Run the matching Cypher template and return LangChain documents."""
        if scene_id:
            scene = self.get_scene(scene_id)
            match = TemplateMatch(scene=scene, score=100, reasons=["explicit_scene"]) if scene else None
        else:
            match = self.match_scene(structured_fields, text=text)
            scene = match.scene if match else None

        if not scene:
            logger.info("未匹配到通风 Cypher 场景模板")
            return [], None

        params = self.build_params(scene, structured_fields, limit=top_k)
        logger.info("执行 Cypher 模板: %s | reasons=%s", scene.id, match.reasons if match else [])

        with driver.session(database=self.neo4j_database) as session:
            result = session.run(scene.query, params)
            records = [record.data() if hasattr(record, "data") else dict(record) for record in result]

        docs = [self._record_to_document(record, scene, structured_fields) for record in records]
        return docs, match

    def _record_to_document(
        self,
        record: Dict[str, Any],
        scene: CypherScene,
        structured_fields: Dict[str, Any],
    ) -> Document:
        article_name = record.get("article_name") or "未知条款"
        article_title = record.get("article_title") or ""
        article_content = record.get("article_content") or ""
        constraints = self._compact_list(record.get("constraints", []))
        requirements = self._compact_list(record.get("requirements", []))
        facilities = self._compact_list(record.get("facilities", []))

        parts = [
            f"【模板场景】{scene.name}",
            f"【定位条款】{article_name} {article_title}".strip(),
        ]
        if article_content:
            parts.append(article_content)
        if constraints:
            parts.append("【结构化约束】\n" + self._format_constraints(constraints))
        if requirements:
            parts.append("【相关要求】\n" + "\n".join(f"- {item}" for item in requirements))
        if facilities:
            parts.append("【涉及设施】" + "、".join(str(item) for item in facilities))

        return Document(
            page_content="\n\n".join(parts),
            metadata={
                "node_id": record.get("node_id"),
                "article_name": article_name,
                "article_title": article_title,
                "retrieval_level": "cypher_template",
                "template_id": scene.id,
                "matched_location": record.get("matched_location"),
                "structured_fields": structured_fields,
            },
        )

    def _format_constraints(self, constraints: Iterable[Any]) -> str:
        lines = []
        for item in constraints:
            if not isinstance(item, dict):
                lines.append(f"- {item}")
                continue

            name = item.get("name") or "约束"
            value_min = item.get("value_min")
            value_max = item.get("value_max")
            unit = item.get("unit") or ""
            observed = item.get("observed")
            compliant = item.get("compliant")

            bounds = []
            if value_min is not None:
                bounds.append(f">={value_min}")
            if value_max is not None:
                bounds.append(f"<={value_max}")
            bound_text = "，".join(bounds) if bounds else "见条款要求"
            observed_text = f"，现场值={observed}{unit}" if observed is not None else ""
            compliant_text = ""
            if compliant is True:
                compliant_text = "，判定=符合"
            elif compliant is False:
                compliant_text = "，判定=不符合"

            lines.append(f"- {name}: {bound_text}{unit}{observed_text}{compliant_text}")
        return "\n".join(lines)

    def _compact_list(self, values: Any) -> List[Any]:
        if not values:
            return []
        if not isinstance(values, list):
            values = [values]
        compact = []
        seen = set()
        for item in values:
            if item in (None, "", {}):
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
            if key in seen:
                continue
            seen.add(key)
            compact.append(item)
        return compact

    def _build_haystack(self, structured_fields: Dict[str, Any], text: str) -> str:
        parts = [text or ""]
        for value in structured_fields.values():
            if isinstance(value, (dict, list)):
                parts.append(json.dumps(value, ensure_ascii=False))
            elif value is not None:
                parts.append(str(value))
        return " ".join(parts)

    def _normalize_scene_id(self, scene_id: str) -> str:
        return scene_id.strip().lower().replace("-", "_").replace(" ", "_")

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return None
        return value

    def _has_value(self, value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, (list, dict)):
            return bool(value)
        return True

