"""
Ventilation concept retriever.

The vision pipeline uses this module between the first and second VL passes:
Pass 1 names uncertain professional concepts, then this retriever returns
definition cards from Neo4j/Milvus. If the concept store has not been built yet,
it falls back to a compact built-in dictionary for the most common hazards.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "bge-small-zh-v1.5")

FALLBACK_CONCEPTS: list[dict[str, Any]] = [
    {
        "name": "串联通风",
        "aliases": ["串联", "一条龙通风"],
        "definition": "一个用风地点的回风又进入另一个用风地点继续使用，后续地点接受的是已经被污染或升温的风流。",
        "identification_features": "检查风流是否先经过一个作业点、硐室或设备区，再进入另一个用风点；关注回风是否直接进入下一用风地点、是否缺少独立新鲜风流供给。",
        "visual_clues": "多个作业点或设备沿同一巷道连续布置，风筒或风流方向指向后续工作面，缺少清晰的进回风分隔设施。",
        "typical_scenarios": "掘进工作面、相邻硐室、局部通风机和风筒布置不清晰的巷道。",
        "hazard_significance": "可能使瓦斯、粉尘、炮烟或热量在下游地点累积，降低后续地点空气质量和事故容错能力。",
    },
    {
        "name": "循环风",
        "aliases": ["循环通风", "局扇循环风"],
        "definition": "局部通风机吸入了自己送出的部分回风，污染空气在局部范围内反复循环。",
        "identification_features": "检查局扇吸入口是否处于回风侧或距回风口过近；观察风筒出风和局扇吸风之间是否形成短路回流。",
        "visual_clues": "局扇、风筒出口和回风通道距离较近，风筒出风方向可能回卷到局扇吸入口，缺少清晰的新鲜风流来源。",
        "typical_scenarios": "掘进工作面局部通风、临时风机布置、风筒末端管理不规范场景。",
        "hazard_significance": "会削弱稀释瓦斯和有害气体的能力，造成瓦斯或粉尘浓度反复升高。",
    },
    {
        "name": "局部通风机",
        "aliases": ["局扇", "局部风机"],
        "definition": "用于向掘进工作面等局部地点压入或抽出风流的通风设备，通常与风筒配合使用。",
        "identification_features": "检查安装地点、吸风来源、备用风机、风电闭锁、风筒连接和供风方向是否符合要求。",
        "visual_clues": "可见圆筒形或箱式风机、电机、支架、风筒接口，通常布置在巷道一侧并与柔性风筒连接。",
        "typical_scenarios": "掘进巷道、独头巷道、局部供风点。",
        "hazard_significance": "布置或管理不当会导致工作面供风不足、循环风、无计划停风等风险。",
    },
    {
        "name": "风筒",
        "aliases": ["导风筒", "局部通风风筒"],
        "definition": "局部通风系统中输送风流的柔性或刚性管道，用于把局部通风机风流送至作业地点。",
        "identification_features": "检查风筒吊挂、破损、漏风、接头、末端距工作面的距离和出风方向。",
        "visual_clues": "悬挂在巷道顶部或侧帮的圆形软管，可能有接头、吊挂点、弯折、破损或末端出口。",
        "typical_scenarios": "掘进工作面、局部通风巷道。",
        "hazard_significance": "漏风、脱节或末端距离过远会导致工作面有效风量不足。",
    },
    {
        "name": "风电闭锁",
        "aliases": ["风电闭锁装置", "闭锁"],
        "definition": "局部通风停止或风量不足时，自动切断相关区域电气设备电源的安全联锁措施。",
        "identification_features": "检查局部通风机、开关、传感器和被控设备之间是否具备联锁关系，停风后是否能切断工作面动力电。",
        "visual_clues": "可能出现局部通风机配电开关、传感器、控制箱、线缆和工作面电气设备；单靠图片通常需要结合现场描述确认。",
        "typical_scenarios": "掘进工作面机电设备和局部通风系统联动检查。",
        "hazard_significance": "闭锁失效会在停风或瓦斯积聚时仍允许电气设备运行，增加爆炸和中毒风险。",
    },
]


@dataclass
class ConceptCard:
    name: str
    aliases: list[str]
    definition: str
    identification_features: str
    visual_clues: str
    typical_scenarios: str
    hazard_significance: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "definition": self.definition,
            "identification_features": self.identification_features,
            "visual_clues": self.visual_clues,
            "typical_scenarios": self.typical_scenarios,
            "hazard_significance": self.hazard_significance,
            "source": self.source,
        }


class VentilationConceptRetriever:
    """Search ventilation concepts by name or semantic description."""

    def __init__(self, connection_manager=None, milvus_client=None, neo4j_driver=None):
        self._cm = connection_manager
        self._milvus = milvus_client
        self._driver = neo4j_driver
        self._embedder = None
        self._collection = "ventilation_concepts"
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._milvus is not None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(MODEL_PATH)
                self._milvus.load_collection(self._collection)
            except Exception as exc:
                logger.warning("概念向量库暂不可用，使用内置概念兜底: %s", exc)
        self._loaded = True

    def search_concepts(self, queries: list[str], extra_text: str = "", top_k: int = 5) -> list[ConceptCard]:
        self.ensure_loaded()
        cards: dict[str, ConceptCard] = {}

        for query in queries:
            for card in self._search_neo4j(query):
                cards[card.name] = self._merge_card(cards.get(card.name), card)

        if self._milvus is not None and self._embedder is not None:
            search_text = " ".join([*queries, extra_text]).strip()
            if search_text:
                for card in self._search_milvus(search_text, top_k):
                    cards[card.name] = self._merge_card(cards.get(card.name), card)

        results = list(cards.values())
        if len(results) < top_k:
            for card in self._search_fallback(queries, extra_text):
                if card.name not in cards:
                    results.append(card)
                    cards[card.name] = card
                if len(results) >= top_k:
                    break

        results.sort(key=lambda card: {"both": 0, "neo4j": 1, "milvus": 2, "fallback": 3}.get(card.source, 4))
        return results[:top_k]

    def format_cards(self, cards: list[ConceptCard]) -> str:
        if not cards:
            return "未检索到明确概念定义。"

        parts = []
        for index, card in enumerate(cards, start=1):
            aliases = "、".join(card.aliases) if card.aliases else "无"
            parts.append(
                "\n".join(
                    [
                        f"{index}. {card.name}（来源：{card.source}）",
                        f"- 别名：{aliases}",
                        f"- 定义：{card.definition}",
                        f"- 现场判别：{card.identification_features}",
                        f"- 视觉线索：{card.visual_clues}",
                        f"- 风险意义：{card.hazard_significance}",
                    ]
                )
            )
        return "\n\n".join(parts)

    def _merge_card(self, existing: ConceptCard | None, incoming: ConceptCard) -> ConceptCard:
        if existing is None:
            return incoming
        existing.source = "both" if existing.source != incoming.source else existing.source
        for field in ("definition", "identification_features", "visual_clues", "typical_scenarios", "hazard_significance"):
            if not getattr(existing, field) and getattr(incoming, field):
                setattr(existing, field, getattr(incoming, field))
        if not existing.aliases and incoming.aliases:
            existing.aliases = incoming.aliases
        return existing

    def _search_neo4j(self, concept_name: str) -> list[ConceptCard]:
        if not self._driver or not concept_name.strip():
            return []
        cards = []
        try:
            with self._driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Concept)
                    WHERE toLower(c.name) CONTAINS toLower($name)
                       OR any(alias IN coalesce(c.aliases, []) WHERE toLower(alias) CONTAINS toLower($name))
                    RETURN c.name AS name, c.aliases AS aliases, c.definition AS definition,
                           c.identification_features AS identification_features,
                           c.visual_clues AS visual_clues,
                           c.typical_scenarios AS typical_scenarios,
                           c.hazard_significance AS hazard_significance
                    LIMIT 5
                    """,
                    name=concept_name.strip(),
                )
                for record in result:
                    cards.append(
                        ConceptCard(
                            name=record.get("name") or concept_name,
                            aliases=record.get("aliases") or [],
                            definition=record.get("definition") or "",
                            identification_features=record.get("identification_features") or "",
                            visual_clues=record.get("visual_clues") or "",
                            typical_scenarios=record.get("typical_scenarios") or "",
                            hazard_significance=record.get("hazard_significance") or "",
                            source="neo4j",
                        )
                    )
        except Exception as exc:
            logger.warning("Neo4j 概念检索失败: %s", exc)
        return cards

    def _search_milvus(self, text: str, top_k: int) -> list[ConceptCard]:
        cards = []
        try:
            vector = self._embedder.encode(text, normalize_embeddings=True).tolist()
            results = self._milvus.search(
                collection_name=self._collection,
                data=[vector],
                limit=top_k,
                output_fields=["name", "definition", "visual_clues", "identification_features"],
            )
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity") or hit.get("fields") or {}
                    cards.append(
                        ConceptCard(
                            name=entity.get("name") or "",
                            aliases=[],
                            definition=entity.get("definition") or "",
                            identification_features=entity.get("identification_features") or "",
                            visual_clues=entity.get("visual_clues") or "",
                            typical_scenarios="",
                            hazard_significance="",
                            source="milvus",
                        )
                    )
        except Exception as exc:
            logger.warning("Milvus 概念检索失败: %s", exc)
        return [card for card in cards if card.name]

    def _search_fallback(self, queries: list[str], extra_text: str = "") -> list[ConceptCard]:
        text = " ".join([*queries, extra_text]).lower()
        scored: list[tuple[float, dict[str, Any]]] = []
        broad_terms = ["通风", "风机", "局扇", "风筒", "回风", "掘进", "工作面", "风流"]

        for item in FALLBACK_CONCEPTS:
            terms = [item["name"], *item.get("aliases", [])]
            score = sum(2 for term in terms if term and term.lower() in text)
            score += sum(0.2 for term in broad_terms if term in text)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            ConceptCard(
                name=item["name"],
                aliases=item.get("aliases") or [],
                definition=item.get("definition") or "",
                identification_features=item.get("identification_features") or "",
                visual_clues=item.get("visual_clues") or "",
                typical_scenarios=item.get("typical_scenarios") or "",
                hazard_significance=item.get("hazard_significance") or "",
                source="fallback",
            )
            for _, item in scored
        ]
