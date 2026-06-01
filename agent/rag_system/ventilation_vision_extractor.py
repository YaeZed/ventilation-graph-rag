"""
Qwen-VL extractor for ventilation hazard images.

The image flow is intentionally two-pass:
1. Observe the site photo and name uncertain ventilation concepts.
2. Retrieve concept definition cards, then analyze the same image again with
   those definitions injected into the prompt.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class VisionExtractionResult:
    scene_id: str
    scene_name: str
    structured_fields: dict[str, Any]
    description: str
    confidence: float = 0.0
    raw_classification: str = ""
    raw_extraction: str = ""
    raw_observations: str = ""
    uncertain_concepts: list[str] = field(default_factory=list)
    concepts_retrieved: list[dict[str, Any]] = field(default_factory=list)
    key_observations: list[str] = field(default_factory=list)
    primary_hazard: str = ""
    risk_level: str = "需要注意"


@dataclass
class MultiImageResult(VisionExtractionResult):
    per_image_observations: dict[int, str] = field(default_factory=dict)
    cross_image_findings: list[str] = field(default_factory=list)


class VentilationVisionExtractor:
    """Observe, classify, and extract structured fields from a site image."""

    def __init__(
        self,
        config: Any = None,
        scene_schemas: list[dict[str, Any]] | None = None,
        client: OpenAI | None = None,
        concept_retriever: Any = None,
    ):
        self.config = config
        self.scene_schemas = scene_schemas or []
        self.concept_retriever = concept_retriever
        self.model_name = (
            getattr(config, "vl_model", None)
            or os.getenv("QWEN_VL_MODEL")
            or os.getenv("VL_MODEL")
            or "qwen3.5-omni-plus"
        )
        self.client = client or OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY") or "sk-dummy",
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        logger.info("通风视觉提取模块初始化完成，模型: %s", self.model_name)

    def extract(self, image_path: str, user_question: str = "") -> VisionExtractionResult:
        """Run the full observe -> concept retrieval -> analysis flow."""
        image_url = self._image_to_url(image_path)
        observation = self.observe(image_url=image_url, user_question=user_question)
        concepts = self.retrieve_concepts(observation)
        return self.analyze_with_concepts(
            image_url=image_url,
            user_question=user_question,
            observation=observation,
            concepts=concepts,
        )

    def extract_multi(self, image_paths: list[str], user_question: str = "") -> MultiImageResult:
        """Run independent observation for each image, then one joint analysis pass."""
        if not image_paths:
            raise ValueError("至少需要 1 张图片")
        if len(image_paths) == 1:
            single = self.extract(image_paths[0], user_question=user_question)
            return MultiImageResult(
                **single.__dict__,
                per_image_observations={1: single.raw_observations or single.description},
                cross_image_findings=[],
            )

        image_urls = [self._image_to_url(path) for path in image_paths]
        observations = []
        for index, image_url in enumerate(image_urls, start=1):
            observations.append(
                self.observe(
                    image_url=image_url,
                    user_question=(
                        f"{user_question or '请判断现场通风安全隐患'}\n"
                        f"这是同一现场的第 {index}/{len(image_urls)} 张照片，请保留可与其他照片交叉验证的线索。"
                    ),
                )
            )

        merged_observation = self._merge_observations(observations)
        concepts = self.retrieve_concepts(merged_observation)
        return self.analyze_multi_with_concepts(
            image_urls=image_urls,
            user_question=user_question,
            observations=observations,
            concepts=concepts,
        )

    def observe(self, image_url: str, user_question: str = "") -> dict[str, Any]:
        """Pass 1: observe the image without forcing a final conclusion."""
        prompt = f"""你是一位矿井通风安全检查员，正在查看井下现场照片。

请仔细观察图片，先不要急于下最终结论，回答以下问题：
1. 你看到了哪些通风设施、设备和环境特征？
2. 图片中哪些现象可能提示存在安全隐患？
3. 有哪些通风专业概念需要更多定义才能确认？例如“循环风”“串联通风”“风电闭锁”等。
4. 初步猜测属于哪种检查场景？

用户补充问题：{user_question or "无"}

请返回严格 JSON，不要输出 Markdown：
{{
  "raw_observations": "对图片可见内容的客观观察",
  "uncertain_concepts": ["概念1", "概念2"],
  "preliminary_scene": "初步场景",
  "preliminary_concern": "初步担忧"
}}"""
        raw = self._chat_with_image(image_url, prompt, temperature=0.3)
        data = self._parse_json(raw)
        concepts = data.get("uncertain_concepts") or []
        if not isinstance(concepts, list):
            concepts = [concepts]
        return {
            "raw_observations": str(data.get("raw_observations") or ""),
            "uncertain_concepts": [str(item).strip() for item in concepts if str(item).strip()],
            "preliminary_scene": str(data.get("preliminary_scene") or ""),
            "preliminary_concern": str(data.get("preliminary_concern") or ""),
            "raw_response": raw,
        }

    def retrieve_concepts(self, observation: dict[str, Any]) -> list[Any]:
        """Retrieve professional concept definition cards for Pass 2."""
        if not self.concept_retriever:
            return []
        extra_text = " ".join(
            [
                observation.get("raw_observations", ""),
                observation.get("preliminary_scene", ""),
                observation.get("preliminary_concern", ""),
            ]
        )
        return self.concept_retriever.search_concepts(
            queries=observation.get("uncertain_concepts") or [],
            extra_text=extra_text,
            top_k=5,
        )

    def analyze_with_concepts(
        self,
        image_url: str,
        user_question: str = "",
        observation: dict[str, Any] | None = None,
        concepts: list[Any] | None = None,
    ) -> VisionExtractionResult:
        """Pass 2: analyze the image with concept definitions injected."""
        observation = observation or {}
        concepts = concepts or []
        scene_options = [
            {
                "id": scene.get("id"),
                "name": scene.get("name"),
                "keywords": scene.get("keywords", []),
                "aliases": scene.get("aliases", []),
                "schema": scene.get("schema", {}),
            }
            for scene in self.scene_schemas
        ]
        concept_cards = self._format_concepts(concepts)

        prompt = f"""你是一位经验丰富的煤矿通风安全图像理解专家。

你已经完成第一轮观察：
{json.dumps(observation, ensure_ascii=False, indent=2)}

现在你获得了以下通风专业概念定义，请带着这些定义重新分析图片。

【概念参考卡片】
{concept_cards}

【可选场景与字段 schema】
{json.dumps(scene_options, ensure_ascii=False, indent=2)}

用户补充问题：{user_question or "无"}

请完成：
1. 从可选场景中选择最匹配的 scene_id。
2. 按该场景 schema 提取 structured 字段；不确定填 null。
3. 给出 key_observations、primary_hazard 和 risk_level。
4. 区分图片能直接支持的事实与需要结合现场经验的判断。

risk_level 只能取："正常"、"需要注意"、"疑似隐患"、"明确隐患"。

请返回严格 JSON，不要输出 Markdown：
{{
  "scene_id": "airflow_speed",
  "scene_name": "井巷风速合规",
  "confidence": 0.0,
  "structured": {{}},
  "description": "自然语言描述",
  "key_observations": ["观察1", "观察2"],
  "primary_hazard": "主要隐患判断",
  "risk_level": "疑似隐患"
}}"""
        raw = self._chat_with_image(image_url, prompt, temperature=0.25)
        data = self._parse_json(raw)

        scene_id = self._normalize_scene_id(str(data.get("scene_id") or ""))
        scene_schema = self._find_scene_schema(scene_id)
        if not scene_schema:
            scene_id = self._fallback_scene_id(
                " ".join(
                    [
                        raw,
                        user_question,
                        observation.get("raw_observations", ""),
                        observation.get("preliminary_scene", ""),
                    ]
                )
            )
            scene_schema = self._find_scene_schema(scene_id or "")

        if not scene_schema:
            raise ValueError(f"无法识别图片场景: {raw}")

        structured = data.get("structured") or {}
        if not isinstance(structured, dict):
            structured = {}
        cleaned = self._clean_structured_fields(structured, scene_schema.get("schema", {}))
        description = str(data.get("description") or structured.get("description") or "").strip()
        if description:
            cleaned["description"] = description
        cleaned["scene"] = scene_schema["id"]

        key_observations = data.get("key_observations") or []
        if not isinstance(key_observations, list):
            key_observations = [key_observations]

        return VisionExtractionResult(
            scene_id=scene_schema["id"],
            scene_name=data.get("scene_name") or scene_schema.get("name", scene_schema["id"]),
            structured_fields=cleaned,
            description=description,
            confidence=float(data.get("confidence") or 0.0),
            raw_classification=json.dumps(observation, ensure_ascii=False),
            raw_extraction=raw,
            raw_observations=observation.get("raw_observations", ""),
            uncertain_concepts=observation.get("uncertain_concepts", []),
            concepts_retrieved=[self._concept_to_dict(card) for card in concepts],
            key_observations=[str(item).strip() for item in key_observations if str(item).strip()],
            primary_hazard=str(data.get("primary_hazard") or ""),
            risk_level=self._normalize_risk_level(str(data.get("risk_level") or "")),
        )

    def analyze_multi_with_concepts(
        self,
        image_urls: list[str],
        user_question: str = "",
        observations: list[dict[str, Any]] | None = None,
        concepts: list[Any] | None = None,
    ) -> MultiImageResult:
        """Pass 2 for multiple images: reason over all observations and images together."""
        observations = observations or []
        concepts = concepts or []
        scene_options = [
            {
                "id": scene.get("id"),
                "name": scene.get("name"),
                "keywords": scene.get("keywords", []),
                "aliases": scene.get("aliases", []),
                "schema": scene.get("schema", {}),
            }
            for scene in self.scene_schemas
        ]
        concept_cards = self._format_concepts(concepts)
        observation_blocks = []
        for index, observation in enumerate(observations, start=1):
            observation_blocks.append(
                "\n".join(
                    [
                        f"【图片 {index} 观察】",
                        f"- 客观观察：{observation.get('raw_observations', '')}",
                        f"- 待确认概念：{observation.get('uncertain_concepts', [])}",
                        f"- 初步场景：{observation.get('preliminary_scene', '')}",
                        f"- 初步担忧：{observation.get('preliminary_concern', '')}",
                    ]
                )
            )

        prompt = f"""你是一位经验丰富的煤矿通风安全图像理解专家，正在联合分析同一现场的多张照片。

你已经完成各图独立观察：
{chr(10).join(observation_blocks) or "无独立观察记录"}

【概念参考卡片】
{concept_cards}

【可选场景与字段 schema】
{json.dumps(scene_options, ensure_ascii=False, indent=2)}

用户补充问题：{user_question or "无"}

请综合所有图片和概念定义，完成：
1. 判断各图之间可能的空间关系、因果关系或证据互补关系。
2. 识别单张图难以确认、但多图联合可以加强判断的问题。
3. 从可选场景中选择最匹配的 scene_id，并按 schema 提取 structured 字段。
4. 给出综合 key_observations、per_image_observations、cross_image_findings、primary_hazard 和 risk_level。

risk_level 只能取："正常"、"需要注意"、"疑似隐患"、"明确隐患"。

请返回严格 JSON，不要输出 Markdown：
{{
  "scene_id": "airflow_speed",
  "scene_name": "井巷风速合规",
  "confidence": 0.0,
  "structured": {{}},
  "description": "多图综合描述",
  "key_observations": ["综合观察1", "综合观察2"],
  "per_image_observations": {{"1": "图片1观察", "2": "图片2观察"}},
  "cross_image_findings": ["跨图关联发现1"],
  "primary_hazard": "主要隐患判断",
  "risk_level": "疑似隐患"
}}"""
        raw = self._chat_with_images(image_urls, prompt, temperature=0.25)
        data = self._parse_json(raw)

        scene_id = self._normalize_scene_id(str(data.get("scene_id") or ""))
        scene_schema = self._find_scene_schema(scene_id)
        if not scene_schema:
            scene_id = self._fallback_scene_id(
                " ".join(
                    [
                        raw,
                        user_question,
                        " ".join(str(item.get("raw_observations", "")) for item in observations),
                        " ".join(str(item.get("preliminary_scene", "")) for item in observations),
                    ]
                )
            )
            scene_schema = self._find_scene_schema(scene_id or "")

        if not scene_schema:
            raise ValueError(f"无法识别多图场景: {raw}")

        structured = data.get("structured") or {}
        if not isinstance(structured, dict):
            structured = {}
        cleaned = self._clean_structured_fields(structured, scene_schema.get("schema", {}))
        description = str(data.get("description") or structured.get("description") or "").strip()
        if description:
            cleaned["description"] = description
        cleaned["scene"] = scene_schema["id"]

        key_observations = self._coerce_string_list(data.get("key_observations"))
        cross_image_findings = self._coerce_string_list(data.get("cross_image_findings"))
        per_image_observations = self._coerce_per_image_observations(
            data.get("per_image_observations"),
            observations,
        )
        uncertain_concepts = self._unique_strings(
            concept
            for observation in observations
            for concept in observation.get("uncertain_concepts", [])
        )

        return MultiImageResult(
            scene_id=scene_schema["id"],
            scene_name=data.get("scene_name") or scene_schema.get("name", scene_schema["id"]),
            structured_fields=cleaned,
            description=description,
            confidence=float(data.get("confidence") or 0.0),
            raw_classification=json.dumps(observations, ensure_ascii=False),
            raw_extraction=raw,
            raw_observations="\n".join(
                f"图片{index}: {observation.get('raw_observations', '')}"
                for index, observation in enumerate(observations, start=1)
            ),
            uncertain_concepts=uncertain_concepts,
            concepts_retrieved=[self._concept_to_dict(card) for card in concepts],
            key_observations=key_observations,
            primary_hazard=str(data.get("primary_hazard") or ""),
            risk_level=self._normalize_risk_level(str(data.get("risk_level") or "")),
            per_image_observations=per_image_observations,
            cross_image_findings=cross_image_findings,
        )

    def _chat_with_image(self, image_url: str, prompt: str, temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _chat_with_images(self, image_urls: list[str], prompt: str, temperature: float = 0.1) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _image_to_url(self, image_path: str) -> str:
        if image_path.startswith(("http://", "https://", "data:")):
            return image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _parse_json(self, content: str) -> dict[str, Any]:
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

    def _clean_structured_fields(
        self,
        structured: dict[str, Any],
        schema: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for field, spec in schema.items():
            value = structured.get(field)
            if value in ("", "未知", "不确定"):
                value = None

            field_type = spec.get("type")
            if value is not None and field_type == "number":
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    value = None
            elif value is not None and field_type == "boolean":
                value = self._coerce_bool(value)

            cleaned[field] = value
        return cleaned

    def _coerce_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"true", "yes", "1", "有", "是", "具备", "存在"}:
            return True
        if text in {"false", "no", "0", "无", "否", "不具备", "不存在"}:
            return False
        return None

    def _find_scene_schema(self, scene_id: str) -> dict[str, Any] | None:
        scene_key = self._normalize_scene_id(scene_id)
        for scene in self.scene_schemas:
            if scene.get("id") == scene_key:
                return scene
            aliases = scene.get("aliases", [])
            if scene_id in aliases or scene_id == scene.get("name"):
                return scene
        return None

    def _fallback_scene_id(self, text: str) -> str | None:
        best_scene = None
        best_score = 0
        for scene in self.scene_schemas:
            score = 0
            for keyword in scene.get("keywords", []):
                if keyword and keyword in text:
                    score += 1
            for alias in scene.get("aliases", []):
                if alias and alias in text:
                    score += 1
            if scene.get("name") and scene.get("name") in text:
                score += 1
            if score > best_score:
                best_scene = scene
                best_score = score
        return best_scene.get("id") if best_scene else None

    def _normalize_scene_id(self, scene_id: str) -> str:
        return scene_id.strip().lower().replace("-", "_").replace(" ", "_")

    def _normalize_risk_level(self, risk_level: str) -> str:
        allowed = {"正常", "需要注意", "疑似隐患", "明确隐患"}
        value = risk_level.strip()
        return value if value in allowed else "需要注意"

    def _merge_observations(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "raw_observations": "\n".join(
                f"图片{index}: {observation.get('raw_observations', '')}"
                for index, observation in enumerate(observations, start=1)
            ),
            "uncertain_concepts": self._unique_strings(
                concept
                for observation in observations
                for concept in observation.get("uncertain_concepts", [])
            ),
            "preliminary_scene": "；".join(
                str(observation.get("preliminary_scene", ""))
                for observation in observations
                if observation.get("preliminary_scene")
            ),
            "preliminary_concern": "；".join(
                str(observation.get("preliminary_concern", ""))
                for observation in observations
                if observation.get("preliminary_concern")
            ),
        }

    def _coerce_string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    def _coerce_per_image_observations(
        self,
        value: Any,
        fallback_observations: list[dict[str, Any]],
    ) -> dict[int, str]:
        if isinstance(value, dict):
            items = value.items()
            result = {}
            for key, text in items:
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    continue
                if str(text).strip():
                    result[index] = str(text).strip()
            if result:
                return result

        return {
            index: str(observation.get("raw_observations", "")).strip()
            for index, observation in enumerate(fallback_observations, start=1)
            if str(observation.get("raw_observations", "")).strip()
        }

    def _unique_strings(self, values) -> list[str]:
        result = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            result.append(text)
        return result

    def _format_concepts(self, concepts: list[Any]) -> str:
        if self.concept_retriever and hasattr(self.concept_retriever, "format_cards"):
            return self.concept_retriever.format_cards(concepts)
        if not concepts:
            return "未检索到明确概念定义。"
        return json.dumps([self._concept_to_dict(card) for card in concepts], ensure_ascii=False, indent=2)

    def _concept_to_dict(self, card: Any) -> dict[str, Any]:
        if hasattr(card, "as_dict"):
            return card.as_dict()
        if isinstance(card, dict):
            return card
        return {
            "name": getattr(card, "name", ""),
            "aliases": getattr(card, "aliases", []),
            "definition": getattr(card, "definition", ""),
            "identification_features": getattr(card, "identification_features", ""),
            "visual_clues": getattr(card, "visual_clues", ""),
            "typical_scenarios": getattr(card, "typical_scenarios", ""),
            "hazard_significance": getattr(card, "hazard_significance", ""),
            "source": getattr(card, "source", ""),
        }
