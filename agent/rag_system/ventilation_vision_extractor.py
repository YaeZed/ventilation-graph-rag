"""
Two-stage Qwen-VL extractor for ventilation hazard images.

Stage 1 classifies the image into one of the deterministic Cypher template
scenes. Stage 2 extracts the structured fields required by that scene, plus a
natural-language description for vector fallback retrieval.
"""

import base64
import json
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class VisionExtractionResult:
    scene_id: str
    scene_name: str
    structured_fields: Dict[str, Any]
    description: str
    confidence: float = 0.0
    raw_classification: str = ""
    raw_extraction: str = ""


class VentilationVisionExtractor:
    """Classify a ventilation image and extract scene-specific fields."""

    def __init__(
        self,
        config: Any = None,
        scene_schemas: Optional[List[Dict[str, Any]]] = None,
        client: Optional[OpenAI] = None,
    ):
        self.config = config
        self.scene_schemas = scene_schemas or []
        self.model_name = (
            getattr(config, "vl_model", None)
            or os.getenv("QWEN_VL_MODEL")
            or os.getenv("VL_MODEL")
            or "qwen2.5-vl-72b-instruct"
        )

        self.client = client or OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY") or "sk-dummy",
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        logger.info("通风视觉提取模块初始化完成，模型: %s", self.model_name)

    def extract(self, image_path: str, user_question: str = "") -> VisionExtractionResult:
        """Run two-stage scene classification and structured extraction."""
        image_url = self._image_to_url(image_path)

        scene_id, scene_name, confidence, raw_classification = self._classify_scene(
            image_url=image_url,
            user_question=user_question,
        )
        scene_schema = self._find_scene_schema(scene_id)
        if not scene_schema:
            raise ValueError(f"VL 场景分类结果无对应 schema: {scene_id}")

        structured_fields, description, raw_extraction = self._extract_fields(
            image_url=image_url,
            scene_schema=scene_schema,
            user_question=user_question,
        )
        structured_fields["scene"] = scene_id

        return VisionExtractionResult(
            scene_id=scene_id,
            scene_name=scene_name or scene_schema.get("name", scene_id),
            structured_fields=structured_fields,
            description=description,
            confidence=confidence,
            raw_classification=raw_classification,
            raw_extraction=raw_extraction,
        )

    def _classify_scene(self, image_url: str, user_question: str = "") -> tuple[str, str, float, str]:
        scene_options = [
            {
                "id": scene.get("id"),
                "name": scene.get("name"),
                "keywords": scene.get("keywords", []),
                "aliases": scene.get("aliases", []),
            }
            for scene in self.scene_schemas
        ]
        prompt = f"""你是煤矿通风安全图像理解专家。请只从下面枚举的场景中选择一个最匹配的场景。

场景枚举：
{json.dumps(scene_options, ensure_ascii=False, indent=2)}

用户补充问题：{user_question or "无"}

请返回严格 JSON，不要输出 Markdown：
{{
  "scene_id": "airflow_speed",
  "scene_name": "井巷风速合规",
  "confidence": 0.0,
  "reason": "简要说明"
}}"""
        raw = self._chat_with_image(image_url, prompt)
        data = self._parse_json(raw)
        scene_id = self._normalize_scene_id(str(data.get("scene_id") or ""))
        scene_schema = self._find_scene_schema(scene_id)
        if not scene_schema:
            scene_id = self._fallback_scene_id(raw + " " + user_question)
            scene_schema = self._find_scene_schema(scene_id)

        if not scene_schema:
            raise ValueError(f"无法识别图片场景: {raw}")

        return (
            scene_schema["id"],
            data.get("scene_name") or scene_schema.get("name", scene_schema["id"]),
            float(data.get("confidence") or 0.0),
            raw,
        )

    def _extract_fields(
        self,
        image_url: str,
        scene_schema: Dict[str, Any],
        user_question: str = "",
    ) -> tuple[Dict[str, Any], str, str]:
        prompt = f"""你是煤矿通风安全图像结构化抽取专家。请根据图片和用户问题，按照给定 schema 抽取现场信息。

场景：{scene_schema.get("name")} ({scene_schema.get("id")})
字段 schema：
{json.dumps(scene_schema.get("schema", {}), ensure_ascii=False, indent=2)}

抽取要求：
1. 只能抽取图片或用户问题中能支持的信息；不确定的字段填 null。
2. 数值字段只返回数字，不要带单位。
3. 布尔字段返回 true/false/null。
4. description 用自然语言概括图片中的通风设施、地点、可见风险和不确定点。

用户补充问题：{user_question or "无"}

请返回严格 JSON，不要输出 Markdown：
{{
  "structured": {{}},
  "description": ""
}}"""
        raw = self._chat_with_image(image_url, prompt)
        data = self._parse_json(raw)
        structured = data.get("structured") or {}
        if not isinstance(structured, dict):
            structured = {}

        cleaned = self._clean_structured_fields(structured, scene_schema.get("schema", {}))
        description = str(data.get("description") or structured.get("description") or "").strip()
        if description:
            cleaned["description"] = description
        return cleaned, description, raw

    def _chat_with_image(self, image_url: str, prompt: str) -> str:
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
            temperature=0.1,
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

    def _clean_structured_fields(
        self,
        structured: Dict[str, Any],
        schema: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
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

    def _coerce_bool(self, value: Any) -> Optional[bool]:
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

    def _find_scene_schema(self, scene_id: str) -> Optional[Dict[str, Any]]:
        scene_key = self._normalize_scene_id(scene_id)
        for scene in self.scene_schemas:
            if scene.get("id") == scene_key:
                return scene
            aliases = scene.get("aliases", [])
            if scene_id in aliases or scene_id == scene.get("name"):
                return scene
        return None

    def _fallback_scene_id(self, text: str) -> Optional[str]:
        best_scene = None
        best_score = 0
        for scene in self.scene_schemas:
            score = 0
            for keyword in scene.get("keywords", []):
                if keyword and keyword in text:
                    score += 1
            if score > best_score:
                best_scene = scene
                best_score = score
        return best_scene.get("id") if best_scene else None

    def _normalize_scene_id(self, scene_id: str) -> str:
        return scene_id.strip().lower().replace("-", "_").replace(" ", "_")

