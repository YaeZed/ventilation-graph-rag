"""Vision recognition evaluation helpers for real image samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EMPTY_VALUES = {"", "null", "none", "unknown", "未知", "不确定", "无"}


@dataclass
class FieldComparison:
    field: str
    expected: Any
    actual: Any
    matched: bool
    reason: str


def evaluate_vision_samples(pipeline, samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Run Qwen-VL extraction for uploaded samples and compare with labels."""
    results = []
    total_scene_expected = 0
    total_scene_matches = 0
    total_fields = 0
    total_field_matches = 0
    sample_scores = []

    for index, sample in enumerate(samples, start=1):
        image_path = sample["image_path"]
        question = sample.get("question") or "请识别图片中的煤矿通风场景和隐患要素"
        expected_scene = _clean_text(sample.get("expected_scene_id"))
        expected_fields = _clean_expected_fields(sample.get("expected_fields") or {})

        vision_result = pipeline.vision_extractor.extract(
            image_path=image_path,
            user_question=question,
        )

        predicted_scene = vision_result.scene_id
        scene_match = None
        if expected_scene:
            total_scene_expected += 1
            scene_match = _normalize_scene(expected_scene) == _normalize_scene(predicted_scene)
            total_scene_matches += int(scene_match)

        field_comparisons = [
            _compare_field(field, expected, vision_result.structured_fields.get(field))
            for field, expected in expected_fields.items()
        ]
        matched_fields = sum(1 for item in field_comparisons if item.matched)
        total_fields += len(field_comparisons)
        total_field_matches += matched_fields

        retrieval_doc_count = 0
        retrieval_error = ""
        template_scene = None
        if getattr(pipeline, "template_engine", None):
            try:
                docs, template_match = pipeline.template_engine.execute(
                    driver=pipeline.connection_manager.get_neo4j_driver(),
                    structured_fields=vision_result.structured_fields,
                    scene_id=vision_result.scene_id,
                    text=vision_result.description,
                    top_k=5,
                )
                retrieval_doc_count = len(docs)
                template_scene = getattr(template_match, "scene_id", None) if template_match else None
            except Exception as exc:
                retrieval_error = str(exc)

        score_parts = []
        if scene_match is not None:
            score_parts.append(1.0 if scene_match else 0.0)
        if field_comparisons:
            score_parts.append(matched_fields / len(field_comparisons))
        sample_score = sum(score_parts) / len(score_parts) if score_parts else None
        if sample_score is not None:
            sample_scores.append(sample_score)

        results.append(
            {
                "id": sample.get("id") or f"sample-{index}",
                "file_name": sample.get("file_name") or f"sample-{index}",
                "question": question,
                "expected_scene_id": expected_scene,
                "scene_match": scene_match,
                "field_accuracy": _ratio(matched_fields, len(field_comparisons)),
                "sample_score": sample_score,
                "prediction": {
                    "scene_id": vision_result.scene_id,
                    "scene_name": vision_result.scene_name,
                    "confidence": vision_result.confidence,
                    "structured_fields": vision_result.structured_fields,
                    "description": vision_result.description,
                    "risk_level": getattr(vision_result, "risk_level", ""),
                    "primary_hazard": getattr(vision_result, "primary_hazard", ""),
                    "key_observations": getattr(vision_result, "key_observations", []),
                    "uncertain_concepts": getattr(vision_result, "uncertain_concepts", []),
                    "concepts_retrieved": getattr(vision_result, "concepts_retrieved", []),
                },
                "retrieval": {
                    "template_scene_id": template_scene,
                    "document_count": retrieval_doc_count,
                    "hit": retrieval_doc_count > 0,
                    "error": retrieval_error,
                },
                "field_comparisons": [
                    {
                        "field": item.field,
                        "expected": item.expected,
                        "actual": item.actual,
                        "matched": item.matched,
                        "reason": item.reason,
                    }
                    for item in field_comparisons
                ],
            }
        )

    summary = {
        "total_samples": len(results),
        "scene_accuracy": _ratio(total_scene_matches, total_scene_expected),
        "field_accuracy": _ratio(total_field_matches, total_fields),
        "overall_accuracy": sum(sample_scores) / len(sample_scores) if sample_scores else None,
        "scene_expected_count": total_scene_expected,
        "field_expected_count": total_fields,
        "retrieval_hit_rate": _ratio(
            sum(1 for item in results if item["retrieval"]["hit"]),
            len(results),
        ),
    }

    return {
        "summary": summary,
        "samples": results,
        "markdown_report": build_markdown_report(summary, results),
    }


def build_markdown_report(summary: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    """Create a compact Markdown report for frontend rendering."""
    lines = [
        "# 真实图片识别精度验证报告",
        "",
        "## 总览",
        "",
        f"- 样本数：{summary['total_samples']}",
        f"- 场景准确率：{_format_percent(summary['scene_accuracy'])}",
        f"- 字段准确率：{_format_percent(summary['field_accuracy'])}",
        f"- 综合准确率：{_format_percent(summary['overall_accuracy'])}",
        f"- Cypher 检索命中率：{_format_percent(summary['retrieval_hit_rate'])}",
        "",
        "## 样本明细",
        "",
    ]

    for item in samples:
        prediction = item["prediction"]
        lines.extend(
            [
                f"### {item['file_name']}",
                "",
                f"- 期望场景：{item.get('expected_scene_id') or '未标注'}",
                f"- 预测场景：{prediction['scene_name']} (`{prediction['scene_id']}`)",
                f"- 场景是否匹配：{_format_match(item.get('scene_match'))}",
                f"- 字段准确率：{_format_percent(item.get('field_accuracy'))}",
                f"- Cypher 命中文档数：{item['retrieval']['document_count']}",
                f"- 图片描述：{prediction.get('description') or '无'}",
                "",
            ]
        )
        if item["field_comparisons"]:
            lines.extend(["| 字段 | 期望 | 识别结果 | 是否匹配 |", "|---|---:|---:|---|"])
            for field in item["field_comparisons"]:
                lines.append(
                    "| {field} | {expected} | {actual} | {matched} |".format(
                        field=field["field"],
                        expected=_format_value(field["expected"]),
                        actual=_format_value(field["actual"]),
                        matched=_format_match(field["matched"]),
                    )
                )
            lines.append("")
    return "\n".join(lines).strip()


def _compare_field(field: str, expected: Any, actual: Any) -> FieldComparison:
    if expected is None:
        return FieldComparison(field, expected, actual, actual is None, "期望为空")

    expected_bool = _coerce_bool(expected) if _is_bool_expectation(expected) else None
    actual_bool = _coerce_bool(actual)
    if expected_bool is not None:
        matched = expected_bool == actual_bool
        return FieldComparison(field, expected_bool, actual_bool, matched, "布尔值比较")

    expected_num = _coerce_number(expected)
    actual_num = _coerce_number(actual)
    if expected_num is not None:
        tolerance = max(0.01, abs(expected_num) * 0.05)
        matched = actual_num is not None and abs(expected_num - actual_num) <= tolerance
        return FieldComparison(field, expected_num, actual_num, matched, f"数值容差 ±{tolerance:.3g}")

    expected_text = _normalize_text(expected)
    actual_text = _normalize_text(actual)
    matched = bool(expected_text and actual_text) and (
        expected_text == actual_text
        or expected_text in actual_text
        or actual_text in expected_text
    )
    return FieldComparison(field, expected, actual, matched, "文本包含/等值比较")


def _clean_expected_fields(fields: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in fields.items():
        if not key:
            continue
        if isinstance(value, str) and value.strip().lower() in EMPTY_VALUES:
            continue
        cleaned[str(key).strip()] = value
    return cleaned


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_scene(value: Any) -> str:
    return _clean_text(value).lower().replace("-", "_").replace(" ", "_")


def _normalize_text(value: Any) -> str:
    return "".join(_clean_text(value).lower().split())


def _coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = _clean_text(value).lower()
    if text in {"true", "yes", "1", "有", "是", "具备", "存在"}:
        return True
    if text in {"false", "no", "0", "无", "否", "不具备", "不存在"}:
        return False
    return None


def _is_bool_expectation(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if value is None:
        return False
    text = _clean_text(value).lower()
    return text in {"true", "false", "yes", "no", "有", "无", "是", "否", "具备", "不具备", "存在", "不存在"}


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _format_percent(value: Any) -> str:
    if value is None:
        return "未标注"
    return f"{float(value) * 100:.1f}%"


def _format_match(value: Any) -> str:
    if value is None:
        return "未标注"
    return "匹配" if value else "不匹配"


def _format_value(value: Any) -> str:
    if value is None:
        return "null"
    return str(value).replace("|", "\\|")
