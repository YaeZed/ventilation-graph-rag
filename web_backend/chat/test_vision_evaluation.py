"""Smoke tests for vision evaluation metrics."""

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from vision_evaluation import evaluate_vision_samples


class FakeVisionExtractor:
    def extract(self, image_path, user_question=""):
        return SimpleNamespace(
            scene_id="local_ventilation",
            scene_name="局部通风机与风筒",
            confidence=0.88,
            structured_fields={
                "scene": "local_ventilation",
                "facility_type": "局部通风机",
                "distance_to_return_air_m": 11.0,
                "has_backup_fan": True,
            },
            description="图片显示局部通风机和风筒。",
        )


class FakeTemplateEngine:
    def execute(self, **kwargs):
        return [Document(page_content="局部通风机安装要求", metadata={"node_id": "1"})], SimpleNamespace(
            scene_id="local_ventilation"
        )


class FakeConnectionManager:
    def get_neo4j_driver(self):
        return object()


class FakePipeline:
    vision_extractor = FakeVisionExtractor()
    template_engine = FakeTemplateEngine()
    connection_manager = FakeConnectionManager()


def test_evaluate_vision_samples_metrics():
    report = evaluate_vision_samples(
        FakePipeline(),
        [
            {
                "id": "sample-1",
                "file_name": "fan.jpg",
                "image_path": str(Path("fan.jpg")),
                "question": "识别局部通风机",
                "expected_scene_id": "local_ventilation",
                "expected_fields": {
                    "facility_type": "局部通风机",
                    "distance_to_return_air_m": 10.8,
                    "has_backup_fan": True,
                    "door_count": 1,
                },
            }
        ],
    )

    assert report["summary"]["total_samples"] == 1
    assert report["summary"]["scene_accuracy"] == 1.0
    assert report["summary"]["field_accuracy"] == 0.75
    assert report["summary"]["retrieval_hit_rate"] == 1.0
    assert "真实图片识别精度验证报告" in report["markdown_report"]


if __name__ == "__main__":
    test_evaluate_vision_samples_metrics()
    print("vision evaluation tests passed")
