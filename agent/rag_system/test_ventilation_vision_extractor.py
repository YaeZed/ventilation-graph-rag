"""
Smoke tests for the two-stage ventilation vision extractor.

The test uses a fake OpenAI-compatible client and a tiny local image file, so it
does not call the real Qwen-VL service.
"""

import base64
import sys
import types
from pathlib import Path


try:
    from langchain_core.documents import Document
except ModuleNotFoundError:
    langchain_core = types.ModuleType("langchain_core")
    documents = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content="", metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents.Document = Document
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.documents"] = documents


try:
    from openai import OpenAI
except ModuleNotFoundError:
    openai = types.ModuleType("openai")

    class OpenAI:
        def __init__(self, *args, **kwargs):
            pass

    openai.OpenAI = OpenAI
    sys.modules["openai"] = openai


from ventilation_cypher_templates import VentilationCypherTemplateEngine
from ventilation_vision_extractor import VentilationVisionExtractor


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            content = '{"scene_id": "airflow_speed", "scene_name": "井巷风速合规", "confidence": 0.91, "reason": "看到巷道风速信息"}'
        else:
            content = '{"structured": {"location": "掘进中的岩巷", "airflow_speed": 0.2}, "description": "图片显示掘进中的岩巷，现场风速约0.2m/s。"}'
        return Obj(choices=[Obj(message=Obj(content=content))])


class FakeClient:
    def __init__(self):
        self.chat = Obj(completions=FakeCompletions())


def test_vision_extractor_two_stage():
    engine = VentilationCypherTemplateEngine()
    extractor = VentilationVisionExtractor(
        scene_schemas=engine.list_scene_schemas(),
        client=FakeClient(),
    )

    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    tmp_dir = Path(__file__).resolve().parents[2] / ".codex_tmp_pydeps"
    tmp_dir.mkdir(exist_ok=True)
    image_path = tmp_dir / "vision_extractor_sample.png"
    image_path.write_bytes(png_bytes)
    try:
        result = extractor.extract(str(image_path), user_question="判断图片风速是否合规")
    finally:
        try:
            image_path.unlink()
        except OSError:
            pass

    assert result.scene_id == "airflow_speed"
    assert result.structured_fields["location"] == "掘进中的岩巷"
    assert result.structured_fields["airflow_speed"] == 0.2
    assert "0.2m/s" in result.description


if __name__ == "__main__":
    test_vision_extractor_two_stage()
    print("ventilation vision extractor tests passed")
