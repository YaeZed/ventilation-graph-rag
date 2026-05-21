"""
Smoke tests for the deterministic ventilation Cypher template engine.

These tests avoid a live Neo4j dependency by using a tiny fake driver.
"""

import sys
import types


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


from ventilation_cypher_templates import VentilationCypherTemplateEngine


class FakeRecord(dict):
    def data(self):
        return dict(self)


class FakeSession:
    def __init__(self):
        self.last_query = None
        self.last_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, params):
        self.last_query = query
        self.last_params = params
        return [
            FakeRecord(
                node_id="ART_第一百五十七条",
                article_name="第一百五十七条",
                article_title="井巷允许风流速度限值及特殊情形调整规定",
                article_content="井巷中的风流速度应当符合表6要求。",
                matched_location="掘进中的岩巷",
                constraints=[
                    {
                        "name": "最低风速",
                        "value_min": 0.15,
                        "value_max": None,
                        "unit": "m/s",
                        "observed": params.get("airflow_speed"),
                        "compliant": True,
                    }
                ],
                requirements=[],
                facilities=[],
            )
        ]


class FakeDriver:
    def __init__(self):
        self.session_obj = FakeSession()

    def session(self, database=None):
        self.database = database
        return self.session_obj


def test_match_and_execute_airflow_speed():
    engine = VentilationCypherTemplateEngine()
    fields = {
        "location": "掘进中的岩巷",
        "airflow_speed": 0.2,
        "description": "掘进中的岩巷风速为0.2m/s",
    }

    match = engine.match_scene(fields)
    assert match is not None
    assert match.scene.id == "airflow_speed"

    docs, execute_match = engine.execute(FakeDriver(), fields, top_k=3)
    assert execute_match.scene.id == "airflow_speed"
    assert len(docs) == 1
    assert docs[0].metadata["template_id"] == "airflow_speed"
    assert "第一百五十七条" in docs[0].page_content
    assert "判定=符合" in docs[0].page_content


def test_scene_schema_export_contains_vl_fields():
    engine = VentilationCypherTemplateEngine()
    schemas = {scene["id"]: scene for scene in engine.list_scene_schemas()}
    assert "air_quality" in schemas
    assert "methane_concentration" in schemas["air_quality"]["schema"]
    assert "local_ventilation" in schemas
    assert "distance_to_return_air_m" in schemas["local_ventilation"]["schema"]


if __name__ == "__main__":
    test_match_and_execute_airflow_speed()
    test_scene_schema_export_contains_vl_fields()
    print("ventilation cypher template tests passed")

