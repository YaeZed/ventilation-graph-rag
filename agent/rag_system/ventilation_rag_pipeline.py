"""Coal-mine ventilation safety GraphRAG pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, PARENT_DIR)

load_dotenv(dotenv_path=os.path.join(PARENT_DIR, "..", "..", ".env"))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ventilation_pipeline")


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


from ventilation_data_preparation import VentilationDataPreparationModule
from ventilation_generation import VentilationGenerationModule
from ventilation_hybrid_retrieval import VentilationHybridRetrieval
from ventilation_graph_rag_retrieval import VentilationGraphRAGRetrieval
from ventilation_milvus_index_construction import VentilationMilvusIndexConstruction
from ventilation_query_router import VentilationQueryRouter
from ventilation_cypher_templates import VentilationCypherTemplateEngine
from ventilation_vision_extractor import VentilationVisionExtractor
from ventilation_concept_retriever import VentilationConceptRetriever
from connection_manager import ConnectionManager


class VentilationConfig:
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "160722yaesakura")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    milvus_host = os.getenv("MILVUS_HOST", "localhost")
    milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
    collection_name = os.getenv("MILVUS_COLLECTION", "ventilation_safety")
    vector_dimension = 512
    embedding_model = os.path.join(BASE_DIR, "..", "..", "models", "bge-small-zh-v1.5")

    llm_model = os.getenv("LLM_MODEL", "qwen-plus")
    vl_model = os.getenv("QWEN_VL_MODEL", os.getenv("VL_MODEL", "qwen3.5-omni-plus"))
    temperature = 0.1
    max_tokens = 2048

    chunk_size = 800
    chunk_overlap = 100


class VentilationRAGPipeline:
    """Main GraphRAG pipeline for text and image-grounded safety analysis."""

    def __init__(self, cfg: VentilationConfig | None = None, force_rebuild_index: bool = False):
        self.cfg = cfg or VentilationConfig()
        self.force_rebuild = force_rebuild_index

        self.data_module = None
        self.milvus_module = None
        self.hybrid_ret = None
        self.graph_ret = None
        self.router = None
        self.generator = None
        self.template_engine = None
        self.vision_extractor = None
        self.concept_retriever = None
        self.connection_manager = ConnectionManager.get_instance(self.cfg)

    def initialize(self) -> None:
        logger.info("=" * 60)
        logger.info("矿井通风安全规程智能辨识系统 - 初始化")
        logger.info("=" * 60)

        cfg = self.cfg
        self.connection_manager.configure(cfg)

        logger.info("[1/5] 初始化 LLM 生成模块...")
        self.generator = VentilationGenerationModule(
            model_name=cfg.llm_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        llm_client = self.generator.client

        logger.info("[2/5] 从 Neo4j 加载规程数据...")
        neo4j_driver = self.connection_manager.get_neo4j_driver(verify=True)
        self.data_module = VentilationDataPreparationModule(
            uri=cfg.neo4j_uri,
            user=cfg.neo4j_user,
            password=cfg.neo4j_password,
            database=cfg.neo4j_database,
            neo4j_driver=neo4j_driver,
        )
        stats = self.data_module.load_graph_data()
        logger.info("  Neo4j 数据: %s", stats)

        docs = self.data_module.build_article_documents()
        chunks = self.data_module.chunk_documents(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        logger.info("  文档: %s 篇，分块: %s 块", len(docs), len(chunks))

        logger.info("[3/5] 初始化 Milvus 向量索引...")
        milvus_client = self.connection_manager.get_milvus_client()
        self.milvus_module = VentilationMilvusIndexConstruction(
            host=cfg.milvus_host,
            port=cfg.milvus_port,
            collection_name=cfg.collection_name,
            dimension=cfg.vector_dimension,
            model_name=cfg.embedding_model,
            milvus_client=milvus_client,
        )
        if self.force_rebuild or not self.milvus_module.has_collection():
            logger.info("  正在构建向量索引...")
            self.milvus_module.create_collection(force_recreate=self.force_rebuild)
            self.milvus_module.build_vector_index(chunks)
        else:
            logger.info("  向量索引已存在，直接加载")
            self.milvus_module.load_collection()

        logger.info("[4/5] 初始化检索模块...")
        self.hybrid_ret = VentilationHybridRetrieval(
            config=cfg,
            milvus_module=self.milvus_module,
            data_module=self.data_module,
            llm_client=llm_client,
            neo4j_driver=neo4j_driver,
        )
        self.hybrid_ret.initialize(chunks)

        self.graph_ret = VentilationGraphRAGRetrieval(
            config=cfg,
            llm_client=llm_client,
            neo4j_driver=neo4j_driver,
        )
        self.graph_ret.initialize()

        logger.info("[5/5] 初始化路由、模板、概念和视觉模块...")
        self.router = VentilationQueryRouter(
            traditional_retrieval=self.hybrid_ret,
            graph_rag_retrieval=self.graph_ret,
            config=cfg,
            llm_client=llm_client,
        )
        self.template_engine = VentilationCypherTemplateEngine(neo4j_database=cfg.neo4j_database)
        self.concept_retriever = VentilationConceptRetriever(
            connection_manager=self.connection_manager,
            milvus_client=milvus_client,
            neo4j_driver=neo4j_driver,
        )
        self.vision_extractor = VentilationVisionExtractor(
            config=cfg,
            scene_schemas=self.template_engine.list_scene_schemas(),
            client=llm_client,
            concept_retriever=self.concept_retriever,
        )

        logger.info("所有模块初始化完成")
        logger.info("=" * 60)

    def query(self, question: str, top_k: int = 5, stream: bool = False, image_path: str | None = None):
        if not self.router:
            raise RuntimeError("请先调用 initialize() 初始化流水线")

        logger.info("\n%s\n问题: %s\n%s", "-" * 60, question, "-" * 60)

        if image_path:
            if stream:
                return self.query_image_events(question=question, image_path=image_path, top_k=top_k)
            return self._query_with_image(question=question, image_path=image_path, top_k=top_k, stream=False)

        docs, analysis = self.router.route_query(question, top_k=top_k)
        strategy = analysis.recommended_strategy.value

        if len(docs) < 2 and strategy == "graph_rag":
            logger.info("  GraphRAG 仅返回 %s 个文档，降级到混合检索", len(docs))
            hybrid_docs = self.hybrid_ret.hybrid_search(question, top_k=top_k)
            if hybrid_docs:
                docs = hybrid_docs
                strategy = "hybrid_fallback"

        self._last_strategy = strategy
        self._last_doc_count = len(docs)

        route_stats = self.router.get_route_statistics()
        logger.info("  检索到 %s 个相关文档 | 策略: %s | 路由统计: %s", len(docs), strategy, route_stats)

        if stream:
            return self.generator.generate_adaptive_answer_stream(question, docs)
        return self.generator.generate_adaptive_answer(question, docs)

    def query_image_events(self, question: str, image_path: str, top_k: int = 5):
        """Yield step/token/done dictionaries for SSE image analysis."""
        if not self.vision_extractor or not self.template_engine:
            raise RuntimeError("视觉提取或 Cypher 模板引擎尚未初始化")

        image_url = self.vision_extractor._image_to_url(image_path)

        yield {"type": "step", "step": "vision_observe", "message": "正在初步观察图片..."}
        observation = self.vision_extractor.observe(image_url=image_url, user_question=question)
        uncertain = observation.get("uncertain_concepts") or []
        yield {
            "type": "step",
            "step": "vision_observe_done",
            "message": f"识别到 {len(uncertain)} 个待确认概念：{'、'.join(uncertain) if uncertain else '无'}",
            "data": {
                "uncertain_concepts": uncertain,
                "raw_observations": observation.get("raw_observations", ""),
                "preliminary_scene": observation.get("preliminary_scene", ""),
            },
        }

        yield {"type": "step", "step": "concept_search", "message": "正在检索通风概念定义..."}
        concepts = self.vision_extractor.retrieve_concepts(observation)
        concept_names = [getattr(card, "name", "") for card in concepts if getattr(card, "name", "")]
        yield {
            "type": "step",
            "step": "concept_search_done",
            "message": f"检索到 {len(concepts)} 个概念定义",
            "data": {"concept_count": len(concepts), "concepts": concept_names},
        }

        yield {"type": "step", "step": "vision_analyze", "message": "正在结合概念定义深度分析图片..."}
        vision_result = self.vision_extractor.analyze_with_concepts(
            image_url=image_url,
            user_question=question,
            observation=observation,
            concepts=concepts,
        )
        yield {
            "type": "step",
            "step": "vision_analyze_done",
            "message": f"场景：{vision_result.scene_name} | 风险：{vision_result.risk_level}",
            "data": {
                "scene_id": vision_result.scene_id,
                "scene_name": vision_result.scene_name,
                "risk_level": vision_result.risk_level,
                "primary_hazard": vision_result.primary_hazard,
                "key_observations": vision_result.key_observations,
            },
        }

        yield {"type": "step", "step": "cypher_match", "message": "正在匹配规程模板..."}
        retrieval_question, docs, match = self._retrieve_docs_for_image(question, vision_result, top_k)
        yield {
            "type": "step",
            "step": "cypher_match_done",
            "message": f"匹配到 {len(docs)} 条相关规程内容",
            "data": {
                "doc_count": len(docs),
                "template_scene_id": getattr(match, "scene_id", None) if match else None,
            },
        }

        yield {"type": "step", "step": "generating", "message": "正在生成辨识报告..."}
        for chunk in self.generator.generate_image_answer_stream(retrieval_question, docs, vision_result):
            yield {"type": "token", "content": chunk}
        yield {"type": "done", "message": "completed"}

    def _query_with_image(self, question: str, image_path: str, top_k: int, stream: bool):
        if not self.vision_extractor or not self.template_engine:
            raise RuntimeError("视觉提取或 Cypher 模板引擎尚未初始化")

        logger.info("执行图片前置识别: %s", image_path)
        vision_result = self.vision_extractor.extract(image_path=image_path, user_question=question)
        retrieval_question, docs, _match = self._retrieve_docs_for_image(question, vision_result, top_k)

        if stream:
            return self.generator.generate_image_answer_stream(retrieval_question, docs, vision_result)
        return self.generator.generate_image_answer(retrieval_question, docs, vision_result)

    def _retrieve_docs_for_image(self, question: str, vision_result: Any, top_k: int):
        retrieval_question = self._build_image_retrieval_question(question, vision_result)
        docs, match = self.template_engine.execute(
            driver=self.connection_manager.get_neo4j_driver(),
            structured_fields=vision_result.structured_fields,
            scene_id=vision_result.scene_id,
            text=vision_result.description,
            top_k=top_k,
        )

        if len(docs) < top_k and vision_result.description:
            fallback_docs = self.hybrid_ret.hybrid_search(vision_result.description, top_k=top_k)
            docs = self._merge_docs(docs, fallback_docs)

        for doc in docs:
            doc.metadata["vision_scene_id"] = vision_result.scene_id
            doc.metadata["vision_scene_name"] = vision_result.scene_name
            doc.metadata["vision_confidence"] = vision_result.confidence
            doc.metadata["vision_risk_level"] = getattr(vision_result, "risk_level", "")

        self._last_strategy = "vision_cypher_template"
        self._last_doc_count = len(docs)
        self._last_vision_result = vision_result
        self._last_template_match = match

        logger.info("图片检索完成 | 场景=%s | 文档数=%s", vision_result.scene_id, len(docs))
        return retrieval_question, docs, match

    def _build_image_retrieval_question(self, question: str, vision_result: Any) -> str:
        structured = {
            key: value
            for key, value in vision_result.structured_fields.items()
            if value is not None
        }
        return (
            f"{question}\n\n"
            f"【图片识别场景】{vision_result.scene_name} ({vision_result.scene_id})\n"
            f"【风险等级】{getattr(vision_result, 'risk_level', '需要注意')}\n"
            f"【主要隐患】{getattr(vision_result, 'primary_hazard', '')}\n"
            f"【关键观察】{getattr(vision_result, 'key_observations', [])}\n"
            f"【图片结构化字段】{structured}\n"
            f"【图片描述】{vision_result.description}"
        )

    def _merge_docs(self, primary_docs, fallback_docs):
        merged = []
        seen = set()
        for doc in primary_docs + fallback_docs:
            key = (
                doc.metadata.get("node_id"),
                doc.metadata.get("article_name"),
                doc.page_content[:120],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
        return merged

    def close(self) -> None:
        for mod in [self.data_module, self.milvus_module, self.hybrid_ret, self.graph_ret]:
            if mod and hasattr(mod, "close"):
                try:
                    mod.close()
                except Exception:
                    pass
        if self.connection_manager:
            self.connection_manager.close_all()
        logger.info("所有连接已关闭")


def main() -> None:
    _configure_console_encoding()

    parser = argparse.ArgumentParser(description="矿井通风安全规程智能辨识系统")
    parser.add_argument("--build-index", action="store_true", help="强制重建 Milvus 向量索引")
    parser.add_argument("-q", "--question", type=str, help="直接输入问题")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回文档数")
    parser.add_argument("--stream", action="store_true", help="流式输出答案")
    parser.add_argument("--image", type=str, help="可选图片路径")
    args = parser.parse_args()

    pipeline = VentilationRAGPipeline(force_rebuild_index=args.build_index)
    try:
        pipeline.initialize()
        if args.question:
            if args.stream:
                print("\n回答：")
                for item in pipeline.query(args.question, top_k=args.top_k, stream=True, image_path=args.image):
                    if isinstance(item, dict):
                        if item.get("type") == "token":
                            print(item.get("content", ""), end="", flush=True)
                        elif item.get("type") == "step":
                            print(f"\n[{item.get('step')}] {item.get('message')}")
                    else:
                        print(item, end="", flush=True)
                print()
            else:
                answer = pipeline.query(args.question, top_k=args.top_k, image_path=args.image)
                print(f"\n回答：\n{answer}")
        else:
            print("\n矿井通风安全规程智能辨识系统")
            print("输入问题进行查询，输入 quit 或 q 退出\n")
            while True:
                try:
                    question = input("请输入问题：").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not question:
                    continue
                if question.lower() in {"quit", "q", "exit"}:
                    break

                print("\n正在检索和生成答案...\n")
                if args.stream:
                    for chunk in pipeline.query(question, top_k=args.top_k, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    answer = pipeline.query(question, top_k=args.top_k)
                    print(f"回答：\n{answer}\n")
                    print("-" * 60)
    finally:
        pipeline.close()
        print("\n系统已退出")


if __name__ == "__main__":
    main()
