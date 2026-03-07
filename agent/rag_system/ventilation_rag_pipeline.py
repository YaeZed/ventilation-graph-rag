"""
矿井通风安全规程 - 智能辨识系统 RAG 流水线

整合以下模块：
    1. VentilationDataPreparationModule  - 从 Neo4j 读取规程数据
    2. MilvusIndexConstructionModule     - 向量索引（通风规程版）
    3. HybridRetrievalModule             - 混合检索
    4. GraphRAGRetrieval                 - 图 RAG 检索
    5. IntelligentQueryRouter            - 智能路由
    6. VentilationGenerationModule       - 专家级答案生成

用法：
    python ventilation_rag_pipeline.py                    # 交互式问答
    python ventilation_rag_pipeline.py --build-index      # 强制重建向量索引
    python ventilation_rag_pipeline.py -q "掘进工作面最低风速要求是多少"
"""

import sys
import os
import logging
import argparse
from dotenv import load_dotenv

# ── 路径设置 ─────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.join(BASE_DIR, '..')
sys.path.insert(0, PARENT_DIR)

# 加载 .env（DASHSCOPE_API_KEY）
load_dotenv(dotenv_path=os.path.join(PARENT_DIR, '..', '..', '.env'))
load_dotenv()   

# ── 日志 ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ventilation_pipeline")

# ── 导入模块 ─────────────────────────────────────────────────
from ventilation_data_preparation import VentilationDataPreparationModule
from ventilation_generation import VentilationGenerationModule
from ventilation_hybrid_retrieval import VentilationHybridRetrieval
from ventilation_graph_rag_retrieval import VentilationGraphRAGRetrieval
from rag_modules.milvus_index_construction import MilvusIndexConstructionModule
from rag_modules.intelligent_query_router import IntelligentQueryRouter

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
class VentilationConfig:
    # Neo4j (lowercase to match hybrid_retrieval / graph_rag_retrieval expectations)
    neo4j_uri      = "bolt://localhost:7687"
    neo4j_user     = "neo4j"
    neo4j_password = "160722yaesakura"

    # Milvus
    milvus_host      = "localhost"
    milvus_port      = 19530
    collection_name  = "ventilation_safety"
    vector_dimension = 512
    embedding_model  = "BAAI/bge-small-zh-v1.5"

    # LLM (lowercase to match extract_query_keywords / understand_graph_query)
    llm_model   = "qwen-plus"
    temperature = 0.1
    max_tokens  = 2048

    # 文档分块
    chunk_size    = 800
    chunk_overlap = 100


# ══════════════════════════════════════════════════════════════
# 主流水线类
# ══════════════════════════════════════════════════════════════
class VentilationRAGPipeline:
    """
    通风安全规程智能检索流水线

    初始化顺序：
        1. 数据准备（Neo4j → 文档）
        2. Milvus 向量索引
        3. 混合检索 + 图 RAG 检索
        4. 智能路由
        5. 答案生成
    """

    def __init__(self, cfg: VentilationConfig = None, force_rebuild_index: bool = False):
        self.cfg = cfg or VentilationConfig()
        self.force_rebuild = force_rebuild_index

        self.data_module   = None
        self.milvus_module = None
        self.hybrid_ret    = None
        self.graph_ret     = None
        self.router        = None
        self.generator     = None

    def initialize(self):
        """按顺序初始化所有模块"""
        logger.info("=" * 60)
        logger.info("🚀 矿井通风安全规程智能辨识系统 - 初始化")
        logger.info("=" * 60)

        cfg = self.cfg

        # ── 1. LLM 客户端（生成模块先初始化，路由也需要它）
        logger.info("[1/5] 初始化 LLM 生成模块...")
        self.generator = VentilationGenerationModule(
            model_name=cfg.llm_model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        llm_client = self.generator.client   # 供路由器复用

        # ── 2. 数据准备
        logger.info("[2/5] 从 Neo4j 加载规程数据...")
        self.data_module = VentilationDataPreparationModule(
            uri=cfg.neo4j_uri, user=cfg.neo4j_user, password=cfg.neo4j_password
        )
        stats = self.data_module.load_graph_data()
        logger.info(f"  Neo4j 数据：{stats}")

        docs = self.data_module.build_article_documents()
        chunks = self.data_module.chunk_documents(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
        logger.info(f"  文档：{len(docs)} 篇，分块：{len(chunks)} 块")

        # ── 3. Milvus 向量索引
        logger.info("[3/5] 初始化 Milvus 向量索引...")
        self.milvus_module = MilvusIndexConstructionModule(
            host=cfg.milvus_host,
            port=cfg.milvus_port,
            collection_name=cfg.collection_name,
            dimension=cfg.vector_dimension,
            model_name=cfg.embedding_model,
        )

        if self.force_rebuild or not self.milvus_module.has_collection():
            logger.info("  正在构建向量索引（首次或强制重建）...")
            self.milvus_module.create_collection(force_recreate=self.force_rebuild)
            self.milvus_module.build_vector_index(chunks)
        else:
            logger.info("  向量索引已存在，直接加载")
            self.milvus_module.load_collection()

        # ── 4. 检索模块（通风规程优化版）
        logger.info("[4/5] 初始化检索模块...")
        self.hybrid_ret = VentilationHybridRetrieval(
            config=cfg, milvus_module=self.milvus_module,
            data_module=self.data_module, llm_client=llm_client
        )
        self.hybrid_ret.initialize(chunks)

        self.graph_ret = VentilationGraphRAGRetrieval(config=cfg, llm_client=llm_client)
        self.graph_ret.initialize()

        # ── 5. 智能路由
        logger.info("[5/5] 初始化智能查询路由器...")
        self.router = IntelligentQueryRouter(
            traditional_retrieval=self.hybrid_ret,
            graph_rag_retrieval=self.graph_ret,
            llm_client=llm_client,
            config=cfg,
        )

        logger.info("✅ 所有模块初始化完成！")
        logger.info("=" * 60)

    def query(self, question: str, top_k: int = 5, stream: bool = False) -> str:
        """
        智能问答接口

        Args:
            question: 用户提问
            top_k: 检索返回数量
            stream: 是否流式输出

        Returns:
            答案字符串（stream=False）或 生成器（stream=True）
        """
        if not self.router:
            raise RuntimeError("请先调用 initialize() 初始化流水线")

        logger.info(f"\n{'─'*60}\n❓ 问题：{question}\n{'─'*60}")

        # 路由 + 检索（route_query 返回 (List[Document], QueryAnalysis) 元组）
        docs, analysis = self.router.route_query(question, top_k=top_k)
        strategy = analysis.recommended_strategy.value

        # ── Fallback：GraphRAG 文档数不足时（< 2），降级到混合检索 ──
        # 场景：graph_rag 找不到匹配节点（0文档），或只找到1个内容不足的文档
        if len(docs) < 2 and strategy == "graph_rag":
            logger.info(f"  GraphRAG 仅返回 {len(docs)} 个文档，自动降级到混合检索...")
            hybrid_docs = self.hybrid_ret.hybrid_search(question, top_k=top_k)
            if hybrid_docs:
                docs     = hybrid_docs
                strategy = "hybrid_fallback"

        # 记录给测试脚本使用
        self._last_strategy  = strategy
        self._last_doc_count = len(docs)

        route_stats = self.router.get_route_statistics()
        logger.info(f"  检索到 {len(docs)} 个相关文档 | 策略: {strategy} | 路由统计: {route_stats}")

        # 生成答案
        if stream:
            return self.generator.generate_adaptive_answer_stream(question, docs)
        else:
            answer = self.generator.generate_adaptive_answer(question, docs)
            return answer

    def close(self):
        """释放所有连接"""
        for mod in [self.data_module, self.milvus_module,
                    self.hybrid_ret, self.graph_ret]:
            if mod and hasattr(mod, 'close'):
                try:
                    mod.close()
                except Exception:
                    pass
        logger.info("所有连接已关闭")


# ══════════════════════════════════════════════════════════════
# 命令行入口
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="矿井通风安全规程智能辨识系统")
    parser.add_argument("--build-index", action="store_true",
                        help="强制重建 Milvus 向量索引")
    parser.add_argument("-q", "--question", type=str,
                        help="直接输入问题（非交互模式）")
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索返回文档数（默认5）")
    parser.add_argument("--stream", action="store_true",
                        help="流式输出答案")
    args = parser.parse_args()

    pipeline = VentilationRAGPipeline(force_rebuild_index=args.build_index)

    try:
        pipeline.initialize()

        if args.question:
            # 单次问答模式
            if args.stream:
                print("\n💬 回答：")
                for chunk in pipeline.query(args.question, top_k=args.top_k, stream=True):
                    print(chunk, end="", flush=True)
                print()
            else:
                answer = pipeline.query(args.question, top_k=args.top_k)
                print(f"\n💬 回答：\n{answer}")
        else:
            # 交互式问答模式
            print("\n🏭 矿井通风安全规程智能辨识系统")
            print("   输入问题进行查询，输入 'quit' 或 'q' 退出\n")
            while True:
                try:
                    question = input("❓ 请输入问题：").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not question:
                    continue
                if question.lower() in ('quit', 'q', 'exit'):
                    break

                print("\n🔍 正在检索和生成答案...\n")
                if args.stream:
                    print("💬 回答：")
                    for chunk in pipeline.query(question, top_k=args.top_k, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    answer = pipeline.query(question, top_k=args.top_k)
                    print(f"💬 回答：\n{answer}\n")
                    print("─" * 60)

    finally:
        pipeline.close()
        print("\n👋 系统已退出")


if __name__ == "__main__":
    main()
