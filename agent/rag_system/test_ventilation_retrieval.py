"""
通风混合检索模块 (VentilationHybridRetrieval) 功能测试脚本

验证：
1. 模块初始化与图索引构建
2. 关键词提取逻辑 (Mock LLM)
3. 邻居节点获取逻辑 (Mock Neo4j)
4. 综合检索结果拼装
"""

import logging
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

# 导入待测模块
from ventilation_hybrid_retrieval import VentilationHybridRetrieval

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Mock 组件 ---

class MockConfig:
    llm_model = "qwen-plus"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

@dataclass
class MockNode:
    node_id: str
    name: str
    properties: Dict[str, Any]

class MockDataModule:
    def __init__(self):
        self.articles = [MockNode("art_1", "第一百条", {"content": "规程内容"})]
        self.parameters = []
        self.requirements = []
        self.facilities = [MockNode("fac_1", "通风机", {})]
        self.locations = []
        self.relationships = [("art_1", "SPECIFIES", "fac_1")]

class MockMilvusModule:
    def similarity_search(self, query, k=5):
        return [
            {
                "text": "向量检索到的内容",
                "metadata": {"node_id": "art_1", "article_name": "第一百条"},
                "score": 0.9
            }
        ]

def test_retrieval():
    logger.info(">>> 开始测试 VentilationHybridRetrieval...\n")
    
    # 1. 准备 Mock
    config = MockConfig()
    data_module = MockDataModule()
    milvus_module = MockMilvusModule()
    
    llm_client = MagicMock()
    # 模拟 LLM 返回关键词 JSON
    llm_client.chat.completions.create.return_value.choices[0].message.content = '{"entity_keywords": ["通风机"], "topic_keywords": ["安装要求"]}'
    
    # 2. 初始化
    retriever = VentilationHybridRetrieval(config, data_module, milvus_module, llm_client)
    
    # 3. 拦截 Neo4j 驱动以免真实连接
    retriever.driver = MagicMock()
    mock_session = retriever.driver.session.return_value.__enter__.return_value
    mock_session.run.return_value = [{"name": "关联地点A"}, {"name": "关联参数B"}]

    # 4. 测试初始化与索引构建
    logger.info("测试步骤 1: 延迟初始化图索引")
    retriever.initialize()
    logger.info(f"图实体统计: {retriever.graph_indexing.get_statistics()}")

    # 5. 测试混合检索
    logger.info("\n测试步骤 2: 执行混合检索 (Hybrid Search)")
    query = "通风机怎么安装？"
    results = retriever.hybrid_search(query, top_k=2)
    
    logger.info(f"检索到 {len(results)} 条结果")
    for i, doc in enumerate(results):
        logger.info(f"结果 {i+1}:")
        logger.info(f"  内容: {doc.page_content[:100]}...")
        logger.info(f"  元数据: {doc.metadata}")

    # 6. 验证领域特定逻辑
    logger.info("\n测试步骤 3: 验证领域特定补全逻辑")
    # 检查是否包含 Mock 出来的邻居
    found_neighbors = "关联地点A" in results[0].page_content
    logger.info(f"是否包含邻居信息: {found_neighbors}")

    logger.info("\n>>> 测试完成！")

if __name__ == "__main__":
    test_retrieval()
