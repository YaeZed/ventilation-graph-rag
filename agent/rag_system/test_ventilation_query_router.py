"""
通风智能路由模块 (VentilationQueryRouter) 功能测试脚本

验证：
1. 关键词与关系强度分析 (Mock LLM)
2. 检索策略推荐逻辑
"""

import logging
from unittest.mock import MagicMock

# 导入待测模块
from ventilation_query_router import VentilationQueryRouter, RetrievalStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class MockConfig:
    llm_model = "qwen-plus"

def test_router():
    logger.info(">>> 开始测试 VentilationQueryRouter...\n")
    
    # 1. 准备 Mock
    config = MockConfig()
    llm_client = MagicMock()
    
    # 2. 初始化
    router = VentilationQueryRouter(config, llm_client)

    # 3. 测试复杂问题路由 (期望 Graph RAG)
    logger.info("测试场景 1: 复杂多跳问题")
    llm_client.chat.completions.create.return_value.choices[0].message.content = '{"keyword_intensity": 5, "relationship_connectivity": 9, "reasoning_complexity": 8, "query_characteristics": ["跨越条款", "多跳"], "recommended_strategy": "graph_rag"}'
    
    strategy, analysis = router.route_query("主要通风机停运后的备用切换和瓦斯检查联动流程是什么？")
    logger.info(f"路由策略: {strategy}, 特征分析: {analysis['query_characteristics']}")
    assert strategy == RetrievalStrategy.GRAPH_RAG

    # 4. 测试简单问题路由 (期望 Hybrid Traditional)
    logger.info("\n测试场景 2: 简单数值查找")
    llm_client.chat.completions.create.return_value.choices[0].message.content = '{"keyword_intensity": 9, "relationship_connectivity": 2, "reasoning_complexity": 2, "query_characteristics": ["精准关键词", "单条款"], "recommended_strategy": "hybrid_traditional"}'
    
    strategy, analysis = router.route_query("第一百五十条内容是什么？")
    logger.info(f"路由策略: {strategy}, 特征分析: {analysis['query_characteristics']}")
    assert strategy == RetrievalStrategy.HYBRID_TRADITIONAL

    logger.info("\n>>> 测试完成！")

if __name__ == "__main__":
    test_router()
