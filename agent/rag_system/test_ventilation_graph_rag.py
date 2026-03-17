"""
通风图 RAG 检索模块 (VentilationGraphRAGRetrieval) 功能测试脚本

验证：
1. 规程意图理解 (Mock LLM)
2. 图搜索逻辑流程 (Mock Neo4j)
3. 条款内容回查与格式化
"""

import logging
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

# 导入待测模块
from ventilation_graph_rag_retrieval import VentilationGraphRAGRetrieval, QueryType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Mock 组件 ---

class MockConfig:
    llm_model = "qwen-plus"
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"

def test_graph_rag():
    logger.info(">>> 开始测试 VentilationGraphRAGRetrieval...\n")
    
    # 1. 准备 Mock
    config = MockConfig()
    llm_client = MagicMock()
    # 模拟 LLM 返回意图解析 JSON
    llm_client.chat.completions.create.return_value.choices[0].message.content = '{"query_type": "multi_hop", "source_entities": ["局部通风机"], "target_entities": [], "relation_types": ["INVOLVES_FACILITY"], "max_depth": 2}'
    
    # 2. 初始化
    grag = VentilationGraphRAGRetrieval(config, llm_client)
    
    # 3. 拦截 Neo4j 驱动
    grag.driver = MagicMock()
    mock_session = grag.driver.session.return_value.__enter__.return_value
    
    # 模拟多跳路径返回
    mock_session.run.side_effect = [
        # 第一次调用：_build_graph_index (空)
        MagicMock(),
        # 第二次调用：_execute_multi_hop
        [
            {
                "ns": [{"node_id": "art_121", "name": "第一百二十一条"}],
                "rs": [{"type": "INVOLVES_FACILITY"}],
                "len": 1
            }
        ],
        # 第三次调用：_fetch_article_content
        [
            {
                "node_id": "art_121",
                "name": "第一百二十一条",
                "title": "瓦斯管理",
                "content": "必须建立可靠的通风系统。",
                "params": [{"name": "瓦斯浓度", "value_max": 1.0}],
                "reqs": [{"name": "安装要求", "content": "应安装在进风巷。"}]
            }
        ]
    ]

    # 4. 测试索引构建
    logger.info("测试步骤 1: 验证索引构建调用")
    grag.initialize()
    
    # 5. 测试意图解析
    logger.info("\n测试步骤 2: 验证通风领域意图解析")
    q = grag.understand_graph_query("局部通风机安装在那？")
    logger.info(f"解析类型: {q.query_type}, 源实体: {q.source_entities}")

    # 6. 测试完整搜索流
    logger.info("\n测试步骤 3: 验证图 RAG 搜索与内容回查")
    docs = grag.graph_rag_search("测试查询", top_k=1)
    
    if docs:
        logger.info(f"检索到文档: {docs[0].metadata['article_name']}")
        logger.info(f"文档内容摘要:\n{docs[0].page_content[:150]}...")
        
        # 验证是否包含了回查到的数值参数
        has_params = "瓦斯浓度" in docs[0].page_content
        logger.info(f"是否包含回查的指标: {has_params}")

    logger.info("\n>>> 测试完成！")

if __name__ == "__main__":
    test_graph_rag()
