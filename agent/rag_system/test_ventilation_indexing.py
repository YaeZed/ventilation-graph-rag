"""
通风图索引模块 (VentilationGraphIndexingModule) 功能测试脚本

验证：
1. 实体键值对创建（Article, Parameter, Facility, etc.）
2. 关系键值对创建（CONSTRAINS, APPLIES_TO, etc.）
3. 去重逻辑（基于名称合并内容）
4. 统计信息输出
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

# 导入待测模块
from ventilation_graph_indexing import VentilationGraphIndexingModule

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- 模拟 Neo4j 节点类 ---
@dataclass
class MockNode:
    node_id: str
    name: str
    properties: Dict[str, Any]

def test_indexing():
    logger.info(">>> 开始测试 VentilationGraphIndexingModule...\n")
    
    # 1. 准备 Mock 数据
    # 模拟两个相同的条款（测试去重）
    articles = [
        MockNode("art_001", "第一百二十一条", {"title": "瓦斯抽采", "content": "必须建立瓦斯抽采系统。"}),
        MockNode("art_001_dup", "第一百二十一条", {"title": "瓦斯抽采", "content": "抽采系统应持续运行。"}),
    ]
    # 模拟一个参数
    parameters = [
        MockNode("param_001", "风速上限", {"value_max": 4.0, "unit": "m/s"})
    ]
    # 模拟一个设施
    facilities = [
        MockNode("fac_001", "局部通风机", {"description": "用于掘进工作面通风"})
    ]
    # 模拟一个地点
    locations = [
        MockNode("loc_001", "回风巷", {})
    ]
    # 模拟空的安全要求
    requirements = []

    # 2. 初始化模块
    indexer = VentilationGraphIndexingModule()

    # 3. 测试实体创建
    logger.info("步骤1: 创建实体键值对")
    indexer.create_entity_key_values(articles, parameters, requirements, facilities, locations)
    
    stats = indexer.get_statistics()
    logger.info(f"创建后统计: {stats}")
    
    # 4. 测试去重
    logger.info("\n步骤2: 执行去重逻辑")
    indexer.deduplicate_entities_and_relations()
    stats_after = indexer.get_statistics()
    logger.info(f"去重后统计: {stats_after}")
    
    # 验证去重结果
    art_kv = indexer.get_entities_by_key("第一百二十一条")[0]
    logger.info(f"合并后的条款内容:\n{art_kv.value_content}")

    # 5. 测试关系创建
    logger.info("\n步骤3: 创建关系键值对")
    # (条款) --[SPECIFIES]--> (参数)
    relationships = [
        ("art_001", "CONSTRAINS", "param_001"),
        ("fac_001", "APPLIES_TO", "loc_001")
    ]
    indexer.create_relation_key_values(relationships)
    
    rel_stats = indexer.get_statistics()
    logger.info(f"添加关系后统计: {rel_stats}")

    # 6. 测试检索
    logger.info("\n步骤4: 测试按键检索")
    results = indexer.get_entities_by_key("局部通风机")
    if results:
        logger.info(f"找到设施: {results[0].entity_name}, 类型: {results[0].entity_type}")

    rel_results = indexer.get_relations_by_key("适用地点")
    if rel_results:
        logger.info(f"找到关系键 '适用地点' 对应的内容:\n{rel_results[0].value_content}")

    logger.info("\n>>> 测试完成！")

if __name__ == "__main__":
    test_indexing()
