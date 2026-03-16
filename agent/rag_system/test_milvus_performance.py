import time
import logging
from typing import List, Dict, Any
from ventilation_milvus_index_construction import VentilationMilvusIndexConstruction

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MilvusTest")

def run_performance_test():
    """测试不同 HNSW 参数下的检索性能和召回率（假设已有数据）"""
    # 同步 pipeline 的配置名称
    milvus_module = VentilationMilvusIndexConstruction(collection_name="ventilation_safety")
    
    if not milvus_module.has_collection():
        print(f"❌ 错误：集合 '{milvus_module.collection_name}' 不存在，请先运行 pipeline 构建基础数据。")
        return

    # 测试用的测试问题
    test_queries = [
        "掘进工作面最低风速要求",
        "矿井主要通风机性能测试周期",
        "防灭火专项设计包含内容",
        "瓦斯抽采系统运行规范"
    ]

    # 1. 获取黄金标准（FLAT 索引，暴力搜索，召回率 100%）
    print("\n--- 正在建立黄金标准 (FLAT Index) ---")
    milvus_module.client.release_collection(milvus_module.collection_name)
    milvus_module.client.drop_index(milvus_module.collection_name, "vector")
    
    flat_params = milvus_module.client.prepare_index_params()
    flat_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    milvus_module.client.create_index(milvus_module.collection_name, flat_params)
    milvus_module.client.load_collection(milvus_module.collection_name)
    
    gold_standard = {}
    for q in test_queries:
        results = milvus_module.similarity_search(q, k=5)
        gold_standard[q] = set(hit["id"] for hit in results)

    # 2. 测试不同的 HNSW 参数组合
    # M: 邻居数 (影响索引大小和精度)
    # efConstruction: 构建时搜索范围 (影响构建速度和索引质量)
    m_values = [8, 16, 32]
    ef_values = [64, 128, 256]

    print(f"\n{'M':>2} | {'ef':>3} | {'建索引时间':>8} | {'查询耗时':>10} | {'召回率 (%)':>10}")
    print("-" * 55)

    for m in m_values:
        for ef in ef_values:
            # 计时：创建索引
            start_idx = time.time()
            milvus_module.create_index(M=m, ef_construction=ef)
            idx_time = time.time() - start_idx
            
            # 计时：平均查询时间
            start_query = time.time()
            total_recall = 0
            for q in test_queries:
                results = milvus_module.similarity_search(q, k=5)
                current_ids = set(hit["id"] for hit in results)
                # 计算与黄金标准重合度
                recall = len(current_ids.intersection(gold_standard[q])) / 5
                total_recall += recall
            
            avg_query_time = (time.time() - start_query) / len(test_queries)
            avg_recall = (total_recall / len(test_queries)) * 100
            
            print(f"{m:2d} | {ef:3d} | {idx_time:7.2f}s | {avg_query_time:8.4f}s | {avg_recall:9.1f}%")

if __name__ == "__main__":
    run_performance_test()
