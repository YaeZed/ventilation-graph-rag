"""
矿井通风安全规程 - 智能查询路由模块

完全独立的路由模块，负责分析用户问题的通风专业特性，并分发到最合适的检索策略。
不再依赖外部 rag_modules。
"""

import json
import logging
from typing import Dict, Any, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)

class RetrievalStrategy(Enum):
    """检索策略枚举"""
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 混合传统检索（向量+图关键词）
    GRAPH_RAG = "graph_rag"                    # 深度图 RAG 检索
    COMBINED = "combined"                      # 组合策略
    HYBRID_FALLBACK = "hybrid_fallback"        # 降级传统检索

class VentilationQueryRouter:
    """
    通风安全规程智能路由模块 - 独立版
    核心功能：
    1. 意图特征分析：识别问题是否涉及复杂规程链条
    2. 检索策略推荐与执行：根据分析分发到对应的独立检索模块
    """

    def __init__(self, traditional_retrieval, graph_rag_retrieval, config, llm_client):
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.config = config
        self.llm_client = llm_client
        
        # 路由统计
        self.route_stats = {
            "traditional_count": 0,
            "graph_rag_count": 0,
            "combined_count": 0,
            "total_queries": 0
        }
        logger.info("通风智能路由模块初始化完成")

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """使用 LLM 分析问题的通风领域特征"""
        prompt = f"""作为矿井通风专家，分析以下技术问题，评估其检索需求。

问题：{query}

评估维度：
1. 关键词强度 (1-10)：问题中是否有明确的条款号、设施名。
2. 关系连接度 (1-10)：问题是否涉及多个规程节点间的互联。
3. 推理复杂度 (1-10)：是否需要逻辑推导。

请返回 JSON 格式：
{{
  "keyword_intensity": 8,
  "relationship_connectivity": 4,
  "reasoning_complexity": 3,
  "query_characteristics": ["明确条款", "数值判定"],
  "recommended_strategy": "hybrid_traditional"
}}

检索策略建议指南：
- 如果涉及“程序、步骤、联动、隔离、原因、影响”等逻辑链条：**必须优先建议 graph_rag**
- 如果关系连接度或复杂度 >= 5：建议 graph_rag
- 如果关键词强度很高且问题简单：建议 hybrid_traditional
- 其余情况：建议 combined
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.config, 'llm_model', 'qwen-plus'),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"): content = content[7:-3].strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"路由特征分析失败: {e}")
            return {"recommended_strategy": "combined"}

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Any], Any]:
        """执行路由并返回检索到的文档"""
        analysis_data = self.analyze_query(query)
        strategy_str = analysis_data.get("recommended_strategy", "combined")
        
        # 激进化调整：如果分析中包含关键特征，强制提升到 graph_rag
        trigger_keywords = ["程序", "步骤", "联动", "隔离", "关联", "影响"]
        query_chars = analysis_data.get("query_characteristics", [])
        if any(tk in query or any(tk in char for char in query_chars) for tk in trigger_keywords):
            if strategy_str == "hybrid_traditional":
                logger.info("检测到流程关联特征，将检索策略从 hybrid_traditional 提升至 graph_rag")
                strategy_str = "graph_rag"
        
        # 包装成类似父类的结构以兼容 Pipeline
        from dataclasses import dataclass
        @dataclass
        class SimpleAnalysis:
            recommended_strategy: Any
        
        @dataclass
        class StrategyVal:
            value: str
        
        analysis = SimpleAnalysis(recommended_strategy=StrategyVal(value=strategy_str))
        
        self.route_stats["total_queries"] += 1
        docs = []

        if strategy_str == "hybrid_traditional":
            self.route_stats["traditional_count"] += 1
            docs = self.traditional_retrieval.hybrid_search(query, top_k)
        elif strategy_str == "graph_rag":
            self.route_stats["graph_rag_count"] += 1
            docs = self.graph_rag_retrieval.graph_rag_search(query, top_k)
        else:
            self.route_stats["combined_count"] += 1
            # 组合逻辑：简单合并
            docs = self.traditional_retrieval.hybrid_search(query, top_k // 2)
            graph_docs = self.graph_rag_retrieval.graph_rag_search(query, top_k // 2)
            docs.extend(graph_docs)
            
        return docs, analysis

    def get_route_statistics(self) -> Dict[str, Any]:
        return self.route_stats
