#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矿井通风安全知识图谱构建器 - 生成 Neo4j 导入格式 CSV
"""

import os
import logging
import pandas as pd
import uuid
from typing import List, Dict, Set
# 导入之前定义的模型类
from ventilation_safety_agent import RegulationArticle, SafetyMetric, SafetyRequirement

logger = logging.getLogger(__name__)

class VentilationKGBuilder:
    """
    通风安全图谱构建器
    
    将结构化的规程条款数据转化为图数据库节点和关系，
    并导出为 Neo4j 批量导入格式的 CSV 文件。
    
    图谱结构：
        Article -[CONSTRAINS]-> Parameter -[APPLIES_TO]-> Location
        Article -[SPECIFIES]-> Requirement -[INVOLVES_FACILITY]-> Facility
    """
    
    def __init__(self, output_dir: str = "./safety_output"):
        """
        初始化构建器
        
        Args:
            output_dir: CSV文件输出目录，默认为 ./safety_output
        """
        logger.info(f"初始化VentilationKGBuilder，输出目录: {output_dir}")
        self.output_dir = output_dir
        self.nodes = []
        self.relationships = []
        # 用 set 追踪已存在的节点ID，避免重复添加 Location/Facility 节点
        self._existing_node_ids: Set[str] = set()
        
        # 确保输出目录存在
        # os.makedirs(exist_ok=True) 比 os.path.exists + os.makedirs 更简洁
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"输出目录已就绪: {output_dir}")

    def _add_node(self, node: dict) -> bool:
        """
        添加节点（内部方法，自动去重）
        
        Args:
            node: 节点字典，必须包含 'id:ID' 键
        
        Returns:
            bool: True=新增成功，False=节点已存在跳过
        """
        node_id = node["id:ID"]
        if node_id in self._existing_node_ids:
            logger.debug(f"节点已存在，跳过: {node_id}")
            return False
        self._existing_node_ids.add(node_id)
        self.nodes.append(node)
        return True

    def process_article(self, article: RegulationArticle):
        """
        将解析后的规程条款转化为图节点和关系
        
        Args:
            article: 已解析的规程条款对象
        """
        logger.info(f"开始构建图谱节点: {article.article_number}")
        
        # ── 1. 创建顶级节点：Article (条款) ──────────────────────────
        article_node_id = f"ART_{article.article_number}"
        self._add_node({
            "id:ID": article_node_id,
            ":LABEL": "Article",
            "name": article.article_number,
            "title": article.title,
            "content": article.content
        })

        # ── 2. 处理安全指标：Parameter (参数) ───────────────────────
        for metric in article.metrics:
            # 使用 metric.id 而不是随机uuid，保证可追溯性
            param_id = f"PAR_{metric.id}"
            self._add_node({
                "id:ID": param_id,
                ":LABEL": "Parameter",
                "name": metric.name,
                "value_min": metric.threshold_min,
                "value_max": metric.threshold_max,
                "unit": metric.unit
            })
            
            # 关系：条款 -[CONSTRAINS]-> 参数
            self.relationships.append({
                ":START_ID": article_node_id,
                ":END_ID": param_id,
                ":TYPE": "CONSTRAINS",
                "desc": "约束参数数值"
            })
            logger.debug(f"添加关系: {article_node_id} -[CONSTRAINS]-> {param_id}")

            # 如果指标有适用地点，创建 Location 节点（自动去重）
            if metric.location:
                loc_id = f"LOC_{metric.location}"
                is_new = self._add_node({
                    "id:ID": loc_id,
                    ":LABEL": "Location",
                    "name": metric.location
                })
                if is_new:
                    logger.debug(f"新增Location节点: {loc_id}")
                
                # 关系：参数 -[APPLIES_TO]-> 地点
                self.relationships.append({
                    ":START_ID": param_id,
                    ":END_ID": loc_id,
                    ":TYPE": "APPLIES_TO",
                    "desc": "适用于特定地点"
                })

        # ── 3. 处理逻辑要求：Requirement (具体要求) ─────────────────
        for req in article.requirements:
            req_node_id = f"REQ_{req.id}"
            self._add_node({
                "id:ID": req_node_id,
                ":LABEL": "Requirement",
                "name": req.logic_type,
                "content": req.description
            })

            # 关系：条款 -[SPECIFIES]-> 要求
            self.relationships.append({
                ":START_ID": article_node_id,
                ":END_ID": req_node_id,
                ":TYPE": "SPECIFIES",
                "desc": "规定了具体安全要求"
            })
            logger.debug(f"添加关系: {article_node_id} -[SPECIFIES]-> {req_node_id}")

            # 处理关联设施：Facility (设施)（自动去重）
            for facility in req.associated_facilities:
                fac_id = f"FAC_{facility}"
                is_new = self._add_node({
                    "id:ID": fac_id,
                    ":LABEL": "Facility",
                    "name": facility
                })
                if is_new:
                    logger.debug(f"新增Facility节点: {fac_id}")
                
                # 关系：要求 -[INVOLVES_FACILITY]-> 设施
                self.relationships.append({
                    ":START_ID": req_node_id,
                    ":END_ID": fac_id,
                    ":TYPE": "INVOLVES_FACILITY",
                    "desc": "涉及通风设施"
                })
        
        logger.info(
            f"条款图谱构建完成: {article.article_number} | "
            f"当前节点总数: {len(self.nodes)} | "
            f"当前关系总数: {len(self.relationships)}"
        )

    def export_to_neo4j_csv(self):
        """
        导出为 Neo4j 批量导入格式的 CSV 文件
        
        生成两个文件：
        - nodes.csv:         所有节点（Article/Parameter/Location/Requirement/Facility）
        - relationships.csv: 所有关系（CONSTRAINS/APPLIES_TO/SPECIFIES/INVOLVES_FACILITY）
        """
        logger.info("开始导出Neo4j CSV文件")
        
        if not self.nodes:
            logger.warning("节点列表为空，没有数据可导出")
            return
        
        # 转换为 DataFrame 并去重（防御性去重，正常情况下 _add_node 已处理）
        nodes_df = pd.DataFrame(self.nodes).drop_duplicates(subset=['id:ID'])
        rels_df = pd.DataFrame(self.relationships)

        nodes_path = os.path.join(self.output_dir, "nodes.csv")
        rels_path = os.path.join(self.output_dir, "relationships.csv")

        nodes_df.to_csv(nodes_path, index=False, encoding='utf-8')
        rels_df.to_csv(rels_path, index=False, encoding='utf-8')

        logger.info(f"✅ Neo4j CSV 文件已生成至: {self.output_dir}")
        logger.info(f"📊 节点总数: {len(nodes_df)}，关系总数: {len(rels_df)}")
        print(f"✅ Neo4j CSV 文件已生成至: {self.output_dir}")
        print(f"📊 节点总数: {len(nodes_df)}，关系总数: {len(rels_df)}")

# 使用示例
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    builder = VentilationKGBuilder()
    # 假设这里传入从 Agent 获取的 article 对象
    # builder.process_article(parsed_article)
    # builder.export_to_neo4j_csv()