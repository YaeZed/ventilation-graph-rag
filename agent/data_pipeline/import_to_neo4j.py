#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import_to_neo4j.py
将 safety_output/ 目录下的 nodes.csv 和 relationships.csv 导入 Neo4j

使用方法:
    python import_to_neo4j.py

前提:
    - Neo4j 已通过 docker-compose up -d 启动
    - pip install neo4j pandas python-dotenv
"""

import os
import logging
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

# ── 日志配置 ───────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── 常量配置 ───────────────────────────────────────────
# CSV 文件路径（相对于本脚本的位置）
NODES_CSV        = "safety_output/nodes.csv"
RELATIONSHIPS_CSV = "safety_output/relationships.csv"

# Neo4j 连接信息（与 docker-compose.yml 保持一致）
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "160722yaesakura"


def create_constraints(session):
    """
    创建唯一约束（相当于给 id:ID 列建索引，加快 MERGE 速度）
    IF NOT EXISTS 保证重复运行不会报错
    """
    logger.info("创建唯一约束...")
    constraints = [
        "CREATE CONSTRAINT art_id IF NOT EXISTS FOR (n:Article)     REQUIRE n.node_id IS UNIQUE",
        "CREATE CONSTRAINT par_id IF NOT EXISTS FOR (n:Parameter)   REQUIRE n.node_id IS UNIQUE",
        "CREATE CONSTRAINT loc_id IF NOT EXISTS FOR (n:Location)    REQUIRE n.node_id IS UNIQUE",
        "CREATE CONSTRAINT req_id IF NOT EXISTS FOR (n:Requirement) REQUIRE n.node_id IS UNIQUE",
        "CREATE CONSTRAINT fac_id IF NOT EXISTS FOR (n:Facility)    REQUIRE n.node_id IS UNIQUE",
    ]
    for cypher in constraints:
        session.run(cypher)
    logger.info("唯一约束创建完成")


def import_nodes(session, df: pd.DataFrame):
    """
    批量导入节点
    
    nodes.csv 的列：
        id:ID  :LABEL  name  title  content  value_min  value_max  unit
    """
    logger.info(f"开始导入节点，共 {len(df)} 条...")
    
    success = 0
    fail    = 0
    
    for _, row in df.iterrows():
        node_id = row["id:ID"]
        label   = row[":LABEL"]
        
        # 构建属性字典（NaN 转为 None，Neo4j 不支持 NaN）
        props = {
            "node_id":   node_id,
            "name":      _safe(row.get("name")),
            "title":     _safe(row.get("title")),
            "content":   _safe(row.get("content")),
            "value_min": _safe_float(row.get("value_min")),
            "value_max": _safe_float(row.get("value_max")),
            "unit":      _safe(row.get("unit")),
        }
        
        # 用 apoc.merge.node 或手动拼接 Cypher（这里用 f-string 拼接 label）
        # 注意：label 不能用参数传递，必须直接写在 Cypher 里
        cypher = f"""
            MERGE (n:{label} {{node_id: $node_id}})
            SET n += $props
        """
        
        try:
            session.run(cypher, node_id=node_id, props=props)
            success += 1
        except Exception as e:
            logger.warning(f"节点导入失败: {node_id} | 错误: {e}")
            fail += 1
    
    logger.info(f"节点导入完成 ✅  成功: {success}，失败: {fail}")


def import_relationships(session, df: pd.DataFrame):
    """
    批量导入关系
    
    relationships.csv 的列：
        :START_ID  :END_ID  :TYPE  desc
    """
    logger.info(f"开始导入关系，共 {len(df)} 条...")
    
    success = 0
    fail    = 0
    
    for _, row in df.iterrows():
        start_id = row[":START_ID"]
        end_id   = row[":END_ID"]
        rel_type = row[":TYPE"]
        desc     = _safe(row.get("desc"))
        
        # 关系类型同样不能用参数传递
        cypher = f"""
            MATCH (a {{node_id: $start_id}})
            MATCH (b {{node_id: $end_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.desc = $desc
        """
        
        try:
            session.run(cypher, start_id=start_id, end_id=end_id, desc=desc)
            success += 1
        except Exception as e:
            logger.warning(f"关系导入失败: {start_id} -[{rel_type}]-> {end_id} | 错误: {e}")
            fail += 1
    
    logger.info(f"关系导入完成 ✅  成功: {success}，失败: {fail}")


def print_stats(session):
    """打印导入后的图谱统计信息"""
    print("\n" + "=" * 55)
    print("📊 Neo4j 图谱统计")
    print("=" * 55)
    
    # 节点统计
    result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
    print(f"\n{'节点类型':<20} {'数量':>8}")
    print("-" * 30)
    total_nodes = 0
    for record in result:
        print(f"{record['label']:<20} {record['cnt']:>8}")
        total_nodes += record['cnt']
    print(f"{'合计':<20} {total_nodes:>8}")
    
    # 关系统计
    result = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS cnt ORDER BY cnt DESC")
    print(f"\n{'关系类型':<30} {'数量':>8}")
    print("-" * 40)
    total_rels = 0
    for record in result:
        print(f"{record['rel_type']:<30} {record['cnt']:>8}")
        total_rels += record['cnt']
    print(f"{'合计':<30} {total_rels:>8}")
    print("=" * 55)


# ── 工具函数 ───────────────────────────────────────────
def _safe(val):
    """将 NaN/None 转为 None（Neo4j 可接受），否则转 str"""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def _safe_float(val):
    """将字符串/NaN 转为 float 或 None"""
    if val is None:
        return None
    if isinstance(val, float):
        return None if pd.isna(val) else val
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def build_article_relations(session):
    """
    在共享同一设施(Facility)或地点(Location)的条款之间建立 RELATED_TO 关系
    
    逻辑：
        Article -[SPECIFIES]-> Requirement -[INVOLVES_FACILITY]-> Facility
        如果两个条款共享同一个 Facility，则用 RELATED_TO 连接
        同理适用于共享 Location 的条款
    """
    logger.info("建立条款间 RELATED_TO 关系...")
    
    # 基于共享 Facility 建立关联
    cypher_facility = """
        MATCH (a1:Article)-[:SPECIFIES]->(:Requirement)-[:INVOLVES_FACILITY]->(f:Facility)
              <-[:INVOLVES_FACILITY]-(:Requirement)<-[:SPECIFIES]-(a2:Article)
        WHERE a1.node_id < a2.node_id
        MERGE (a1)-[r:RELATED_TO {basis: 'shared_facility', via: f.name}]->(a2)
        RETURN count(r) AS created
    """
    result = session.run(cypher_facility)
    fac_count = result.single()["created"]
    logger.info(f"  基于共享设施建立关系: {fac_count} 条")

    # 基于共享 Location 建立关联
    cypher_location = """
        MATCH (a1:Article)-[:CONSTRAINS]->(:Parameter)-[:APPLIES_TO]->(l:Location)
              <-[:APPLIES_TO]-(:Parameter)<-[:CONSTRAINS]-(a2:Article)
        WHERE a1.node_id < a2.node_id
        MERGE (a1)-[r:RELATED_TO {basis: 'shared_location', via: l.name}]->(a2)
        RETURN count(r) AS created
    """
    result = session.run(cypher_location)
    loc_count = result.single()["created"]
    logger.info(f"  基于共享地点建立关系: {loc_count} 条")
    
    logger.info(f"条款间关联建立完成，共新增 RELATED_TO 关系: {fac_count + loc_count} 条")


def extract_references(session):
    """
    扫描每个条款的 content 字段，用正则提取对其他条款的显式引用，
    建立 REFERENCES 关系。
    
    示例：
        第一百五十七条 content 包含"符合本规程第一百五十六条规定"
        → 第一百五十七条 -[REFERENCES]-> 第一百五十六条
    """
    import re
    
    logger.info("扫描条款内容，提取显式引用关系...")
    
    # 1. 读取所有条款的 node_id 和 content
    result = session.run("MATCH (a:Article) RETURN a.node_id AS id, a.content AS content, a.name AS name")
    articles = [(r["id"], r["name"], r["content"] or "") for r in result]
    
    # 中文数字条款编号正则：匹配"第X条"格式
    # 例：第一百五十六条、第一百九十二条
    ref_pattern = re.compile(r'第[一二三四五六七八九十百千]+条')
    
    ref_count = 0
    for node_id, name, content in articles:
        # 找到这个条款里引用的所有其他条款名
        found = set(ref_pattern.findall(content))
        
        # 排除自引用
        found.discard(name)
        
        for ref_name in found:
            # 确认被引用的条款确实存在于图谱中，再建立关系
            cypher = """
                MATCH (src:Article {node_id: $src_id})
                MATCH (tgt:Article {name: $tgt_name})
                MERGE (src)-[r:REFERENCES {type: 'explicit_citation'}]->(tgt)
                RETURN count(r) AS created
            """
            res = session.run(cypher, src_id=node_id, tgt_name=ref_name)
            created = res.single()
            if created and created["created"] > 0:
                logger.debug(f"  {name} -[REFERENCES]-> {ref_name}")
                ref_count += 1
    
    logger.info(f"显式引用关系建立完成，共新增 REFERENCES 关系: {ref_count} 条")
    return ref_count


def clear_database(session):
    """清空数据库中所有节点和关系"""
    logger.info("清空数据库...")
    result = session.run("MATCH (n) DETACH DELETE n")
    summary = result.consume()
    logger.info(f"已删除节点: {summary.counters.nodes_deleted}，关系: {summary.counters.relationships_deleted}")


# ── 主程序 ────────────────────────────────────────────
def main():
    # 加载 .env（兼容从任意目录运行）
    load_dotenv(dotenv_path="../../../.env")
    
    logger.info("=" * 55)
    logger.info("🚀 Neo4j 导入程序启动")
    logger.info("=" * 55)
    
    # 1. 读取 CSV
    logger.info(f"读取节点文件: {NODES_CSV}")
    nodes_df = pd.read_csv(NODES_CSV, dtype=str)          # 全部读为字符串，保留原始值
    logger.info(f"读取关系文件: {RELATIONSHIPS_CSV}")
    rels_df  = pd.read_csv(RELATIONSHIPS_CSV, dtype=str)
    
    logger.info(f"节点数: {len(nodes_df)}，关系数: {len(rels_df)}")
    
    # 2. 连接 Neo4j
    logger.info(f"连接 Neo4j: {NEO4J_URI}")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # 3. 清空旧数据
        clear_database(session)
        
        # 4. 创建约束
        create_constraints(session)
        
        # 5. 导入节点
        import_nodes(session, nodes_df)
        
        # 6. 导入关系
        import_relationships(session, rels_df)
        
        # 7. 建立条款间 RELATED_TO 关联
        build_article_relations(session)
        
        # 8. 建立显式 REFERENCES 引用关系
        extract_references(session)
        
        # 9. 打印统计
        print_stats(session)
    
    driver.close()
    logger.info("✅ 导入完成！")
    print(f"\n🌐 打开 Neo4j Browser 查看图谱: http://localhost:7474")
    print(f"   用户名: {NEO4J_USER}  密码: {NEO4J_PASSWORD}")
    print(f"\n💡 在 Browser 中输入以下 Cypher 查看数据:")
    print("   MATCH (n) RETURN n LIMIT 50")


if __name__ == "__main__":
    main()
