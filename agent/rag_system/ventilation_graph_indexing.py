"""
矿井通风安全规程 - 图索引模块（继承实现）

继承自 cooking 版本的 GraphIndexingModule。
重写实体创建、关系主题词生成以及统计功能，
从而复用通用的去重、倒排索引构建和查询逻辑。
"""
import logging
from typing import Dict, List, Tuple, Any
from collections import defaultdict

from rag_modules.graph_indexing import GraphIndexingModule, EntityKeyValue, RelationKeyValue

logger = logging.getLogger(__name__)

class VentilationGraphIndexingModule(GraphIndexingModule):
    """
    通风规程图索引模块
    继承了父类的：
        - create_relation_key_values
        - deduplicate_entities_and_relations
        - _rebuild_key_mappings
        - get_entities_by_key / get_relations_by_key
    """
    
    def __init__(self, config=None, llm_client=None):
        super().__init__(config, llm_client)

    # ──────────────────────────────────────────────────────────
    # 1. 覆盖实体键值对创建
    # ──────────────────────────────────────────────────────────
    def create_entity_key_values(
        self,
        articles:     List[Any],
        parameters:   List[Any],
        requirements: List[Any],
        facilities:   List[Any],
        locations:    List[Any],
    ) -> Dict[str, EntityKeyValue]:
        """为五类通风规程节点批量创建键值对"""
        logger.info("开始创建通风规程实体键值对...")
        
        # ── 条款（Article）────────────────────────────────────
        for art in articles:
            eid   = art.node_id
            name  = art.name
            props = art.properties
            parts = [f"条款编号：{name}"]
            
            if props.get("title"):
                parts.append(f"主题：{props['title']}")
            if props.get("content"):
                parts.append(f"原文摘要：{props['content'][:100]}...")
                
            keys = [name]
            if props.get("title"):
                keys.append(props["title"])
                
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Article",
                metadata={"node_id": eid, "properties": props}
            )
            self.entity_kv_store[eid] = kv
            for k in kv.index_keys:
                self.key_to_entities[k].append(eid)
                
        # ── 安全指标（Parameter）──────────────────────────────
        for param in parameters:
            eid   = param.node_id
            name  = param.name
            props = param.properties
            parts = [f"安全指标：{name}"]
            
            v_min = props.get("value_min")
            v_max = props.get("value_max")
            unit  = props.get("unit", "")
            if v_min is not None and v_min != "":
                parts.append(f"下限：≥{v_min} {unit}")
            if v_max is not None and v_max != "":
                parts.append(f"上限：≤{v_max} {unit}")
                
            keys = [name, "安全指标", "参数", "限值"]
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Parameter",
                metadata={"node_id": eid, "properties": props}
            )
            self.entity_kv_store[eid] = kv
            for k in kv.index_keys:
                self.key_to_entities[k].append(eid)
                
        # ── 安全要求（Requirement）────────────────────────────
        for req in requirements:
            eid   = req.node_id
            name  = req.name       
            props = req.properties
            parts = [f"要求类型：{name}"]
            
            if props.get("content"):
                parts.append(f"内容：{props['content']}")
                
            keys = [name, "安全要求", "规定", "规范"]
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Requirement",
                metadata={"node_id": eid, "properties": props}
            )
            self.entity_kv_store[eid] = kv
            for k in kv.index_keys:
                self.key_to_entities[k].append(eid)
                
        # ── 通风设施（Facility）───────────────────────────────
        for fac in facilities:
            eid   = fac.node_id
            name  = fac.name
            props = fac.properties
            parts = [f"通风设施：{name}"]
            
            keys  = [name, "设施", "设备", "通风装置"]
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Facility",
                metadata={"node_id": eid, "properties": props}
            )
            self.entity_kv_store[eid] = kv
            for k in kv.index_keys:
                self.key_to_entities[k].append(eid)
                
        # ── 适用地点（Location）───────────────────────────────
        for loc in locations:
            eid   = loc.node_id
            name  = loc.name
            props = loc.properties
            parts = [f"适用地点：{name}"]
            
            keys  = [name, "地点", "场所", "工作面", "巷道"]
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Location",
                metadata={"node_id": eid, "properties": props}
            )
            self.entity_kv_store[eid] = kv
            for k in kv.index_keys:
                self.key_to_entities[k].append(eid)
                
        logger.info(f"实体键值对创建完成：{len(self.entity_kv_store)} 个")
        return self.entity_kv_store

    # ──────────────────────────────────────────────────────────
    # 2. 覆盖关系主题词生成
    # ──────────────────────────────────────────────────────────
    def _generate_relation_index_keys(
        self,
        source_entity: EntityKeyValue,
        target_entity: EntityKeyValue,
        relation_type: str,
    ) -> List[str]:
        """根据关系类型生成通风领域主题索引词"""
        keys = [relation_type]
        
        mapping = {
            "CONSTRAINS":        ["限值要求", "参数约束", "数值规定", source_entity.entity_name, target_entity.entity_name],
            "APPLIES_TO":        ["适用范围", "适用地点", "使用场所", source_entity.entity_name, target_entity.entity_name],
            "SPECIFIES":         ["安全要求", "操作规范", "规定事项", source_entity.entity_name],
            "INVOLVES_FACILITY": ["涉及设施", "使用设备", "配备要求", target_entity.entity_name],
            "REFERENCES":        ["引用条款", "参考规定", "关联规程"],
            "RELATED_TO":        ["相关条款", "关联内容", "同类规定"],
        }
        
        keys.extend(mapping.get(relation_type, [source_entity.entity_name, target_entity.entity_name]))
        
        # 视情况调用父类的 LLM 增强
        if getattr(self.config, 'enable_llm_relation_keys', False):
            enhanced_keys = self._llm_enhance_relation_keys(source_entity, target_entity, relation_type)
            keys.extend(enhanced_keys)
            
        return list(set(keys))

    # ──────────────────────────────────────────────────────────
    # 3. 覆盖统计方法
    # ──────────────────────────────────────────────────────────
    def get_statistics(self) -> Dict[str, Any]:
        """通风专用的统计字段"""
        return {
            "total_entities":   len(self.entity_kv_store),
            "total_relations":  len(self.relation_kv_store),
            "total_entity_keys": sum(len(kv.index_keys) for kv in self.entity_kv_store.values()),
            "total_relation_keys": sum(len(kv.index_keys) for kv in self.relation_kv_store.values()),
            "entity_types": {
                t: sum(1 for kv in self.entity_kv_store.values() if kv.entity_type == t)
                for t in ("Article", "Parameter", "Requirement", "Facility", "Location")
            }
        }