"""
矿井通风安全规程 - 图索引模块（独立重构版）

实现实体和关系的键值对结构 (K,V)，为混合检索提供图谱支撑。
"""

import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class EntityKeyValue:
    """实体键值对"""
    entity_name: str
    index_keys: List[str]  # 索引键列表
    value_content: str     # 详细描述内容
    entity_type: str       # 实体类型 (Article, Parameter, etc.)
    metadata: Dict[str, Any]

@dataclass 
class RelationKeyValue:
    """关系键值对"""
    relation_id: str
    index_keys: List[str]  # 多个索引键
    value_content: str     # 关系描述内容
    relation_type: str     # 关系类型
    source_entity: str     # 源实体 ID
    target_entity: str     # 目标实体 ID
    metadata: Dict[str, Any]

class VentilationGraphIndexingModule:
    """
    通风规程图索引模块 - 独立版
    负责将 Neo4j 的图结构转化为可被检索的键值对映射。
    """
    
    def __init__(self, config=None, llm_client=None):
        self.config = config
        self.llm_client = llm_client
        
        # 键值对存储
        self.entity_kv_store: Dict[str, EntityKeyValue] = {}
        self.relation_kv_store: Dict[str, RelationKeyValue] = {}
        
        # 索引映射：key -> entity/relation IDs
        self.key_to_entities: Dict[str, List[str]] = defaultdict(list)
        self.key_to_relations: Dict[str, List[str]] = defaultdict(list)
        
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
        
        # 1. 处理条款（Article）
        for art in articles:
            eid = art.node_id
            name = art.name
            props = getattr(art, 'properties', {})
            
            parts = [f"条款名称：{name}"]
            if props.get("title"):
                parts.append(f"主题：{props['title']}")
            if props.get("content"):
                parts.append(f"原文摘要：{props['content'][:200]}...")
                
            keys = [name]
            if props.get("title"):
                keys.append(props["title"])
                
            kv = EntityKeyValue(
                entity_name=name, 
                index_keys=list(set(keys)),
                value_content="\n".join(parts), 
                entity_type="Article",
                metadata={"node_id": eid, "properties": props}
            )
            self._add_entity_to_store(eid, kv)
                
        # 2. 处理安全指标（Parameter）
        for param in parameters:
            eid = param.node_id
            name = param.name
            props = getattr(param, 'properties', {})
            
            parts = [f"安全指标：{name}"]
            v_min, v_max, unit = props.get("value_min"), props.get("value_max"), props.get("unit", "")
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
            self._add_entity_to_store(eid, kv)
                
        # 3. 处理安全要求（Requirement）
        for req in requirements:
            eid = req.node_id
            name = req.name       
            props = getattr(req, 'properties', {})
            parts = [f"要求类型：{name}"]
            if props.get("content"):
                parts.append(f"详细内容：{props['content']}")
                
            keys = [name, "安全要求", "规定", "规程要求"]
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Requirement",
                metadata={"node_id": eid, "properties": props}
            )
            self._add_entity_to_store(eid, kv)
                
        # 4. 处理通风设施（Facility）
        for fac in facilities:
            eid = fac.node_id
            name = fac.name
            props = getattr(fac, 'properties', {})
            parts = [f"通风设施：{name}"]
            keys = [name, "设施", "设备", "通风装置", "机械设备"]
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Facility",
                metadata={"node_id": eid, "properties": props}
            )
            self._add_entity_to_store(eid, kv)
                
        # 5. 处理适用地点（Location）
        for loc in locations:
            eid = loc.node_id
            name = loc.name
            props = getattr(loc, 'properties', {})
            parts = [f"适用地点：{name}"]
            keys = [name, "地点", "场所", "工作面", "巷道"]
            kv = EntityKeyValue(
                entity_name=name, index_keys=list(set(keys)),
                value_content="\n".join(parts), entity_type="Location",
                metadata={"node_id": eid, "properties": props}
            )
            self._add_entity_to_store(eid, kv)
                
        logger.info(f"实体键值对创建完成：{len(self.entity_kv_store)} 个")
        return self.entity_kv_store

    def _add_entity_to_store(self, eid: str, kv: EntityKeyValue):
        """辅助方法：添加实体并建立键映射"""
        self.entity_kv_store[eid] = kv
        for k in kv.index_keys:
            self.key_to_entities[k].append(eid)

    def create_relation_key_values(self, relationships: List[Tuple[str, str, str]]) -> Dict[str, RelationKeyValue]:
        """为通风规程逻辑关系创建键值对"""
        logger.info("开始创建通风规程关系键值对...")
        
        for i, (source_id, relation_type, target_id) in enumerate(relationships):
            relation_id = f"v_rel_{i}_{source_id}_{target_id}"
            source_entity = self.entity_kv_store.get(source_id)
            target_entity = self.entity_kv_store.get(target_id)
            
            if not source_entity or not target_entity:
                continue
            
            content_parts = [
                f"关系类型: {relation_type}",
                f"源项: {source_entity.entity_name} ({source_entity.entity_type})",
                f"关联项: {target_entity.entity_name} ({target_entity.entity_type})"
            ]
            
            index_keys = self._generate_relation_index_keys(source_entity, target_entity, relation_type)
            
            relation_kv = RelationKeyValue(
                relation_id=relation_id,
                index_keys=index_keys,
                value_content='\n'.join(content_parts),
                relation_type=relation_type,
                source_entity=source_id,
                target_entity=target_id,
                metadata={"source_name": source_entity.entity_name, "target_name": target_entity.entity_name}
            )
            
            self.relation_kv_store[relation_id] = relation_kv
            for key in index_keys:
                self.key_to_relations[key].append(relation_id)
        
        return self.relation_kv_store

    def _generate_relation_index_keys(
        self, source_entity: EntityKeyValue, target_entity: EntityKeyValue, relation_type: str
    ) -> List[str]:
        """生成通风领域特定的关系主题索引词"""
        keys = [relation_type]
        mapping = {
            "CONSTRAINS":        ["限值要求", "安全参数", source_entity.entity_name, target_entity.entity_name],
            "APPLIES_TO":        ["适用地点", "应用范围", source_entity.entity_name, target_entity.entity_name],
            "SPECIFIES":         ["安全规定", "操作规范", source_entity.entity_name],
            "INVOLVES_FACILITY": ["涉及设施", "设备配备", target_entity.entity_name],
            "REFERENCES":        ["引用条款", "关联规程"],
        }
        keys.extend(mapping.get(relation_type, [source_entity.entity_name, target_entity.entity_name]))
        
        if getattr(self.config, 'enable_llm_relation_keys', False):
            enhanced = self._llm_enhance_relation_keys(source_entity, target_entity, relation_type)
            keys.extend(enhanced)
            
        return list(set(keys))

    def _llm_enhance_relation_keys(self, source, target, rel_type) -> List[str]:
        """使用 LLM 增强关系的主题词（通风安全语境）"""
        prompt = f"作为矿井通风专家，分析以下关系并给出3个主题关键词：\n源：{source.entity_name}\n关系：{rel_type}\n目标：{target.entity_name}\n返回JSON: {{\"keywords\": []}}"
        try:
            response = self.llm_client.chat.completions.create(
                model=getattr(self.config, 'llm_model', 'qwen-plus'),
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.choices[0].message.content.strip()).get("keywords", [])
        except:
            return []

    def deduplicate_entities_and_relations(self):
        """去重逻辑：基于名称合并实体描述"""
        logger.info("执行实体/关系去重...")
        # 实体去重
        name_map = defaultdict(list)
        for eid, kv in self.entity_kv_store.items():
            name_map[kv.entity_name].append(eid)
        
        for ids in name_map.values():
            if len(ids) > 1:
                main_id = ids[0]
                for extra_id in ids[1:]:
                    self.entity_kv_store[main_id].value_content += f"\n补充：{self.entity_kv_store[extra_id].value_content}"
                    del self.entity_kv_store[extra_id]

        # 重映射索引
        self._rebuild_key_mappings()

    def _rebuild_key_mappings(self):
        """重建倒排索引"""
        self.key_to_entities.clear()
        self.key_to_relations.clear()
        for eid, kv in self.entity_kv_store.items():
            for k in kv.index_keys: self.key_to_entities[k].append(eid)
        for rid, kv in self.relation_kv_store.items():
            for k in kv.index_keys: self.key_to_relations[k].append(rid)

    def get_entities_by_key(self, key: str) -> List[EntityKeyValue]:
        return [self.entity_kv_store[eid] for eid in self.key_to_entities.get(key, []) if eid in self.entity_kv_store]

    def get_relations_by_key(self, key: str) -> List[RelationKeyValue]:
        return [self.relation_kv_store[rid] for rid in self.key_to_relations.get(key, []) if rid in self.relation_kv_store]

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_entities": len(self.entity_kv_store),
            "total_relations": len(self.relation_kv_store),
            "entity_types": { t: sum(1 for kv in self.entity_kv_store.values() if kv.entity_type == t) 
                            for t in ("Article", "Parameter", "Requirement", "Facility", "Location") }
        }