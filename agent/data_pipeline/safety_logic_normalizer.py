#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全逻辑标准化工具 - 处理矿井通风规程中的数值约束表达
"""

import re
from typing import Tuple, Optional, Dict

class SafetyLogicNormalizer:
    """安全约束标准化器"""
    
    def __init__(self):
        # 逻辑算子映射表：将规程术语映射为数学符号 [cite: 158, 160]
        self.logic_mappings = {
            # 上限约束
            "不得超过": "<=",
            "不超过": "<=",
            "上限为": "<=",
            "最大为": "<=",
            "小于": "<",
            
            # 下限约束
            "不低于": ">=",
            "不得低于": ">=",
            "在...以上": ">",
            "最小为": ">=",
            "必须在": ">",
            
            # 禁止类逻辑 [cite: 46, 50]
            "严禁": "FORBIDDEN",
            "禁止": "FORBIDDEN",
            "不得": "FORBIDDEN"
        }
        
        # 矿井特有单位标准化 [cite: 6, 10, 15]
        self.unit_mappings = {
            "％": "%",
            "百分之": "%",
            "m/s": "m/s",
            "m.s-1": "m/s",
            "m³/min": "m3/min",
            "m³/min.kW": "m3/min/kW",
            "℃": "°C",
            "度": "°C"
        }

    def parse_constraint(self, text: str) -> Dict:
        """
        解析规程条文中的数值约束
        示例输入: "氧气浓度不低于20％" -> {"value": 20.0, "operator": ">=", "unit": "%"} [cite: 3]
        """
        result = {
            "value": None,
            "operator": "==",
            "unit": "",
            "raw_text": text
        }
        
        # 1. 提取数值（支持小数）
        value_match = re.search(r'(\d+(\.\d+)?)', text)
        if value_match:
            result["value"] = float(value_match.group(1))
            
        # 2. 匹配逻辑算子
        for key, op in self.logic_mappings.items():
            if key in text:
                result["operator"] = op
                break
                
        # 3. 匹配并标准化单位 [cite: 6, 10]
        for key, unit in self.unit_mappings.items():
            if key in text:
                result["unit"] = unit
                break
                
        return result

    def check_compliance(self, measured_value: float, constraint: Dict) -> Tuple[bool, str]:
        """
        执行合规性判定逻辑 [cite: 160, 162]
        """
        val = constraint["value"]
        op = constraint["operator"]
        
        if val is None:
            return True, "无数值约束，需人工判定"
            
        # 逻辑判定引擎
        is_compliant = True
        if op == "<=":
            is_compliant = measured_value <= val
        elif op == ">=":
            is_compliant = measured_value >= val
        elif op == "<":
            is_compliant = measured_value < val
        elif op == ">":
            is_compliant = measured_value > val
            
        status = "合规" if is_compliant else "隐患"
        detail = f"实测值 {measured_value} {status} (规程要求 {op} {val})"
        
        return is_compliant, detail

# 演示功能
if __name__ == "__main__":
    normalizer = SafetyLogicNormalizer()
    
    # 测试案例 1：氧气浓度 [cite: 3]
    print(normalizer.parse_constraint("氧气浓度不低于20％")) 
    
    # 测试案例 2：风速 
    print(normalizer.parse_constraint("最高风速不得超过4m/s"))
    
    # 测试案例 3：温度 
    print(normalizer.parse_constraint("必须在2℃以上"))