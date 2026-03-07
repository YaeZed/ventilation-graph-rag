import json
import logging
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 数据类定义（用@dataclass装饰器自动生成__init__等方法）
# ============================================================

@dataclass
class SafetyMetric:
    """安全指标数据类，存储条款中的数值型参数"""
    id: str                              # 指标唯一ID，如"第一百五十六条-M1"
    name: str                            # 指标名称，如"风速"
    threshold_min: Optional[float] = None  # 最小阈值（Optional表示可以为None）
    threshold_max: Optional[float] = None  # 最大阈值
    unit: str = ""                       # 单位，如"m/s"
    location: str = ""                   # 适用地点，如"采煤工作面"

@dataclass
class SafetyRequirement:
    """安全要求数据类，存储条款中的逻辑性要求"""
    id: str                                              # 要求唯一ID
    description: str                                     # 要求描述原文
    logic_type: str                                      # 分类：数值约束/设施配置/禁止行为
    associated_facilities: List[str] = field(default_factory=list)  # 关联设施列表

@dataclass
class RegulationArticle:
    """规程条款数据类，聚合一条完整的规程信息"""
    article_number: str                                  # 条款编号，如"第一百五十六条"
    title: str                                           # 主题摘要
    content: str                                         # 原文内容
    metrics: List[SafetyMetric] = field(default_factory=list)        # 安全指标列表
    requirements: List[SafetyRequirement] = field(default_factory=list)  # 安全要求列表


# ============================================================
# 通风安全Agent类
# ============================================================

class VentilationSafetyAgent:
    """
    通风安全知识提取Agent
    
    使用LLM将非结构化的规程条文转化为结构化的知识对象
    """
    
    def __init__(self, api_key: str, base_url: str):
        """
        初始化Agent
        
        Args:
            api_key: API密钥（通义千问/OpenAI兼容接口）
            base_url: API基础URL
        """
        logger.info(f"初始化VentilationSafetyAgent，base_url: {base_url}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.system_prompt = """你现在是一名精通 2025 版《煤矿安全规程》的矿井通风安全专家。
你的任务是将非结构化的规程条文转化为计算机可理解的逻辑知识库。

### 核心解析规则：
1. **识别表格**：识别 Markdown 表格并提取数值。
2. **提取实体**：提取指标、要求及物理设施。
3. **逻辑分类**：区分数值约束、设施配置及禁止行为。

### 必须返回的标准 JSON 格式（严格按此结构）：
{
    "article_number": "例如：第一百五十七条",
    "title": "主题摘要（简短描述条款核心内容）",
    "metrics": [
        {
            "name": "指标名称（如：最低风速）",
            "threshold_min": 最小数值或null,
            "threshold_max": 最大数值或null,
            "unit": "单位（如：m/s）",
            "location": "适用地点（如：采煤工作面）"
        }
    ],
    "requirements": [
        {
            "description": "规定原文",
            "logic_type": "数值约束 或 设施配置 或 禁止行为",
            "associated_facilities": ["相关设施名称"]
        }
    ]
}"""

    def extract_logic(self, content: str) -> Optional[RegulationArticle]:
        """
        使用LLM从条款文本中提取结构化知识
        
        Args:
            content: 条款原文（包含表格的Markdown文本）
        
        Returns:
            RegulationArticle对象，解析失败时返回None
        """
        # 函数入口：记录正在处理的条款（只取前20字符作为标识）
        article_preview = content[:20].replace('\n', ' ')
        logger.info(f"开始解析条款: {article_preview}...")
        
        try:
            # DEBUG：记录API调用
            logger.debug(f"调用LLM API，内容长度: {len(content)} 字符")
            
            response = self.client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"请解析以下条文：\n{content}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            # DEBUG：记录API响应
            logger.debug("LLM API调用成功，开始解析JSON")
            
            data = json.loads(response.choices[0].message.content)
            art_num = data.get("article_number", "未知条")
            
            logger.debug(f"JSON解析成功，条款编号: {art_num}")
            
            # 构建RegulationArticle对象
            article = RegulationArticle(
                article_number=art_num,
                title=data.get("title", ""),
                content=content
            )
            
            # 动态生成 Metric ID 并填充
            metrics_data = data.get("metrics", [])
            for i, m in enumerate(metrics_data):
                metric_obj = SafetyMetric(
                    id=f"{art_num}-M{i+1}",  # 自动生成 ID，如"第一百五十六条-M1"
                    name=m.get("name", ""),
                    threshold_min=m.get("threshold_min"),
                    threshold_max=m.get("threshold_max"),
                    unit=m.get("unit", ""),
                    location=m.get("location", "")
                )
                article.metrics.append(metric_obj)
            
            # 动态生成 Requirement ID 并填充
            requirements_data = data.get("requirements", [])
            for i, r in enumerate(requirements_data):
                req_obj = SafetyRequirement(
                    id=f"{art_num}-R{i+1}",  # 自动生成 ID
                    description=r.get("description", ""),
                    logic_type=r.get("logic_type", ""),
                    associated_facilities=r.get("associated_facilities", [])
                )
                article.requirements.append(req_obj)
            
            # 函数完成：INFO，记录解析结果摘要
            logger.info(
                f"条款解析完成: {art_num} | "
                f"指标数: {len(article.metrics)} | "
                f"要求数: {len(article.requirements)}"
            )
            return article
        
        except json.JSONDecodeError as e:
            # JSON解析失败：LLM返回了非法的JSON格式
            logger.error(f"JSON解析失败，条款: {article_preview}... 错误: {e}")
            return None
        except Exception as e:
            # 其他异常：API调用失败、网络问题等
            logger.error(f"条款解析失败: {type(e).__name__} - {e}")
            logger.exception("详细错误堆栈:")
            return None