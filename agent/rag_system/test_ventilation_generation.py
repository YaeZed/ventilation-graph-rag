"""
通风生成模块 (VentilationGenerationModule) 功能测试脚本

验证：
1. 系统 Prompt 构建逻辑 (是否包含 article_name)
2. 同步生成逻辑 (Mock LLM)
3. 提示词格式验证
"""

import logging
from unittest.mock import MagicMock
from langchain_core.documents import Document

# 导入待测模块
from ventilation_generation import VentilationGenerationModule

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def test_generation():
    logger.info(">>> 开始测试 VentilationGenerationModule...\n")
    
    # 1. 准备 Mock 文档
    docs = [
        Document(
            page_content="掘进中的煤巷和半煤岩巷最低风速为 0.25m/s。",
            metadata={"article_name": "第一百五十七条", "retrieval_level": "vector"}
        ),
        Document(
            page_content="局部通风机必须安装在进风巷道中。",
            metadata={"name": "第一百八十三条", "retrieval_level": "entity"}
        )
    ]
    
    # 2. 初始化
    # 强制设置一个 dummy key 避开初始化检查
    import os
    os.environ["DASHSCOPE_API_KEY"] = "sk-test-key"
    
    gen = VentilationGenerationModule()
    
    # 3. 测试 Prompt 构建 (内部私有方法测试)
    logger.info("测试步骤 1: 验证 Prompt 构建逻辑")
    prompt = gen._build_prompt("风速要求是多少？", docs)
    
    # 验证是否正确解析了不同的元数据字段名 (article_name vs name)
    has_art_157 = "第一百五十七条" in prompt
    has_art_183 = "第一百八十三条" in prompt
    has_role = "通风安全专家" in prompt
    
    logger.info(f"包含 157 条款: {has_art_157}")
    logger.info(f"包含 183 条款: {has_art_183}")
    logger.info(f"包含专家角色: {has_role}")
    
    # 4. 测试生成逻辑 (Mock Client)
    logger.info("\n测试步骤 2: 验证同步生成 (Mock)")
    gen.client.chat.completions.create = MagicMock()
    gen.client.chat.completions.create.return_value.choices[0].message.content = "这是一份专业的整改建议..."
    
    answer = gen.generate_adaptive_answer("测试提问", docs)
    logger.info(f"生成回答内容: {answer}")

    logger.info("\n>>> 测试完成！")

if __name__ == "__main__":
    test_generation()
