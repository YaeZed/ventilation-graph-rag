"""
矿井通风安全规程 - 答案生成模块

继承 GenerationIntegrationModule，将 system prompt 和专业角色
替换为矿井通风安全专家。
"""

import sys
import os
import logging
from typing import List

from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from rag_modules.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


class VentilationGenerationModule(GenerationIntegrationModule):
    """
    通风安全规程答案生成模块
    覆盖 generate_adaptive_answer / generate_adaptive_answer_stream
    中的提示词，使用通风安全专家角色。
    """

    SYSTEM_ROLE = "矿井通风安全专家"

    def _build_prompt(self, question: str, documents: List[Document]) -> str:
        """构建通风规程专用提示词"""
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if not content:
                continue
            level = doc.metadata.get('retrieval_level', '')
            prefix = f"[{level.upper()}] " if level else ""
            context_parts.append(f"{prefix}{content}")

        context = "\n\n---\n\n".join(context_parts)

        return f"""你是一位专业的{self.SYSTEM_ROLE}，熟悉《煤矿安全规程》及相关通风安全标准。
请严格依据以下检索到的规程内容回答用户的问题，不得凭空杜撰条款内容。

【检索到的规程内容】
{context}

【用户问题】
{question}

请按以下格式回答：
1. **直接结论**：给出明确的是/否或数值结论
2. **依据条款**：列出引用的具体条款编号和原文关键句
3. **分析说明**：结合场景做必要说明
4. **整改建议**（如涉及违规）：给出具体可操作的整改方向

如果检索内容不足以回答问题，请明确说明"当前规程知识库中暂无相关条款"。

回答："""

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """覆盖父类方法，使用通风规程专业提示词"""
        prompt = self._build_prompt(question, documents)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def generate_adaptive_answer_stream(
        self, question: str, documents: List[Document], max_retries: int = 3
    ):
        """覆盖父类方法，使用通风规程专业提示词（流式）"""
        import time

        prompt = self._build_prompt(question, documents)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=60,
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return

            except Exception as e:
                logger.warning(f"流式生成第 {attempt+1} 次失败: {e}")
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 2
                    print(f"⚠️ 连接中断，{wait}秒后重试...")
                    time.sleep(wait)
                else:
                    # 降级为非流式
                    try:
                        yield self.generate_adaptive_answer(question, documents)
                    except Exception as fe:
                        yield f"抱歉，生成回答时出现网络错误，请稍后重试。错误：{str(fe)}"
