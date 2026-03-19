"""
矿井通风安全规程 - 答案生成模块

负责接收检索到的 Document 列表，并调用 LLM 生成专业、准确的规程解答。
"""

import os
import logging
import time
from typing import List, Generator, Union
from openai import OpenAI
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class VentilationGenerationModule:
    """通风安全规程答案生成模块"""

    SYSTEM_ROLE = "矿井通风安全专家"

    def __init__(self, model_name: str = "qwen-plus", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("环境变量 DASHSCOPE_API_KEY 未设置，生成功能可能无法使用")

        # 初始化 OpenAI 兼容客户端
        # 注意：此处假设用户使用的是 DashScope (Qwen) 兼容接口
        self.client = OpenAI(
            api_key=api_key or "sk-dummy",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        logger.info(f"通风生成模块初始化成功，模型: {model_name}")

    def _build_prompt(self, question: str, documents: List[Document]) -> str:
        """构建通风安全规程专用 Prompt"""
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if not content:
                continue

            # 提取元数据：支持重构后的字段名
            name = doc.metadata.get("article_name") or doc.metadata.get("name") or "未知条款"
            level = doc.metadata.get("retrieval_level", "unknown").upper()

            context_parts.append(f"【参考条款：{name} | 检索方式：{level}】\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        return f"""你是一位专业的{self.SYSTEM_ROLE}，熟悉《煤矿安全规程》及相关生产安全标准。

【严格约束】
1. 只能依据下方【参考规程内容】回答，不得凭空引用未在检索内容中出现的条款或数值。
2. 如果检索内容中有【规程附件：技术参数对照表】，则答案所引用的一切数值必须直接出自该表，
   严禁使用"虽未在检索内容中完整呈现"、"推测"、"据规程记忆"等推断性措辞。
3. 若检索结果确实不含某具体数值，请明确说明"当前检索结果未包含该参数，建议查阅完整版《煤矿安全规程》"，
   不得自行推断或补充数据。

【参考规程内容】
{context}

【用户提问】
{question}

请按以下格式回答：
1. **核合性结论**：明确结论（合规/违规/数值限值）
2. **规程依据**：列出引用的具体条款编号和原文关键句（数值必须标注来源）
3. **专家解析**：结合现场实际做专业解释
4. **管理建议**（如有）：给出预防或整改建议

回答："""

    def generate_adaptive_answer(self, question: str, documents: List[Document]) -> str:
        """同步生成答案"""
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
            logger.error(f"生成失败: {e}")
            return f"抱歉，系统生成解答时出现故障：{str(e)}"

    def generate_adaptive_answer_stream(
        self, question: str, documents: List[Document], max_retries: int = 3
    ) -> Generator[str, None, None]:
        """流式生成答案（带断线重试机制）"""
        prompt = self._build_prompt(question, documents)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # 成功完成

            except Exception as e:
                logger.warning(f"流式生成第 {attempt+1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    yield f"\n[系统提示] 生成中断，请稍后重试。错误: {str(e)}"
