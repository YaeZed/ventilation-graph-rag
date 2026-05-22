"""Answer generation for ventilation safety GraphRAG."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Generator

from langchain_core.documents import Document
from openai import OpenAI

logger = logging.getLogger(__name__)


class VentilationGenerationModule:
    """Generate text and image-grounded ventilation safety answers."""

    SYSTEM_ROLE = "矿井通风安全专家"

    def __init__(self, model_name: str = "qwen-plus", temperature: float = 0.1, max_tokens: int = 2048):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("环境变量 DASHSCOPE_API_KEY 未设置，生成功能可能无法使用")

        self.client = OpenAI(
            api_key=api_key or "sk-dummy",
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        logger.info("通风生成模块初始化成功，模型: %s", model_name)

    def _build_prompt(self, question: str, documents: list[Document]) -> str:
        context = self._format_context(documents)
        return f"""你是一位专业的{self.SYSTEM_ROLE}，熟悉《煤矿安全规程》及相关生产安全标准。

【严格约束】
1. 只能依据下方【参考规程内容】回答，不得凭空引用未在检索内容中出现的条款或数值。
2. 如果检索内容中有【规程附件：技术参数对照表】，答案所引用的一切数值必须直接出自该表。
3. 若检索结果确实不含某个具体数值，请明确说明“当前检索结果未包含该参数，建议查阅完整版《煤矿安全规程》”。

【参考规程内容】
{context}

【用户提问】
{question}

请按以下格式回答：
1. **核合性结论**：明确结论（合规/违规/数值限值）
2. **规程依据**：列出引用的具体条款编号和原文关键句
3. **专家解析**：结合现场实际做专业解释
4. **管理建议**（如有）：给出预防或整改建议

回答："""

    def generate_adaptive_answer(self, question: str, documents: list[Document]) -> str:
        prompt = self._build_prompt(question, documents)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("生成失败: %s", exc)
            return f"抱歉，系统生成解答时出现故障：{str(exc)}"

    def generate_adaptive_answer_stream(
        self,
        question: str,
        documents: list[Document],
        max_retries: int = 3,
    ) -> Generator[str, None, None]:
        prompt = self._build_prompt(question, documents)
        yield from self._stream_prompt(prompt, self.temperature, "生成中断，请稍后重试", max_retries)

    def _build_image_prompt(self, question: str, documents: list[Document], vision_result: Any) -> str:
        context = self._format_context(documents)
        structured_fields = getattr(vision_result, "structured_fields", {}) or {}
        concepts = getattr(vision_result, "concepts_retrieved", []) or []
        observations = getattr(vision_result, "key_observations", []) or []
        observation_text = "\n".join(f"- {item}" for item in observations) or "- 无明确关键观察"

        return f"""你是一位经验丰富的矿井通风安全检查员，善于结合图片分析结果和规程要求，做出专业的隐患判断。

【图片分析结果】
- 识别场景：{getattr(vision_result, "scene_name", "")}（置信度：{getattr(vision_result, "confidence", 0.0)}）
- 风险等级：{getattr(vision_result, "risk_level", "需要注意")}
- 主要隐患判断：{getattr(vision_result, "primary_hazard", "") or "暂无明确结论"}
- 第一轮观察：{getattr(vision_result, "raw_observations", "") or "无"}
- 关键观察：
{observation_text}
- 结构化参数：{json.dumps(structured_fields, ensure_ascii=False)}
- 参考概念定义：
{self._format_concept_summary(concepts)}

【参考规程内容】
{context}

【用户提问】
{question}

请你对图片中的通风安全状况做全面分析判断。

约束：
- 规程依据必须来自【参考规程内容】，不得编造条款编号或数值。
- 如果某个分析点超出当前检索范围，可以在“补充观察”中标注为“基于现场经验的初步判断”。
- 展示从图片观察到合规性判断的推理过程，但不要泄露与任务无关的内部思考。
- 输出 Markdown，结构可灵活组织，不必逐字套模板。

建议包含：
- **推理过程**：从图片观察到合规性判断的完整分析。
- **合规性结论**：明确判定结果和风险等级。
- **规程依据**：引用条款和数值，必须标注来源。
- **整改建议**：可操作的具体措施。
- **补充观察**：超出检索范围但值得关注的细节。

回答："""

    def generate_image_answer(self, question: str, documents: list[Document], vision_result: Any) -> str:
        prompt = self._build_image_prompt(question, documents, vision_result)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.35,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("图片报告生成失败: %s", exc)
            return f"抱歉，系统生成图片辨识报告时出现故障：{str(exc)}"

    def generate_image_answer_stream(
        self,
        question: str,
        documents: list[Document],
        vision_result: Any,
        max_retries: int = 3,
    ) -> Generator[str, None, None]:
        prompt = self._build_image_prompt(question, documents, vision_result)
        yield from self._stream_prompt(prompt, 0.35, "图片辨识报告生成中断，请稍后重试", max_retries)

    def _stream_prompt(
        self,
        prompt: str,
        temperature: float,
        failure_message: str,
        max_retries: int,
    ) -> Generator[str, None, None]:
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as exc:
                logger.warning("流式生成第 %s 次尝试失败: %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    yield f"\n[系统提示] {failure_message}。错误: {str(exc)}"

    def _format_context(self, documents: list[Document]) -> str:
        context_parts = []
        for doc in documents:
            content = doc.page_content.strip()
            if not content:
                continue
            name = doc.metadata.get("article_name") or doc.metadata.get("name") or "未知条款"
            level = doc.metadata.get("retrieval_level", "unknown").upper()
            context_parts.append(f"【参考条款：{name} | 检索方式：{level}】\n{content}")
        return "\n\n---\n\n".join(context_parts) or "当前未检索到可引用的规程内容。"

    def _format_concept_summary(self, concepts: list[dict[str, Any]]) -> str:
        if not concepts:
            return "未检索到明确概念定义。"
        lines = []
        for item in concepts:
            lines.append(
                "\n".join(
                    [
                        f"- {item.get('name', '未知概念')}",
                        f"  - 定义：{item.get('definition', '')}",
                        f"  - 视觉线索：{item.get('visual_clues', '')}",
                        f"  - 判别要点：{item.get('identification_features', '')}",
                    ]
                )
            )
        return "\n".join(lines)
