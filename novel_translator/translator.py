from __future__ import annotations

import time
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig
from .rag import NovelRAG
from .splitter import Chapter


TRANSLATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是资深文学翻译，请将英文小说翻译成自然、连贯、风格统一的简体中文。"
            "翻译时必须保证人名、地名、称谓、时间线和角色口吻前后一致。",
        ),
        (
            "human",
            "请翻译当前章节片段。\n\n"
            "【全局小说大纲】\n{outline}\n\n"
            "【人物设定】\n{characters}\n\n"
            "【上一章中文译文（如有）】\n{previous_chapter}\n\n"
            "【RAG 检索出的相关原文上下文】\n{related_source}\n\n"
            "【当前章节信息】\n"
            "标题：{title}\n"
            "当前片段：第 {segment_index}/{segment_total} 段\n\n"
            "【同章已完成译文尾部（帮助衔接）】\n{previous_segment_tail}\n\n"
            "【待翻译原文】\n{segment}\n\n"
            "要求：\n"
            "1. 只输出中文译文，不要解释。\n"
            "2. 不遗漏剧情与对白。\n"
            "3. 若当前是章节的中间片段，不要重复输出前文内容。\n"
            "4. 保持自然分段和文学性。",
        ),
    ]
)

REVISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是资深文学翻译编辑，请在保持剧情与人物设定一致的前提下，"
            "根据用户意见对译文进行针对性重写。",
        ),
        (
            "human",
            "请根据以下材料，对当前章节译文进行修订并重新输出完整章节中文译文。\n\n"
            "【全局小说大纲】\n{outline}\n\n"
            "【人物设定】\n{characters}\n\n"
            "【上一章中文译文（如有）】\n{previous_chapter}\n\n"
            "【RAG 检索出的相关原文上下文】\n{related_source}\n\n"
            "【章节标题】\n{title}\n\n"
            "【原文】\n{source_text}\n\n"
            "【当前译文】\n{current_translation}\n\n"
            "【用户修改意见】\n{feedback}\n\n"
            "要求：\n"
            "1. 输出完整修订版中文译文，不要解释。\n"
            "2. 保持分段自然、语言流畅。\n"
            "3. 必须落实用户修改意见。\n"
            "4. 不要遗漏原文关键信息。",
        ),
    ]
)


class ChapterTranslator:
    def __init__(self, config: AppConfig, llm, rag: NovelRAG) -> None:
        self.config = config
        self.llm = llm
        self.rag = rag

    def translate(self, chapter: Chapter, outline: str, characters: str) -> str:
        context = self.rag.build_translation_context(chapter)
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ". ", " "],
            chunk_size=self.config.translation_chunk_size,
            chunk_overlap=self.config.translation_chunk_overlap,
        )
        segments = splitter.split_text(chapter.content)

        translated_segments: list[str] = []
        for index, segment in enumerate(segments, start=1):
            previous_segment_tail = "\n".join(translated_segments)[-1200:]
            response = self._invoke_with_retry(
                TRANSLATION_PROMPT.format_messages(
                    outline=context["outline"] or outline,
                    characters=context["characters"] or characters,
                    previous_chapter=context["previous_chapter"],
                    related_source=context["related_source"],
                    title=chapter.title,
                    segment_index=index,
                    segment_total=len(segments),
                    previous_segment_tail=previous_segment_tail,
                    segment=segment,
                )
            )
            translated_segments.append(self._normalize(self._message_to_text(response)))
            print(f"[translate] 章节 {chapter.number} '{chapter.title}' - 已翻译段落 {index}/{len(segments)}")

        return "\n\n".join(segment for segment in translated_segments if segment.strip()).strip()

    def revise_translation(
        self,
        chapter: Chapter,
        *,
        outline: str,
        characters: str,
        current_translation: str,
        feedback: str,
    ) -> str:
        context = self.rag.build_translation_context(chapter)
        response = self._invoke_with_retry(
            REVISION_PROMPT.format_messages(
                outline=context["outline"] or outline,
                characters=context["characters"] or characters,
                previous_chapter=context["previous_chapter"],
                related_source=context["related_source"],
                title=chapter.title,
                source_text=chapter.content,
                current_translation=current_translation,
                feedback=feedback,
            )
        )
        return self._normalize(self._message_to_text(response))

    def _invoke_with_retry(self, messages, max_attempts: int = 6):
        prompt_stats = self._build_prompt_stats(messages)
        for attempt in range(1, max_attempts + 1):
            try:
                return self.llm.invoke(messages)
            except Exception as exc:
                text = str(exc)
                lowered = text.lower()
                has_5xx = bool(re.search(r"status code:\s*5\d{2}", lowered))
                is_transient = has_5xx or any(
                    keyword in lowered
                    for keyword in (
                        "502",
                        "503",
                        "504",
                        "bad gateway",
                        "service temporarily unavailable",
                        "timed out",
                        "timeout",
                        "connection",
                        "connect",
                        "eof",
                    )
                )
                if attempt >= max_attempts or not is_transient:
                    print(
                        "[error] 翻译调用最终失败。"
                        f" attempt={attempt}/{max_attempts}, transient={is_transient}, error={text}"
                    )
                    self._print_prompt_debug(prompt_stats)
                    raise
                sleep_seconds = min(2 ** (attempt - 1), 8)
                print(
                    f"[retry] 翻译调用失败，第 {attempt} 次重试，{sleep_seconds}s 后继续。"
                    f"原因: {text}; prompt_chars={prompt_stats['total_chars']};"
                    f" message_count={prompt_stats['message_count']}"
                )
                if attempt == 1:
                    self._print_prompt_debug(prompt_stats)
                time.sleep(sleep_seconds)

    @staticmethod
    def _message_content_to_text(message) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _build_prompt_stats(self, messages) -> dict[str, object]:
        per_message: list[dict[str, object]] = []
        total_chars = 0
        human_preview = ""
        for index, message in enumerate(messages, start=1):
            text = self._message_content_to_text(message)
            role = getattr(message, "type", message.__class__.__name__)
            size = len(text)
            total_chars += size
            if role.lower() == "human" and not human_preview:
                human_preview = text[:800]
            per_message.append({"index": index, "role": role, "chars": size})
        return {
            "message_count": len(per_message),
            "total_chars": total_chars,
            "per_message": per_message,
            "human_preview": human_preview,
        }

    @staticmethod
    def _print_prompt_debug(prompt_stats: dict[str, object]) -> None:
        print("[debug] Prompt 统计信息：")
        print(
            f"[debug] message_count={prompt_stats['message_count']}, "
            f"total_chars={prompt_stats['total_chars']}"
        )
        for item in prompt_stats["per_message"]:  # type: ignore[index]
            print(
                f"[debug] message[{item['index']}] role={item['role']} chars={item['chars']}"
            )
        preview = str(prompt_stats.get("human_preview", ""))
        if preview:
            normalized = preview.replace("\r", " ").replace("\n", "\\n")
            print(f"[debug] human_preview(<=800): {normalized}")

    @staticmethod
    def _message_to_text(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(item) for item in content).strip()
        return str(content).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = text.replace("```markdown", "").replace("```", "").strip()
        return cleaned
