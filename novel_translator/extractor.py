from __future__ import annotations

import re
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import AppConfig


PARTIAL_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是长篇小说分析助手，擅长从长文本中提炼主线剧情、人物设定和角色关系。",
        ),
        (
            "human",
            "请分析以下小说片段，并输出两部分内容：\n"
            "1. 剧情要点：按时间顺序列出关键事件\n"
            "2. 人物要点：记录新出现或重要人物的身份、关系、称谓、目标、性格\n\n"
            "片段编号：{chunk_index}\n"
            "小说片段：\n{chunk}",
        ),
    ]
)

MERGE_METADATA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是小说编辑，请将零散笔记整理成稳定可复用的翻译资料。"
            "你必须显式整理中英文术语与人名映射，便于后续翻译保持一致。",
        ),
        (
            "human",
            "请基于以下分段分析结果，合并输出：\n"
            "# 小说大纲\n"
            "- 按故事推进顺序总结主要剧情、矛盾冲突和阶段目标\n\n"
            "## 重要术语中英对照\n"
            "- 列出小说中重要组织、地点、科技、事件等术语\n"
            "- 使用表格输出，列名固定为：术语英文 | 术语中文 | 备注\n\n"
            "# 人物设定\n"
            "- 按人物列出身份、关系网、称谓、性格特征、重要经历\n\n"
            "## 重要人物中英对照\n"
            "- 列出重要人物姓名和常见称谓\n"
            "- 使用表格输出，列名固定为：人物英文名 | 人物中文译名 | 别名/称谓 | 关系与身份\n\n"
            "要求：去重、统一名字写法、避免遗漏关键人物。\n\n"
            "分析材料：\n{chunk_notes}",
        ),
    ]
)


class StoryMetadataExtractor:
    def __init__(self, config: AppConfig, llm) -> None:
        self.config = config
        self.llm = llm

    def extract(self, full_text: str) -> tuple[str, str]:
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "！", "？", ". ", " "],
            chunk_size=6000,
            chunk_overlap=400,
        )
        chunks = splitter.split_text(full_text)
        print(f"[extractor] 进入设定提取，共 {len(chunks)} 个文本分段。")

        chunk_notes: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            print(f"[extractor] 分析片段 {index}/{len(chunks)}...")
            response = self._invoke_with_retry(
                PARTIAL_ANALYSIS_PROMPT.format_messages(
                    chunk_index=index,
                    chunk=chunk,
                )
            )
            chunk_notes.append(f"## 片段 {index}\n{self._message_to_text(response)}")

        print("[extractor] 开始合并分段结果...")
        merged_response = self._invoke_with_retry(
            MERGE_METADATA_PROMPT.format_messages(
                chunk_notes="\n\n".join(chunk_notes),
            )
        )
        merged_text = self._message_to_text(merged_response)

        outline_markers = ("小说大纲", "剧情大纲", "故事大纲", "主线剧情")
        character_markers = ("人物设定", "角色设定", "人物关系", "主要人物")
        all_markers = tuple(dict.fromkeys((*outline_markers, *character_markers)))

        outline = self._extract_section_by_markers(merged_text, outline_markers, all_markers)
        characters = self._extract_section_by_markers(merged_text, character_markers, all_markers)

        if not outline:
            outline = self._extract_section(merged_text, "小说大纲")
        if not characters:
            characters = self._extract_section(merged_text, "人物设定")

        if not outline:
            outline = merged_text
        if not characters:
            characters = merged_text

        if outline.strip() == characters.strip():
            split_result = self._split_outline_and_characters_by_position(merged_text, character_markers)
            if split_result:
                outline, characters = split_result
                print("[extractor] 检测到大纲/人物设定重复，已按关键标题重新拆分。")
            else:
                print("[extractor] 警告：大纲与人物设定内容相同，可能是模型输出结构不稳定。")

        print("[extractor] 大纲与人物设定提取完成。")
        return outline.strip(), characters.strip()

    def _invoke_with_retry(self, messages, max_attempts: int = 6):
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
                    raise
                sleep_seconds = min(2 ** (attempt - 1), 8)
                print(f"[retry] 元数据提取调用失败，第 {attempt} 次重试，{sleep_seconds}s 后继续。原因: {text}")
                time.sleep(sleep_seconds)

    @staticmethod
    def _message_to_text(response) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(str(item) for item in content).strip()
        return str(content).strip()

    @staticmethod
    def _extract_section(text: str, heading: str) -> str:
        pattern = rf"(?is)(?:^|\n)#+\s*{re.escape(heading)}\s*(.*?)(?=\n#+\s*[^\n]+|\Z)"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_section_by_markers(text: str, markers: tuple[str, ...], all_markers: tuple[str, ...]) -> str:
        marker_alts = "|".join(re.escape(marker) for marker in markers)
        all_alts = "|".join(re.escape(marker) for marker in all_markers)
        # 支持标题格式：
        # # 小说大纲
        # ## 一、小说大纲
        # 小说大纲：
        pattern = (
            rf"(?is)(?:^|\n)\s*(?:#+\s*)?(?:[一二三四五六七八九十0-9]+[、.．]\s*)?"
            rf"(?:{marker_alts})\s*[:：]?\s*(.*?)"
            rf"(?=\n\s*(?:#+\s*)?(?:[一二三四五六七八九十0-9]+[、.．]\s*)?(?:{all_alts})\b|\Z)"
        )
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _split_outline_and_characters_by_position(
        text: str,
        character_markers: tuple[str, ...],
    ) -> tuple[str, str] | None:
        lowered = text.lower()
        positions: list[int] = []
        for marker in character_markers:
            pos = lowered.find(marker.lower())
            if pos >= 0:
                positions.append(pos)
        if not positions:
            return None
        split_pos = min(positions)
        outline = text[:split_pos].strip()
        characters = text[split_pos:].strip()
        if not outline or not characters or outline == characters:
            return None
        return outline, characters
