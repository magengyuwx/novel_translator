from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

from .config import AppConfig
from .extractor import StoryMetadataExtractor
from .llm_factory import build_chat_model
from .loader import copy_text, load_text, save_text
from .rag import NovelRAG
from .splitter import Chapter, load_saved_chapters, save_chapters, split_novel_into_chapters
from .translator import ChapterTranslator


class NovelTranslationWorkflow:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.from_env()
        self.config.ensure_directories()

        self.llm = build_chat_model(self.config)
        self.rag = NovelRAG(self.config)
        self.extractor = StoryMetadataExtractor(self.config, self.llm)
        self.translator = ChapterTranslator(self.config, self.llm, self.rag)

    def prepare(
        self,
        input_path: str | Path | None = None,
        *,
        max_chapters: int | None = None,
    ) -> list[Chapter]:
        prepare_start = time.perf_counter()
        source_file = self._resolve_input_path(input_path)
        print(f"[prepare] 开始预处理，输入文件: {source_file}")
        full_text = load_text(source_file)

        copy_text(source_file, self.config.original_copy_path)
        split_start = time.perf_counter()
        print("[prepare] 开始章节切分...")
        all_chapters = split_novel_into_chapters(full_text)
        chapters = all_chapters[:max_chapters] if max_chapters else all_chapters
        save_chapters(chapters, self.config.chapters_source_dir)
        print(
            f"[prepare] 章节切分完成，共 {len(chapters)} 章，耗时 {time.perf_counter() - split_start:.2f}s"
        )

        analysis_text = "\n\n".join(chapter.content for chapter in chapters)
        extract_start = time.perf_counter()
        print("[prepare] 开始提取小说大纲与人物设定...")
        outline, characters = self.extractor.extract(analysis_text)
        save_text(self.config.outline_path, f"# 小说大纲\n\n{outline}")
        save_text(self.config.characters_path, f"# 人物设定\n\n{characters}")
        print(f"[prepare] 设定提取完成，耗时 {time.perf_counter() - extract_start:.2f}s")

        self.rag.reset_store()
        self.rag.index_full_novel(analysis_text)
        self.rag.index_chapters(chapters)
        self.rag.index_story_metadata(outline, characters)
        print(f"[prepare] RAG 索引完成，总耗时 {time.perf_counter() - prepare_start:.2f}s")

        return chapters

    def translate(
        self,
        input_path: str | Path | None = None,
        *,
        force: bool = False,
        max_chapters: int | None = None,
    ) -> list[Path]:
        chapters = self._ensure_prepared(input_path, max_chapters=max_chapters)
        outline = load_text(self.config.outline_path)
        characters = load_text(self.config.characters_path)

        output_files: list[Path] = []
        for chapter in chapters:
            if max_chapters is not None and chapter.number > max_chapters:
                break

            target_path = self.config.chapter_translation_path(chapter.number, chapter.title)
            if target_path.exists() and not force:
                print(f"[skip] 第 {chapter.number} 章已有译文：{target_path.name}")
                continue

            chapter_start = time.perf_counter()
            print(f"[translate] 开始翻译第 {chapter.number} 章：{chapter.title}")
            translated_text = self.translator.translate(chapter, outline, characters)
            save_text(target_path, translated_text)
            self.rag.index_translation(chapter, translated_text)
            output_files.append(target_path)
            print(
                f"[done] 已翻译第 {chapter.number} 章 -> {target_path.name} "
                f"(耗时 {time.perf_counter() - chapter_start:.2f}s)"
            )

        return output_files

    def run_all(
        self,
        input_path: str | Path | None = None,
        *,
        force: bool = False,
        max_chapters: int | None = None,
    ) -> tuple[list[Chapter], list[Path]]:
        chapters = self.prepare(input_path, max_chapters=max_chapters)
        outputs = self.translate(
            input_path=input_path,
            force=force,
            max_chapters=max_chapters,
        )
        return chapters, outputs

    def _ensure_prepared(
        self,
        input_path: str | Path | None,
        max_chapters: int | None = None,
    ) -> list[Chapter]:
        chapter_files = list(self.config.chapters_source_dir.glob("chapter_*.txt"))
        if not chapter_files or not self.config.outline_path.exists() or not self.config.characters_path.exists():
            return self.prepare(input_path, max_chapters=max_chapters)

        chapters = load_saved_chapters(self.config.chapters_source_dir)
        self._rebuild_rag_from_disk(chapters, input_path)
        return chapters

    def _rebuild_rag_from_disk(
        self,
        chapters: list[Chapter],
        input_path: str | Path | None,
    ) -> None:
        source_copy = self.config.original_copy_path
        if source_copy.exists():
            full_text = load_text(source_copy)
        else:
            full_text = load_text(self._resolve_input_path(input_path))

        outline = load_text(self.config.outline_path)
        characters = load_text(self.config.characters_path)

        self.rag.reset_store()
        self.rag.index_full_novel(full_text)
        self.rag.index_chapters(chapters)
        self.rag.index_story_metadata(outline, characters)

        for chapter in chapters:
            translated_path = self.config.chapter_translation_path(chapter.number, chapter.title)
            if translated_path.exists():
                self.rag.index_translation(chapter, load_text(translated_path))

    def _resolve_input_path(self, input_path: str | Path | None) -> Path:
        path = Path(input_path or self.config.default_input_file)
        if not path.is_absolute():
            path = (self.config.project_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"找不到原始小说文件：{path}。请先把 txt 文件放到 data/input/ 下，或通过 --input 指定路径。"
            )
        return path

    def clean_rag(self, *, retries: int = 5, retry_sleep: float = 0.8) -> Path:
        rag_dir = self.config.rag_dir
        print(f"[clean-rag] 开始清理 RAG 目录: {rag_dir}")

        try:
            self.rag.vectorstore.delete_collection()
        except Exception:
            pass

        # 尝试释放本进程中的引用，降低 Windows 锁文件概率。
        gc.collect()

        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                if rag_dir.exists():
                    shutil.rmtree(rag_dir)
                rag_dir.mkdir(parents=True, exist_ok=True)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"[clean-rag] 第 {attempt}/{retries} 次删除失败，"
                    f"{retry_sleep}s 后重试。原因: {exc}"
                )
                time.sleep(retry_sleep)
                gc.collect()

        if last_error is not None:
            raise RuntimeError(
                f"清理 RAG 目录失败：{rag_dir}。"
                "请先关闭占用该目录的 python/streamlit/uvicorn 进程后重试。"
            ) from last_error

        self.rag = NovelRAG(self.config)
        self.translator.rag = self.rag
        print(f"[clean-rag] 清理完成: {rag_dir}")
        return rag_dir
