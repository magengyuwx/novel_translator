from __future__ import annotations
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.config import AppConfig
from novel_translator.loader import load_text
from novel_translator.rag import NovelRAG
from novel_translator.splitter import Chapter
from novel_translator.translator import ChapterTranslator
from unit_tests.ollama_test_helper import build_verified_chat_model


class TestTranslator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        self.config = AppConfig(
            project_root=self.root,
            llm_provider="ollama",
            chat_model="qwen2.5:7b",
            base_url="http://localhost:11434",
            embedding_provider="simple",
            embedding_model="dummy",
            translation_chunk_size=1200,
            translation_chunk_overlap=10,
        )
        verified_config, self.llm = build_verified_chat_model(PROJECT_ROOT)
        self.config.chat_model = verified_config.chat_model
        self.config.base_url = verified_config.base_url
        self.config.ensure_directories()
        self.rag = NovelRAG(self.config)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.temp_dir.cleanup()

    def test_translate_with_real_ollama_returns_chinese_text(self) -> None:
        chapter = Chapter(
            number=1,
            title="Chapter 1",
            content=(
                "Evelyn opened the attic window and whispered, "
                "'If Leon is alive, he will come back before dawn.' "
                "Rain tapped against the old roof."
            ),
        )
        self.rag.reset_store()
        self.rag.index_story_metadata(
            "Evelyn searches for Leon while hiding from Captain Grey.",
            "Evelyn is Leon's sister; Captain Grey is an officer with unclear motives.",
        )
        self.rag.index_chapters([chapter])
        translator = ChapterTranslator(self.config, self.llm, self.rag)

        output = translator.translate(
            chapter,
            outline="主线是 Evelyn 寻找 Leon。",
            characters="Evelyn: 女主；Leon: 失踪的哥哥。",
        )

        self.assertRegex(output, r"[\u4e00-\u9fff]")
        self.assertNotIn("```", output)

    def test_message_to_text_handles_list(self) -> None:
        text = ChapterTranslator._message_to_text(["x", "y"])
        self.assertEqual(text, "x\ny")

    def test_translate_starcraft_chapter6_source_excerpt_with_real_ollama(self) -> None:
        source_path = PROJECT_ROOT / "data" / "output" / "chapters" / "source" / "chapter_006_chapter_6.txt"
        if not source_path.exists():
            self.skipTest(f"未找到章节文件: {source_path}")

        # 使用第六章前段文本，验证真实模型翻译流程，降低 502 波动风险。
        source_text = load_text(source_path)
        chapter = Chapter(
            number=6,
            title="CHAPTER 6",
            content=source_text[:6000],
        )

        self.rag.reset_store()
        self.rag.index_story_metadata(
            "Koprulu 区局势升级，记者视角持续记录泰伦与异虫冲突。",
            "Michael Liberty 为核心视角角色；新闻团队与军方关系紧张。",
        )
        self.rag.index_chapters([chapter])

        translator = ChapterTranslator(self.config, self.llm, self.rag)
        output = translator.translate(
            chapter,
            outline="主线围绕战局演变与记者调查推进。",
            characters="Michael Liberty：记者；Anderson：主编。",
        )

        self.assertRegex(output, r"[\u4e00-\u9fff]")
        self.assertGreater(len(output), 120)
        self.assertNotIn("```", output)


if __name__ == "__main__":
    unittest.main()
