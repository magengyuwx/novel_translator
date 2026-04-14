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
from novel_translator.workflow import NovelTranslationWorkflow
from unit_tests.ollama_test_helper import build_verified_chat_model


class TestWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp_dir.name)
        verified_config, _ = build_verified_chat_model(PROJECT_ROOT)
        self.config = AppConfig(
            project_root=self.root,
            llm_provider="ollama",
            chat_model=verified_config.chat_model,
            base_url=verified_config.base_url,
            embedding_provider="simple",
            embedding_model="dummy",
            translation_chunk_size=1200,
            translation_chunk_overlap=100,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
        self.temp_dir.cleanup()

    def test_prepare_with_real_ollama_generates_metadata(self) -> None:
        workflow = NovelTranslationWorkflow(self.config)
        input_file = self.root / "data" / "input" / "novel.txt"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_text(
            "Chapter 1 The Night Train\n"
            "Evelyn boarded the train and searched for Leon's notebook.\n\n"
            "Chapter 2 The Hidden Letter\n"
            "Captain Grey warned her to stop investigating the old harbor.",
            encoding="utf-8",
        )
        result = workflow.prepare("data/input/novel.txt", max_chapters=1)
        self.assertEqual(len(result), 1)
        self.assertTrue(self.config.outline_path.exists())
        self.assertTrue(self.config.characters_path.exists())
        self.assertTrue(self.config.outline_path.read_text(encoding="utf-8").strip())
        self.assertTrue(self.config.characters_path.read_text(encoding="utf-8").strip())

    def test_prepare_with_starcraft_sample_pdf(self) -> None:
        verified_config, _ = build_verified_chat_model(PROJECT_ROOT)
        project_config = AppConfig(
            project_root=PROJECT_ROOT,
            llm_provider="ollama",
            chat_model=verified_config.chat_model,
            base_url=verified_config.base_url,
            embedding_provider="simple",
            embedding_model="dummy",
            translation_chunk_size=1200,
            translation_chunk_overlap=100,
            rag_collection_name="novel_translation_test_prepare",
        )
        workflow = NovelTranslationWorkflow(project_config)
        candidates = list((PROJECT_ROOT / "samples").glob("The Starcraft Archive An Anthology*.pdf"))
        if not candidates:
            self.skipTest("未找到样例文件: samples/The Starcraft Archive An Anthology*.pdf")
        sample_file = candidates[0].resolve()

        chapters = workflow.prepare(str(sample_file), max_chapters=1)

        self.assertGreaterEqual(len(chapters), 1)
        self.assertTrue(project_config.original_copy_path.exists())
        self.assertTrue(project_config.outline_path.exists())
        self.assertTrue(project_config.characters_path.exists())
        self.assertTrue(project_config.outline_path.read_text(encoding="utf-8").strip())
        self.assertTrue(project_config.characters_path.read_text(encoding="utf-8").strip())
        print(f"[test-output] outline: {project_config.outline_path}")
        print(f"[test-output] characters: {project_config.characters_path}")

    def test_translate_with_real_ollama_writes_output(self) -> None:
        workflow = NovelTranslationWorkflow(self.config)
        input_file = self.root / "data" / "input" / "novel.txt"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        input_file.write_text(
            "Chapter 1 The Night Train\n"
            "Evelyn boarded the train and searched for Leon's notebook.",
            encoding="utf-8",
        )

        workflow.prepare("data/input/novel.txt", max_chapters=1)
        outputs = workflow.translate("data/input/novel.txt", force=True, max_chapters=1)
        self.assertEqual(len(outputs), 1)
        text = outputs[0].read_text(encoding="utf-8")
        self.assertRegex(text, r"[\u4e00-\u9fff]")

    def test_resolve_input_path_raises_for_missing_file(self) -> None:
        workflow = NovelTranslationWorkflow(self.config)
        with self.assertRaises(FileNotFoundError):
            workflow._resolve_input_path("data/input/not_exists.txt")


if __name__ == "__main__":
    unittest.main()
