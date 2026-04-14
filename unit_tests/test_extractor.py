from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from novel_translator.extractor import StoryMetadataExtractor
from unit_tests.ollama_test_helper import build_verified_chat_model


class TestExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.llm = build_verified_chat_model(PROJECT_ROOT)

    def test_extract_with_real_ollama_returns_non_empty_sections(self) -> None:
        extractor = StoryMetadataExtractor(self.config, self.llm)
        sample_text = (
            "Chapter 1\n"
            "Evelyn arrived at the station with a broken map. "
            "She wanted to find her missing brother Leon.\n\n"
            "Chapter 2\n"
            "Leon secretly worked for Captain Grey, but he still protected Evelyn in the shadows."
        )

        outline, characters = extractor.extract(sample_text)

        self.assertTrue(outline.strip())
        self.assertTrue(characters.strip())

    def test_message_to_text_supports_list(self) -> None:
        text = StoryMetadataExtractor._message_to_text(["a", "b"])
        self.assertEqual(text, "a\nb")

    def test_extract_section(self) -> None:
        source = "# 小说大纲\n\nA\n\n# 人物设定\n\nB"
        section = StoryMetadataExtractor._extract_section(source, "人物设定")
        self.assertEqual(section, "B")

    def test_extract_section_by_markers_with_numbered_headings(self) -> None:
        source = "## 一、小说大纲\n\n剧情A\n\n## 二、人物设定\n\n人物B"
        outline = StoryMetadataExtractor._extract_section_by_markers(
            source,
            ("小说大纲", "剧情大纲"),
            ("小说大纲", "剧情大纲", "人物设定", "角色设定"),
        )
        characters = StoryMetadataExtractor._extract_section_by_markers(
            source,
            ("人物设定", "角色设定"),
            ("小说大纲", "剧情大纲", "人物设定", "角色设定"),
        )
        self.assertIn("剧情A", outline)
        self.assertIn("人物B", characters)


if __name__ == "__main__":
    unittest.main()
