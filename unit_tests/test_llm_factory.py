from __future__ import annotations

import unittest
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from unit_tests.ollama_test_helper import build_verified_chat_model


class TestLLMFactory(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.model = build_verified_chat_model(PROJECT_ROOT)

    def test_real_llm_invoke_returns_text(self) -> None:
        result = self.model.invoke("请只回复：测试通过")
        content = getattr(result, "content", result)
        text = str(content).strip()

        print(f"真实llm返回结果: {text}")
        self.assertTrue(text)


if __name__ == "__main__":
    unittest.main()
