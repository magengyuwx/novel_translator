from __future__ import annotations

import json
import os
import re
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from novel_translator.config import AppConfig
from novel_translator.llm_factory import build_chat_model


def build_test_config(project_root: Path) -> AppConfig:
    config = AppConfig.from_env(project_root)
    config.llm_provider = "ollama"
    config.chat_model = config.chat_model or "qwen2.5:7b"
    config.base_url = config.base_url or "http://localhost:11434"
    return config


def resolve_available_model(base_url: str, preferred_model: str) -> str:
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise unittest.SkipTest(f"Ollama 服务不可用: {exc}") from exc

    models = payload.get("models") or []
    names = [item.get("name") for item in models if item.get("name")]
    if not names:
        raise unittest.SkipTest("Ollama 可访问，但没有可用模型。请先执行 `ollama pull <model>`。")
    if preferred_model in names:
        return preferred_model
    return names[0]


def build_verified_chat_model(project_root: Path):
    config = build_test_config(project_root)
    config.chat_model = resolve_available_model(config.base_url or "http://localhost:11434", config.chat_model)

    model = build_chat_model(config)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            ping = model.invoke("请只回复：ok")
            text = getattr(ping, "content", ping)
            if not str(text).strip():
                raise unittest.SkipTest("Ollama 模型调用成功但返回为空。")
            return config, model
        except unittest.SkipTest:
            raise
        except Exception as exc:
            last_error = exc
            lowered = str(exc).lower()
            transient = bool(re.search(r"status code:\s*5\d{2}", lowered)) or any(
                key in lowered for key in ("timed out", "timeout", "connection", "connect")
            )
            if not transient or attempt >= 4:
                break
            time.sleep(min(2 ** (attempt - 1), 4))
    raise unittest.SkipTest(f"Ollama 模型调用失败: {last_error}") from last_error
