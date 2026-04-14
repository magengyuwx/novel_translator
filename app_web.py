from __future__ import annotations

from pathlib import Path

import streamlit as st

from novel_translator.config import AppConfig
from novel_translator.loader import load_text, save_text
from novel_translator.splitter import Chapter, load_saved_chapters
from novel_translator.workflow import NovelTranslationWorkflow


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_default_settings() -> dict[str, object]:
    cfg = AppConfig.from_env(PROJECT_ROOT)
    return {
        "llm_provider": cfg.llm_provider,
        "chat_model": cfg.chat_model,
        "base_url": cfg.base_url or "",
        "embedding_provider": cfg.embedding_provider,
        "embedding_model": cfg.embedding_model,
        "temperature": cfg.temperature,
        "rag_top_k": cfg.rag_top_k,
        "translation_chunk_size": cfg.translation_chunk_size,
        "translation_chunk_overlap": cfg.translation_chunk_overlap,
        "rag_collection_name": cfg.rag_collection_name,
    }


def _write_env_file(settings: dict[str, object]) -> None:
    env_lines = [
        f"LLM_PROVIDER={settings['llm_provider']}",
        f"CHAT_MODEL={settings['chat_model']}",
        "API_KEY=",
        "OPENAI_BASE_URL=",
        f"OLLAMA_BASE_URL={settings['base_url']}",
        f"EMBEDDING_PROVIDER={settings['embedding_provider']}",
        f"EMBEDDING_MODEL={settings['embedding_model']}",
        f"TEMPERATURE={settings['temperature']}",
        f"RAG_TOP_K={settings['rag_top_k']}",
        f"TRANSLATION_CHUNK_SIZE={settings['translation_chunk_size']}",
        f"TRANSLATION_CHUNK_OVERLAP={settings['translation_chunk_overlap']}",
        f"RAG_COLLECTION_NAME={settings['rag_collection_name']}",
    ]
    (PROJECT_ROOT / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")


def _settings_signature(settings: dict[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in settings.items()))


def _build_config(settings: dict[str, object]) -> AppConfig:
    return AppConfig(
        project_root=PROJECT_ROOT,
        llm_provider=str(settings["llm_provider"]),
        chat_model=str(settings["chat_model"]),
        base_url=str(settings["base_url"]) or None,
        embedding_provider=str(settings["embedding_provider"]),
        embedding_model=str(settings["embedding_model"]),
        temperature=float(settings["temperature"]),
        rag_top_k=int(settings["rag_top_k"]),
        translation_chunk_size=int(settings["translation_chunk_size"]),
        translation_chunk_overlap=int(settings["translation_chunk_overlap"]),
        rag_collection_name=str(settings["rag_collection_name"]),
    )


def _get_runtime(settings: dict[str, object]) -> tuple[AppConfig, NovelTranslationWorkflow]:
    signature = _settings_signature(settings)
    if st.session_state.get("runtime_signature") != signature:
        config = _build_config(settings)
        workflow = NovelTranslationWorkflow(config)
        st.session_state["runtime_signature"] = signature
        st.session_state["runtime_config"] = config
        st.session_state["runtime_workflow"] = workflow
    return st.session_state["runtime_config"], st.session_state["runtime_workflow"]


def _list_input_candidates() -> list[str]:
    candidates: list[str] = []
    for folder in (PROJECT_ROOT / "samples", PROJECT_ROOT / "data" / "input"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".txt", ".md", ".pdf", ".epub"}:
                candidates.append(str(path.resolve()))
    return candidates


def _resolve_selected_input(candidates: list[str], custom_path: str) -> str:
    path = custom_path.strip()
    if path:
        return str(Path(path).resolve())
    return candidates[0] if candidates else ""


def _render_sidebar() -> tuple[dict[str, object], str, int | None]:
    st.sidebar.header("系统设置")
    defaults = _load_default_settings()

    llm_provider = st.sidebar.selectbox(
        "LLM_PROVIDER",
        options=["ollama", "openai", "openrouter", "custom"],
        index=max(0, ["ollama", "openai", "openrouter", "custom"].index(defaults["llm_provider"]))
        if defaults["llm_provider"] in {"ollama", "openai", "openrouter", "custom"}
        else 0,
    )
    chat_model = st.sidebar.text_input("CHAT_MODEL", value=str(defaults["chat_model"]))
    base_url = st.sidebar.text_input("BASE_URL / OLLAMA_BASE_URL", value=str(defaults["base_url"]))
    embedding_provider = st.sidebar.selectbox(
        "EMBEDDING_PROVIDER",
        options=["simple", "huggingface", "ollama", "openai", "custom"],
        index=max(0, ["simple", "huggingface", "ollama", "openai", "custom"].index(defaults["embedding_provider"]))
        if defaults["embedding_provider"] in {"simple", "huggingface", "ollama", "openai", "custom"}
        else 0,
    )
    embedding_model = st.sidebar.text_input("EMBEDDING_MODEL", value=str(defaults["embedding_model"]))
    temperature = st.sidebar.slider("TEMPERATURE", 0.0, 1.0, float(defaults["temperature"]), 0.05)
    rag_top_k = st.sidebar.number_input("RAG_TOP_K", min_value=1, max_value=20, value=int(defaults["rag_top_k"]))
    translation_chunk_size = st.sidebar.number_input(
        "TRANSLATION_CHUNK_SIZE",
        min_value=500,
        max_value=12000,
        value=int(defaults["translation_chunk_size"]),
        step=100,
    )
    translation_chunk_overlap = st.sidebar.number_input(
        "TRANSLATION_CHUNK_OVERLAP",
        min_value=0,
        max_value=4000,
        value=int(defaults["translation_chunk_overlap"]),
        step=50,
    )
    rag_collection_name = st.sidebar.text_input("RAG_COLLECTION_NAME", value=str(defaults["rag_collection_name"]))
    max_chapters = st.sidebar.number_input("本次最多处理章节数", min_value=1, max_value=200, value=3)

    settings = {
        "llm_provider": llm_provider,
        "chat_model": chat_model,
        "base_url": base_url,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "temperature": float(temperature),
        "rag_top_k": int(rag_top_k),
        "translation_chunk_size": int(translation_chunk_size),
        "translation_chunk_overlap": int(translation_chunk_overlap),
        "rag_collection_name": rag_collection_name,
    }

    if st.sidebar.button("保存设置到 .env"):
        _write_env_file(settings)
        st.sidebar.success("已写入 .env")

    st.sidebar.divider()
    st.sidebar.subheader("输入文件")
    uploader = st.sidebar.file_uploader("上传小说文件（txt/md/pdf/epub）", type=["txt", "md", "pdf", "epub"])
    uploaded_path = ""
    if uploader is not None:
        upload_dir = PROJECT_ROOT / "data" / "input" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / uploader.name
        target.write_bytes(uploader.getbuffer())
        uploaded_path = str(target.resolve())
        st.sidebar.success(f"已保存: {target.name}")

    candidates = _list_input_candidates()
    default_index = 0
    if uploaded_path and uploaded_path in candidates:
        default_index = candidates.index(uploaded_path)
    selected = st.sidebar.selectbox("选择源文件", options=candidates or [""], index=default_index)
    custom_path = st.sidebar.text_input("或手动输入路径（可覆盖上面的选择）", value=uploaded_path)
    input_path = _resolve_selected_input(candidates if candidates else [selected], custom_path or selected)
    return settings, input_path, int(max_chapters)


def _load_outline_and_characters(config: AppConfig) -> tuple[str, str]:
    outline = load_text(config.outline_path) if config.outline_path.exists() else ""
    characters = load_text(config.characters_path) if config.characters_path.exists() else ""
    return outline, characters


def _load_chapters(config: AppConfig) -> list[Chapter]:
    if not config.chapters_source_dir.exists():
        return []
    return load_saved_chapters(config.chapters_source_dir)


def main() -> None:
    st.set_page_config(page_title="长篇小说翻译工作台", layout="wide")
    st.title("长篇小说翻译工作台")
    st.caption("流程：选择文件 -> 分析大纲/人物 -> 分章节 -> 逐章翻译 -> 提意见重生成")

    settings, input_path, max_chapters = _render_sidebar()
    if not input_path:
        st.warning("请先选择或上传输入文件。")
        return

    config, workflow = _get_runtime(settings)

    st.subheader("1. 分析并切章")
    st.code(input_path, language="text")
    if st.button("运行分析与切章", type="primary"):
        try:
            chapters = workflow.prepare(input_path=input_path, max_chapters=max_chapters)
            st.session_state["prepared_input_path"] = input_path
            st.session_state["prepared_chapter_count"] = len(chapters)
            st.success(f"预处理完成，共 {len(chapters)} 章。")
        except Exception as exc:
            st.error(f"预处理失败: {exc}")

    if st.session_state.get("prepared_input_path"):
        st.info(
            f"最近一次预处理输入: {st.session_state['prepared_input_path']} | "
            f"章节数: {st.session_state.get('prepared_chapter_count', 0)}"
        )

    outline, characters = _load_outline_and_characters(config)
    if "outline_text" not in st.session_state:
        st.session_state["outline_text"] = outline
    if "characters_text" not in st.session_state:
        st.session_state["characters_text"] = characters

    st.subheader("2. 大纲与人物设定（可修改）")
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.session_state["outline_text"] = st.text_area("小说大纲", value=st.session_state["outline_text"], height=260)
    with col_meta2:
        st.session_state["characters_text"] = st.text_area(
            "人物设定", value=st.session_state["characters_text"], height=260
        )

    if st.button("保存大纲/人物设定并重建RAG索引"):
        try:
            save_text(config.outline_path, st.session_state["outline_text"])
            save_text(config.characters_path, st.session_state["characters_text"])
            chapters = _load_chapters(config)
            if chapters:
                workflow._rebuild_rag_from_disk(chapters, input_path)  # noqa: SLF001
            st.success("已保存并完成 RAG 重建。")
        except Exception as exc:
            st.error(f"保存失败: {exc}")

    chapters = _load_chapters(config)
    st.subheader("3. 章节翻译")
    if not chapters:
        st.warning("还没有章节数据。请先运行“分析与切章”。")
        return

    chapter_options = {f"第{chapter.number}章 | {chapter.title}": chapter for chapter in chapters}
    selected_label = st.selectbox("选择章节", list(chapter_options.keys()))
    chapter = chapter_options[selected_label]
    target_path = config.chapter_translation_path(chapter.number, chapter.title)
    existing_translation = load_text(target_path) if target_path.exists() else ""
    translation_key = f"translation_text_{chapter.number}"
    feedback_key = f"feedback_text_{chapter.number}"

    if translation_key not in st.session_state:
        st.session_state[translation_key] = existing_translation
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = ""

    col_src, col_zh = st.columns(2)
    with col_src:
        st.markdown("**原文章节**")
        st.text_area("原文", value=chapter.content, height=560, disabled=True, label_visibility="collapsed")
    with col_zh:
        st.markdown("**中文译文**")
        st.session_state[translation_key] = st.text_area(
            "译文",
            value=st.session_state[translation_key],
            height=560,
            label_visibility="collapsed",
        )

    action_col1, action_col2, action_col3 = st.columns([1, 1, 2])
    with action_col1:
        if st.button("翻译当前章节", use_container_width=True):
            try:
                translated = workflow.translator.translate(
                    chapter,
                    outline=st.session_state["outline_text"],
                    characters=st.session_state["characters_text"],
                )
                save_text(target_path, translated)
                workflow.rag.index_translation(chapter, translated)
                st.session_state[translation_key] = translated
                st.success(f"翻译完成: {target_path.name}")
            except Exception as exc:
                st.error(f"翻译失败: {exc}")
    with action_col2:
        if st.button("保存当前译文", use_container_width=True):
            try:
                save_text(target_path, st.session_state[translation_key])
                workflow.rag.index_translation(chapter, st.session_state[translation_key])
                st.success(f"已保存: {target_path.name}")
            except Exception as exc:
                st.error(f"保存失败: {exc}")

    st.markdown("**修改意见（用于重生成）**")
    st.session_state[feedback_key] = st.text_area(
        "请写明希望调整的点：术语、人名、语气、直译/意译程度等。",
        value=st.session_state[feedback_key],
        height=130,
        label_visibility="collapsed",
    )

    if st.button("根据意见重生成当前章节", type="secondary"):
        feedback = st.session_state[feedback_key].strip()
        current_translation = st.session_state[translation_key].strip()
        if not feedback:
            st.warning("请先填写修改意见。")
            return
        if not current_translation:
            st.warning("当前章节还没有译文，请先执行“翻译当前章节”。")
            return
        try:
            revised = workflow.translator.revise_translation(
                chapter,
                outline=st.session_state["outline_text"],
                characters=st.session_state["characters_text"],
                current_translation=current_translation,
                feedback=feedback,
            )
            save_text(target_path, revised)
            workflow.rag.index_translation(chapter, revised)
            st.session_state[translation_key] = revised
            st.success(f"重生成完成: {target_path.name}")
        except Exception as exc:
            st.error(f"重生成失败: {exc}")


if __name__ == "__main__":
    main()
