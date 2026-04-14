# 长篇小说翻译工作流（LangChain）

这是一个面向**长篇小说**的翻译脚手架，核心流程包括：

1. **原文切章**：将整本小说按章节拆分并保存。
2. **信息提取**：从原文提取`小说大纲`与`人物设定`。
3. **RAG 向量检索**：为以下内容建立向量索引：
   - 原文全文
   - 原文各章节
   - 小说大纲
   - 人物设定
   - 每章中文翻译
4. **逐章翻译**：翻译每一章时，自动召回`上一章中文译文 + 大纲 + 人物设定 + 相关原文片段`，提升全局一致性。

---

## 目录结构

```text
translator/
├─ main.py
├─ requirements.txt
├─ .env.example
├─ novel_translator/
│  ├─ config.py
│  ├─ splitter.py
│  ├─ rag.py
│  ├─ extractor.py
│  ├─ translator.py
│  └─ workflow.py
└─ data/
   ├─ input/               # 放原始小说 txt
   ├─ output/
   │  ├─ source/           # 原文归档
   │  ├─ chapters/source/  # 拆分后的英文章节
   │  ├─ chapters/zh/      # 中文译文
   │  └─ metadata/         # 大纲、人物设定
   └─ rag_db/              # Chroma 向量库
```

---

## 环境准备

```bash
conda activate langchain
pip install -r requirements.txt
```
如有需要其他库，可以用pip install ***在langchain环境下安装

---

## 使用方式

### 1) 放入原文

将原始小说文本放到：

```text
data/input/novel.txt
```

或者运行时通过 `--input` 指定路径。当前支持：`txt`、`md`、`pdf`、`epub`。
例如仓库内样例：

```bash
python main.py all --input "samples/Code Name Verity.epub" --max-chapters 1
```

### 2) 先做预处理

```bash
python main.py prepare --input data/input/novel.txt
```

会完成：
- 原文归档
- 章节拆分
- 大纲提取
- 人物设定提取
- RAG 向量化

### 3) 再逐章翻译

```bash
python main.py translate --input data/input/novel.txt
```

调试时建议先限制章节数：

```bash
python main.py translate --input data/input/novel.txt --max-chapters 3
```

### 4) 一键全流程

```bash
python main.py all --input data/input/novel.txt
```

### 5) 清理 RAG 向量库目录

```bash
python main.py clean-rag
```

如遇 Windows 文件锁，可增加重试次数：

```bash
python main.py clean-rag --retries 8
```

### 6) Web 前端（可视化工作台）

```bash
streamlit run app_web.py
```

说明：仓库已内置 `.streamlit/config.toml`，默认关闭 `fileWatcher`，用于规避某些环境下
`transformers` 可选依赖（如 `torchvision`）被 Streamlit 监听器误触发的问题。

支持能力：
- 选择或上传源文件（txt/md/pdf/epub）
- 查看并修改翻译系统设置（可写回 `.env`）
- 按流程执行：分析提纲与人物设定、自动切章
- 逐章并列显示原文/译文，手动触发单章翻译
- 对当前章节填写修改意见并重生成译文

### 7) 前后端分层版本（FastAPI + React）

后端 API：

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

前端页面：

```bash
cd frontend
npm install
npm run dev
```

默认访问：
- 前端：`http://localhost:5173`
- 后端：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`

前端可通过环境变量指定 API 地址：

```bash
# frontend/.env.development
VITE_API_BASE_URL=http://localhost:8000
```

---

## 模块说明

| 模块 | 作用 |
| --- | --- |
| `splitter.py` | 识别章节标题并拆分长篇原文 |
| `extractor.py` | 从原文中抽取小说大纲与人物设定 |
| `rag.py` | 构建 Chroma 向量库，管理检索上下文 |
| `translator.py` | 基于 RAG 上下文逐章翻译为中文 |
| `workflow.py` | 串联预处理、向量化与翻译全过程 |

---

## 测试运行

### 统一执行命令（discover）

```bash
conda activate langchain
python -m unittest discover -s unit_tests -p "test_*.py" -v
```

### 测试运行清单

- `unit_tests/test_config.py`：配置读取、路径构造与目录创建
- `unit_tests/test_loader.py`：文件读取/保存/复制与异常分支
- `unit_tests/test_splitter.py`：章节拆分、fallback、章节落盘回读
- `unit_tests/test_extractor.py`：分段抽取与大纲/人物设定解析（mock）
- `unit_tests/test_translator.py`：翻译分段拼装与文本规范化（mock）
- `unit_tests/test_rag.py`：向量检索过滤与上下文拼装（mock）
- `unit_tests/test_workflow.py`：prepare/translate 主流程（mock）
- `unit_tests/test_llm_factory.py`：真实 Ollama 调用验证（integration 风格）

### 说明

- 运行 `test_llm_factory.py` 前，请确认本地 Ollama 服务已启动，且模型已可用。
- 如果仅想执行纯 mock 测试，可临时排除 `test_llm_factory.py` 单独运行。

---

## 建议

- 长篇小说建议先使用 `--max-chapters` 做小规模验证。
- 若你想断点续跑，已有中文章节会默认跳过；加 `--force` 可重译。
- 如果不同章节存在固定称谓、人名译法偏好，可以把这些规则补充到 `data/output/metadata/characters.md` 后再次运行翻译。
