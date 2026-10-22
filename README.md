# 个人知识库问答 Agent（RAG）

一个从零手写的 RAG 问答 Agent：**混合检索 + 重排序**、**工具调用（摘要/字数统计）**、**多轮对话记忆**，并带**可复现评测**。适合看清 RAG 每一步的原理，也是一个面试可讲的完整项目。

## 核心亮点

- **混合检索**：向量（bge-m3）+ BM25 + RRF 融合，可选 Cross-Encoder 重排（bge-reranker-v2-m3）
- **Agent 工具调用**：模型自主判断「检索回答 / 文档摘要 / 字数统计」
- **多轮对话记忆**：滑动窗口 + 摘要压缩，支持指代消解与连续追问
- **可复现评测**：自建评测集，消融实验量化 Recall@1/@3/@5、MRR、忠实度
- **多格式文档**：Markdown / TXT / PDF / Word
- **工程化**：Gradio 网页、pytest 单测、Dockerfile

## 技术栈

| 环节 | 选择 |
|---|---|
| 对话模型 | DeepSeek（在线 API） |
| 向量化 | 硅基流动 BAAI/bge-m3（在线 API） |
| 重排序 | 硅基流动 BAAI/bge-reranker-v2-m3（在线 API） |
| 向量库 | Chroma（本地持久化） |
| 界面 | Gradio |
| 语言 | Python 3.13 |

## 目录结构

```
rag-agent/
├── rag/
│   ├── config.py     # 配置、客户端工厂、日志
│   ├── load.py       # 文档加载（md/txt/pdf/docx）
│   ├── chunk.py      # 切片
│   ├── embed.py      # 向量化
│   ├── store.py      # Chroma 向量库
│   ├── retrieve.py   # 纯向量检索
│   ├── hybrid.py     # 混合检索（BM25 + RRF）
│   ├── rerank.py     # 重排序
│   ├── tools.py      # 工具：检索/摘要/字数
│   └── agent.py      # Agent：function calling + ReAct + 多轮记忆
├── ingest.py         # 一键入库
├── chat.py           # 命令行问答
├── app.py            # Gradio 网页
├── eval.py           # 评测 + 消融实验
├── data/samples/     # 示例文档
├── data/eval_set.json# 评测集
└── tests/            # pytest 单元测试
```

## 快速开始

```bash
# 1. 建虚拟环境并装依赖
python -m venv venv
venv/Scripts/python -m pip install -r requirements.txt   # Windows（mac/linux 用 venv/bin/python）

# 2. 配置 API Key
cp .env.example .env   # 填 LLM_API_KEY 和 EMBED_API_KEY（RERANK 留空会自动复用）

# 3. 入库
venv/Scripts/python ingest.py

# 4. 命令行问答 / 网页
venv/Scripts/python chat.py
venv/Scripts/python app.py    # 打开 http://localhost:7860
```

## 检索链路

```
问题 → 向量检索(bge-m3) + BM25 → RRF 融合 → bge-reranker-v2-m3 精排 → top-k 切片 → LLM 生成
```

## 评测与消融

```bash
venv/Scripts/python eval.py     # 输出三档消融 + 忠实度，写入 eval_report.md
```

实测（10 篇示例语料、18 条问题）：

| 模式 | Recall@1 | Recall@5 | MRR@5 |
|---|---|---|---|
| 纯向量 | 94.4% | 100% | 0.972 |
| 混合检索（向量+BM25+RRF） | 100% | 100% | 1.000 |
| 混合+重排 | 100% | 100% | 1.000 |

> 小语料上向量本身已较强，提升集中在排序（Recall@1、MRR）；更大更噪的语料上混合检索的优势会更明显。

## Agent 能力

| 你说 | Agent 会 |
|---|---|
| 「过拟合怎么解决？」 | 检索后带 `[1]` 引用回答 |
| 「它有哪些解决方法？」（接上文） | 多轮记忆，接上「过拟合」 |
| 「总结 01_python_学习笔记.md」 | 调用摘要工具 |
| 「统计所有文档字数」 | 调用字数统计工具 |

## 测试

```bash
venv/Scripts/python -m pytest
```

## 下一步

- 混合检索接入知识图谱（GraphRAG / LightRAG 思路）
- 查询改写（HyDE / 多 Query 生成）
- 更大更噪的公开语料上做基准对比（BEIR 等）
- 流式输出、Docker 部署到云
