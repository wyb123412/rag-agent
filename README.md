# 个人知识库问答 Agent（RAG）

一个从零手写的 RAG 问答 Agent：**混合检索 + 重排序 + 知识图谱**、**查询改写（HyDE / 多 Query）**、**工具调用（摘要/字数统计）**、**多轮对话记忆 + 流式输出**，并带**可复现评测**与 **BEIR 基准脚本**。适合看清 RAG 每一步的原理，也是一个面试可讲的完整项目。

## 核心亮点

- **混合检索**：向量（bge-m3）+ BM25 + RRF 融合，可选 Cross-Encoder 重排（bge-reranker-v2-m3）
- **知识图谱**：手写轻量实体关系图，LLM 抽取三元组 + 一跳邻居扩展，与向量/BM25 三路融合
- **查询改写**：HyDE（假设文档）与多 Query 生成，提升召回
- **Agent 工具调用**：模型自主判断「检索回答 / 文档摘要 / 字数统计」
- **多轮记忆 + 流式输出**：滑动窗口 + 摘要压缩，指代消解；网页/命令行逐字输出
- **可复现评测**：自建评测集，消融实验量化 Recall@1/@3/@5、MRR、忠实度
- **BEIR 基准脚本**：本地加载 BEIR 数据集，算 nDCG@10 / Recall@100
- **多格式文档**：Markdown / TXT / PDF / Word，图片（jpg/png 等）自动 OCR 转文字入库
- **工程化**：Gradio 网页、pytest 单测、Docker + docker-compose 一键部署

## 技术栈

| 环节 | 选择 |
|---|---|
| 对话模型 | DeepSeek（在线 API） |
| 向量化 | 硅基流动 BAAI/bge-m3（在线 API） |
| 重排序 | 硅基流动 BAAI/bge-reranker-v2-m3（在线 API） |
| 图片 OCR | 硅基流动 Qwen3-VL-8B（在线 API，可换 DeepSeek-OCR） |
| 向量库 | Chroma（本地持久化） |
| 界面 | Gradio |
| 部署 | Docker + docker-compose |
| 语言 | Python 3.13 |

## 目录结构

```
rag-agent/
├── rag/
│   ├── config.py     # 配置、客户端工厂、日志
│   ├── load.py       # 文档加载（md/txt/pdf/docx/图片）
│   ├── ocr.py        # 图片 OCR（视觉模型转纯文本）
│   ├── chunk.py      # 切片
│   ├── embed.py      # 向量化
│   ├── store.py      # Chroma 向量库
│   ├── retrieve.py   # 纯向量检索
│   ├── hybrid.py     # 混合检索（BM25 + RRF）+ 多 Query 合并
│   ├── rerank.py     # 重排序
│   ├── rewrite.py    # 查询改写（HyDE / 多 Query）
│   ├── graph.py      # 知识图谱（实体关系抽取 + 邻居扩展）
│   ├── metrics.py    # 评测指标（Recall / MRR / nDCG）
│   ├── tools.py      # 工具：检索/摘要/字数
│   └── agent.py      # Agent：function calling + ReAct + 多轮记忆 + 流式
├── ingest.py         # 一键入库
├── graph_build.py    # 构建知识图谱
├── chat.py           # 命令行流式问答
├── app.py            # Gradio 网页
├── eval.py           # 自建评测集 + 消融实验
├── beir_eval.py      # BEIR 基准评测
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

# 4. （可选）构建知识图谱，让知识问答附带实体关系上下文
venv/Scripts/python graph_build.py

# 5. 命令行问答 / 网页
venv/Scripts/python chat.py
venv/Scripts/python app.py    # 打开 http://localhost:7860
```

## 检索链路

```
问题 ──查询改写(HyDE/多Query，可选)──▶ 向量(bge-m3) + BM25 ──RRF 融合──▶ bge-reranker-v2-m3 精排
                                        ▲
                        知识图谱(实体关系 + 一跳邻居) 一并注入 ──▶ top-k 切片 → LLM 生成
```

## 评测与消融

```bash
venv/Scripts/python eval.py     # 输出五档消融 + 忠实度，写入 eval_report.md
```

消融档位：`vector → hybrid → hybrid+rerank → hyde+hybrid+rerank → multi+hybrid+rerank`。

## BEIR 基准

`beir_eval.py` 加载本地 BEIR 格式数据集，跑混合检索并输出 nDCG@10 / Recall@100：

```bash
# 下载 scifact（小数据集，用 HuggingFace datasets）
venv/Scripts/python -c "from datasets import load_dataset; \
  load_dataset('BeIR/scifact', 'corpus').to_json('data/beir/scifact/corpus.jsonl'); \
  load_dataset('BeIR/scifact', 'queries').to_json('data/beir/scifact/queries.jsonl'); \
  load_dataset('BeIR/scifact', 'qrels').to_json('data/beir/scifact/qrels.jsonl')"

# 跑评测（--limit 先冒烟，确认无误再去掉跑全量）
venv/Scripts/python beir_eval.py --data-dir data/beir/scifact --limit 200
```

## Agent 能力

| 你说 | Agent 会 |
|---|---|
| 「过拟合怎么解决？」 | 检索后带 `[1]` 引用回答（含图谱上下文） |
| 「它有哪些解决方法？」（接上文） | 多轮记忆，接上「过拟合」 |
| 「总结 01_python_学习笔记.md」 | 调用摘要工具 |
| 「统计所有文档字数」 | 调用字数统计工具 |

查询改写默认关闭；需要时在 `.env` 里设 `QUERY_REWRITE=hyde` 或 `QUERY_REWRITE=multi` 开启。

## 测试

```bash
venv/Scripts/python -m pytest
```

## 部署

本地 `docker compose up -d --build` 即可；云服务器部署、HTTPS 反代、数据持久化见 [DEPLOY.md](DEPLOY.md)。

## 下一步

- 知识图谱做多跳推理 / 社区检测（更接近完整 GraphRAG）
- 流式检索过程可视化（把「召回→重排」的过程展示给用户）
- 权限与多用户隔离、引用来源精确到切片级高亮
- 向量量化 / 换 Milvus 等更大规模向量库
