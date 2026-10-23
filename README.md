# 个人知识库问答 Agent（RAG）

一个从零手写的 RAG 问答 Agent：**混合检索 + 重排序 + 知识图谱 + 语义切片**、**查询改写（HyDE / 多 Query）**、**Corrective RAG（相关性自检重检索）**、**RAPTOR 分层摘要块**、**工具调用 + 多轮记忆 + 流式输出**，并带**可复现评测（检索消融 + RAGAS 风格生成质量）**与 **BEIR 基准脚本**。全程手写检索循环（不依赖 LangChain），适合看清 RAG 每一步的原理，也是一个面试可讲的完整项目。

## 核心亮点

- **混合检索**：向量（bge-m3）+ BM25 + RRF 融合，可选 Cross-Encoder 重排（bge-reranker-v2-m3）
- **语义切片**：按 Markdown 标题切块、每块自带标题上下文，代码块内的 `#` 不会误切
- **Corrective RAG**：检索后 LLM 相关性自检，不相关则改写查询重检索，降低答非所问
- **RAPTOR 分层索引（可选）**：相似切片贪心聚类 + LLM 摘要成「摘要块」，支撑全局性问题
- **知识图谱**：手写轻量实体关系图，LLM 抽取三元组 + 一跳邻居扩展，与向量/BM25 三路融合
- **查询改写**：HyDE（假设文档）与多 Query 生成，提升召回
- **Agent 工具调用**：模型自主判断「检索回答 / 文档摘要 / 字数统计」
- **多轮记忆 + 流式输出 + 耗时追踪**：滑动窗口 + 摘要压缩；逐字输出；检索链路分步计时
- **可复现评测**：检索消融（Recall/MRR）+ RAGAS 风格生成质量（忠实度 / 相关性 / 上下文精度）
- **BEIR 基准脚本**：本地加载 BEIR 数据集，算 nDCG@10 / Recall@100
- **多格式文档**：Markdown / TXT / PDF / Word，图片（jpg/png 等）自动 OCR 转文字入库
- **工程化**：Gradio 网页（可选登录鉴权）、54 个 pytest 单测、GitHub Actions CI、Docker + docker-compose 一键部署（非 root 运行 + 健康检查）

## 技术栈

| 环节 | 选择 |
|---|---|
| 对话模型 | DeepSeek（在线 API） |
| 向量化 | 硅基流动 BAAI/bge-m3（在线 API） |
| 重排序 | 硅基流动 BAAI/bge-reranker-v2-m3（在线 API） |
| 图片 OCR | 硅基流动 Qwen3-VL-8B（在线 API，可换 DeepSeek-OCR） |
| 向量库 | Chroma（本地持久化） |
| 评测 | 自建评测集 + LLM-as-Judge（RAGAS 风格，手写） |
| 界面 | Gradio |
| 部署 | Docker + docker-compose + GitHub Actions |
| 语言 | Python 3.13 |

## 目录结构

```
rag-agent/
├── rag/
│   ├── config.py     # 配置、客户端工厂、日志
│   ├── load.py       # 文档加载（md/txt/pdf/docx/图片）
│   ├── ocr.py        # 图片 OCR（视觉模型转纯文本）
│   ├── chunk.py      # 切片（固定大小 / 按标题语义切片）
│   ├── embed.py      # 向量化
│   ├── store.py      # Chroma 向量库
│   ├── retrieve.py   # 纯向量检索
│   ├── hybrid.py     # 混合检索（BM25 + RRF）+ 多 Query 合并
│   ├── rerank.py     # 重排序
│   ├── rewrite.py    # 查询改写（HyDE / 多 Query）
│   ├── corrective.py # Corrective RAG（相关性自检 + 改写重检索）
│   ├── raptor.py     # RAPTOR（贪心聚类 + 摘要块）
│   ├── graph.py      # 知识图谱（实体关系抽取 + 邻居扩展）
│   ├── metrics.py    # 检索指标（Recall / MRR / nDCG）
│   ├── gen_eval.py   # 生成质量评测（RAGAS 风格，手写 LLM-as-Judge）
│   ├── trace.py      # 检索链路耗时追踪
│   ├── tools.py      # 工具：检索/摘要/字数
│   └── agent.py      # Agent：function calling + ReAct + 多轮记忆 + 流式
├── ingest.py         # 一键入库（--raptor 可选摘要块）
├── graph_build.py    # 构建知识图谱
├── chat.py           # 命令行流式问答
├── app.py            # Gradio 网页
├── eval.py           # 自建评测集 + 消融 + 生成质量
├── beir_eval.py      # BEIR 基准评测
├── .github/workflows/ci.yml  # GitHub Actions 自动跑测试
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

# 3. 入库（默认按标题语义切片；加 --raptor 额外生成分层摘要块）
venv/Scripts/python ingest.py

# 4. （可选）构建知识图谱，让知识问答附带实体关系上下文
venv/Scripts/python graph_build.py

# 5. 命令行问答 / 网页
venv/Scripts/python chat.py
venv/Scripts/python app.py    # 打开 http://localhost:7860
```

## 检索链路

```
问题 ──▶ 查询改写(HyDE/多Query，可选) ──▶ 向量(bge-m3) + BM25 ──RRF 融合──▶ bge-reranker-v2-m3 精排
                                                        ▲
                                        知识图谱(实体关系 + 一跳邻居) 一并注入
                                                        │
                                Corrective RAG：相关性自检，不相关则改写重检索
                                                        ▼
                                              top-k 切片 → LLM 生成
```

## 评测与消融

```bash
venv/Scripts/python eval.py     # 输出五档消融 + RAGAS 风格生成质量，写入 eval_report.md
```

消融档位：`vector → hybrid → hybrid+rerank → hyde+hybrid+rerank → multi+hybrid+rerank`。

**自建评测集（28 条：18 易 + 10 难，top_k=5，语义切片）实测结果**（切片级 Recall：命中答案所在的那一段，而非仅仅命中文件）：

| 模式 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | 文件级 Recall@5 |
|---|---|---|---|---|---|
| vector | 92.86% | 100.00% | 100.00% | 0.964 | 100.00% |
| hybrid | 92.86% | 100.00% | 100.00% | 0.964 | 100.00% |
| hybrid+rerank | 92.86% | 100.00% | 100.00% | 0.964 | 100.00% |
| hyde+hybrid+rerank | 89.29% | 100.00% | 100.00% | 0.946 | 100.00% |
| multi+hybrid+rerank | 92.86% | 100.00% | 100.00% | 0.964 | 100.00% |

**难例子集（10 条改写/干扰题，更能拉开模式差距）：**

| 模式 | Recall@1 | Recall@5 | MRR@5 |
|---|---|---|---|
| vector | 90.00% | 100.00% | 0.950 |
| hybrid | 90.00% | 100.00% | 0.950 |
| hybrid+rerank | **100.00%** | 100.00% | **1.000** |
| hyde+hybrid+rerank | 90.00% | 100.00% | 0.950 |
| multi+hybrid+rerank | 90.00% | 100.00% | 0.950 |

生成质量（RAGAS 风格，hybrid+rerank）：**忠实度 100% · 答案相关性 100% · 上下文精度 22.14%**。

> 结论（均为实测，面试可讲）：
> - 重排的价值体现在**难例**：改写/干扰题上 `hybrid+rerank` 把切片级 Recall@1 从 90% 拉到 **100%**、MRR@5 到 **1.000**——重排能在噪音里把真正相关的那一段顶到最前。
> - HyDE 在本语料上不升反降（89.29%），说明「假设文档改写」并非万能，因此做成可选项而非默认。
> - 上下文精度偏低（top-5 混入「相邻但非本题」的切片）正是引入 Corrective RAG 的动机。追求更客观可比的结果可换 BEIR 基准（见下）。

## BEIR 基准

`beir_eval.py` 加载本地 BEIR 格式数据集，跑混合检索并输出 nDCG@10 / Recall@100：

```bash
# 下载 scifact（小数据集，走 BEIR 官方服务器，国内可直连）
venv/Scripts/python download_scifact.py

# 跑评测（--limit 先冒烟，确认无误再去掉跑全量）
venv/Scripts/python beir_eval.py --data-dir data/beir/scifact --limit 200
```

scifact 上混合检索（bge-m3 + BM25 + RRF，不重排）实测：**nDCG@10 = 0.667 · Recall@100 = 0.954**（300 条 test 查询）。

## Agent 能力

| 你说 | Agent 会 |
|---|---|
| 「过拟合怎么解决？」 | 检索后带 `[1]` 引用回答（含图谱上下文） |
| 「它有哪些解决方法？」（接上文） | 多轮记忆，接上「过拟合」 |
| 「总结 01_python_学习笔记.md」 | 调用摘要工具 |
| 「统计所有文档字数」 | 调用字数统计工具 |

查询改写默认关闭；需要时在 `.env` 里设 `QUERY_REWRITE=hyde` 或 `QUERY_REWRITE=multi` 开启。Corrective RAG 默认开启（`CORRECTIVE_RAG=1`），检索结果被判定不相关时会自动改写查询重检索一次。

## 测试

```bash
venv/Scripts/python -m pytest   # 54 个单测（离线，无需 key）
```

推送 GitHub 后，`.github/workflows/ci.yml` 会在每次 push / PR 自动跑测试。

## 部署

本地 `docker compose up -d --build` 即可；云服务器一键部署见 [deploy.sh](deploy.sh)，完整步骤（域名 + HTTPS 反代、数据持久化）见 [DEPLOY.md](DEPLOY.md)。

## 下一步

- RAPTOR 做多层递归聚类（当前为一层）
- 知识图谱做多跳推理 / 社区检测（更接近完整 GraphRAG）
- 引用来源精确到切片级高亮
- 换 Milvus / 向量量化支撑更大规模
