"""BEIR 基准评测：加载本地 BEIR 格式数据集，跑混合检索（向量+BM25+RRF），输出 nDCG@10 与 Recall@k。

用法：
  python beir_eval.py --data-dir data/beir/scifact --limit 200

数据集目录结构（BEIR 标准）：
  corpus.jsonl   {"_id":..., "title":..., "text":...}
  queries.jsonl  {"_id":..., "text":...}
  qrels/*.tsv    query_id  0  doc_id  relevance
"""
import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import chromadb

from rag.config import get_embed_client
from rag.embed import embed_texts
from rag.hybrid import HybridSearcher
from rag.metrics import ndcg_at_k, recall_at_k


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            qid, _, docid, score = parts[0], parts[1], parts[2], int(parts[3])
            if score > 0:
                qrels.setdefault(qid, {})[docid] = score
    return qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/beir/scifact")
    ap.add_argument("--limit", type=int, default=0, help="限制语料/查询数量，0=全部（用于冒烟测试）")
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    base = Path(args.data_dir)
    corpus = load_jsonl(base / "corpus.jsonl")
    queries = load_jsonl(base / "queries.jsonl")
    qrel_files = list((base / "qrels").glob("*.tsv"))
    if not qrel_files:
        print(f"未找到 {base / 'qrels'} 下的 .tsv 文件。")
        return
    qrels = load_qrels(qrel_files[0])

    if args.limit:
        corpus = corpus[: args.limit]
        queries = queries[: args.limit]

    print(f"语料 {len(corpus)} 篇、查询 {len(queries)} 条，向量化中…")
    embed = get_embed_client()
    ids = [d["_id"] for d in corpus]
    texts = [d["text"] for d in corpus]
    vectors = embed_texts(embed, texts)

    # 用内存 Chroma + HybridSearcher 复用「向量 + BM25 + RRF」检索（不重排，符合 BEIR 惯例）
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="beir", metadata={"hnsw:space": "cosine"})
    collection.upsert(
        ids=ids, documents=texts, embeddings=vectors,
        metadatas=[{"source": d["_id"]} for d in corpus],
    )
    searcher = HybridSearcher(collection, embed)

    ndcg_total = recall_total = 0.0
    n = 0
    for q in queries:
        qid, qtext = q["_id"], q["text"]
        if qid not in qrels:
            continue
        r = searcher.search(qtext, mode="hybrid", top_k=args.k)
        ranked = [m["source"] for m in r["metadatas"]]
        rel = qrels[qid]
        ndcg_total += ndcg_at_k(rel, ranked, 10)
        recall_total += recall_at_k(list(rel), ranked, args.k)
        n += 1

    if n == 0:
        print("没有可评测的查询。")
        return

    ndcg = ndcg_total / n
    recall = recall_total / n
    print(f"查询 {n} 条：nDCG@10 = {ndcg:.4f}，Recall@{args.k} = {recall:.4f}")

    report = (
        f"# BEIR 评测报告\n\n- 数据集：{args.data_dir}\n- 查询：{n} 条\n\n"
        f"| 指标 | 值 |\n|---|---|\n| nDCG@10 | {ndcg:.4f} |\n| Recall@{args.k} | {recall:.4f} |\n"
    )
    Path("beir_report.md").write_text(report, encoding="utf-8")
    print("报告已写入 beir_report.md")


if __name__ == "__main__":
    main()
