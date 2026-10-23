"""一键入库：加载 → 切片（语义/固定）→ 向量化 → 写入向量库；可选 RAPTOR 摘要块。

用法：python ingest.py [--raptor]
"""
import sys
from pathlib import Path

# Windows 下强制 UTF-8 输出，避免中文乱码
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from rag.config import (DATA_DIR, CHROMA_DIR, CHUNK_MODE, RAPTOR_SIM_THRESHOLD,
                        get_embed_client, get_llm_client)
from rag.load import load_documents
from rag.chunk import chunk_text, semantic_chunk_text
from rag.embed import embed_texts
from rag.store import reset_collection, upsert_chunks


def _chunk(text: str) -> list[str]:
    return semantic_chunk_text(text) if CHUNK_MODE == "semantic" else chunk_text(text)


def ingest(raptor: bool = False) -> int:
    """执行入库（load→chunk→embed→store），返回写入的块数。raptor=True 时额外生成摘要块。"""
    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"⚠️  在 {DATA_DIR} 下没找到文档，请先把文档放进去。")
        return 0

    print(f"📄 找到 {len(docs)} 个文档，开始切片（{CHUNK_MODE}）…")

    ids, chunks, metadatas = [], [], []
    for doc in docs:
        name = Path(doc["path"]).name
        for i, c in enumerate(_chunk(doc["text"])):
            ids.append(f"{name}-{i}")
            chunks.append(c)
            metadatas.append({"source": doc["path"]})

    print(f"✂️  共切成 {len(chunks)} 块，开始向量化…")
    embed_client = get_embed_client()
    vectors = embed_texts(embed_client, chunks)

    # 每次入库都重建集合，避免旧数据/不同维度向量残留导致冲突
    collection = reset_collection(CHROMA_DIR)
    upsert_chunks(collection, ids, chunks, vectors, metadatas)

    n = len(chunks)
    if raptor:
        from rag.raptor import build_summary_blocks
        llm = get_llm_client()
        blocks = build_summary_blocks(llm, chunks, vectors, RAPTOR_SIM_THRESHOLD)
        if blocks:
            s_ids = [f"summary-{i}" for i in range(len(blocks))]
            s_texts = [b["text"] for b in blocks]
            s_metas = [{"source": metadatas[b["members"][0]]["source"], "type": "summary"} for b in blocks]
            s_vectors = embed_texts(embed_client, s_texts)
            upsert_chunks(collection, s_ids, s_texts, s_vectors, s_metas)
            n += len(blocks)
            print(f"🧠 RAPTOR：{len(blocks)} 个摘要块已入库")

    print(f"✅ 入库完成：{n} 块已写入 {CHROMA_DIR}")
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raptor", action="store_true", help="额外生成 RAPTOR 摘要块（聚类 + LLM 摘要）")
    args = ap.parse_args()
    ingest(raptor=args.raptor)


if __name__ == "__main__":
    main()
