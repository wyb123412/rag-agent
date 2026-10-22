"""一键入库：加载 → 切片 → 向量化 → 写入向量库。

用法：python ingest.py
"""
import sys
from pathlib import Path

# Windows 下强制 UTF-8 输出，避免中文乱码
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from rag.config import DATA_DIR, CHROMA_DIR, get_embed_client
from rag.load import load_documents
from rag.chunk import chunk_text
from rag.embed import embed_texts
from rag.store import reset_collection, upsert_chunks


def ingest() -> int:
    """执行入库（load→chunk→embed→store），返回写入的块数。"""
    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"⚠️  在 {DATA_DIR} 下没找到文档，请先把文档放进去。")
        return 0

    print(f"📄 找到 {len(docs)} 个文档，开始切片…")

    ids, chunks, metadatas = [], [], []
    for doc in docs:
        name = Path(doc["path"]).name
        for i, c in enumerate(chunk_text(doc["text"])):
            ids.append(f"{name}-{i}")
            chunks.append(c)
            metadatas.append({"source": doc["path"]})

    print(f"✂️  共切成 {len(chunks)} 块，开始向量化…")
    embed_client = get_embed_client()
    vectors = embed_texts(embed_client, chunks)

    # 每次入库都重建集合，避免旧数据/不同维度向量残留导致冲突
    collection = reset_collection(CHROMA_DIR)
    upsert_chunks(collection, ids, chunks, vectors, metadatas)

    print(f"✅ 入库完成：{len(chunks)} 块已写入 {CHROMA_DIR}")
    return len(chunks)


def main():
    ingest()


if __name__ == "__main__":
    main()
