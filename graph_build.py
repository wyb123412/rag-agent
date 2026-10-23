"""构建知识图谱：对 data/samples 下的文档切片后，逐块抽取实体-关系三元组，存为 data/graph.json。

用法：python graph_build.py（需要 LLM key）
"""
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from rag.config import DATA_DIR, GRAPH_PATH, get_llm_client
from rag.load import load_documents
from rag.chunk import chunk_text
from rag.graph import build_graph, save_graph


def main():
    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"⚠️  在 {DATA_DIR} 下没找到文档。")
        return

    chunks = [c for d in docs for c in chunk_text(d["text"])]
    print(f"📄 {len(docs)} 个文档 → {len(chunks)} 个切片，开始抽取实体关系…")

    llm = get_llm_client()
    graph = build_graph(chunks, llm=llm)
    save_graph(GRAPH_PATH, graph)

    print(f"✅ 图谱已保存到 {GRAPH_PATH}")
    print(f"   实体 {len(graph['entities'])} 个，关系 {len(graph['triples'])} 条")


if __name__ == "__main__":
    main()
