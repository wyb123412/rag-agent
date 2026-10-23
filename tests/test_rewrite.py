"""查询改写的离线单测：多 Query 结果的 RRF 合并（用假 searcher，不发请求）。"""
from rag.hybrid import search_multi


class FakeSearcher:
    def __init__(self, docs_by_query):
        self._docs = docs_by_query

    def search(self, query, mode="hybrid+rerank", top_k=4, qvec=None):
        docs = self._docs.get(query, [])[:top_k]
        return {"documents": docs, "metadatas": [{"source": f"s-{d}"} for d in docs]}


def test_search_multi_merges_by_rrf():
    searcher = FakeSearcher({
        "q1": ["a", "b", "c"],
        "q2": ["a", "d", "c"],
    })
    r = search_multi(searcher, ["q1", "q2"], mode="hybrid", top_k=2)
    # "a" 在两个查询里都排第一，RRF 分数最高
    assert r["documents"][0] == "a"
    assert len(r["documents"]) == 2
    # 元数据与文档一一对应
    assert [m["source"] for m in r["metadatas"]] == [f"s-{d}" for d in r["documents"]]
