"""hybrid 检索的离线单测：用假 collection + 假 embed client 验证 RRF 融合（不发请求）。"""
from rag.hybrid import HybridSearcher

DOCS = ["苹果 香蕉 橙子", "苹果 手机 发布会", "汽车 引擎 维修"]


class _D:
    def __init__(self, i):
        self.index = i
        self.embedding = [1.0, 0.0, 0.0]


class _Resp:
    def __init__(self, n):
        self.data = [_D(i) for i in range(n)]


class FakeEmbedClient:
    def __init__(self):
        self.embeddings = self

    def create(self, model, input):
        return _Resp(len(input))


class FakeCollection:
    def __init__(self):
        self.ids = ["d0", "d1", "d2"]
        self.documents = DOCS
        self.metadatas = [{"source": f"f{i}.md"} for i in range(3)]

    def get(self, include=None):
        return {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas}

    def query(self, query_embeddings=None, n_results=1, include=None):
        n = min(n_results, len(self.ids))
        return {
            "ids": [self.ids[:n]],
            "documents": [self.documents[:n]],
            "metadatas": [self.metadatas[:n]],
            "distances": [[0.0] * n],
        }


def _searcher():
    return HybridSearcher(FakeCollection(), FakeEmbedClient())


def test_hybrid_rrf_ranks_docs():
    # 向量和 BM25 都认为 d0、d1 相关（都含"苹果"），d2 无关 → 融合后排前二
    r = _searcher().search("苹果", mode="hybrid", top_k=2)
    assert r["documents"] == DOCS[:2]
    assert [m["source"] for m in r["metadatas"]] == ["f0.md", "f1.md"]


def test_vector_mode_returns_distances():
    r = _searcher().search("苹果", mode="vector", top_k=1)
    assert r["documents"] == ["苹果 香蕉 橙子"]
    assert "distances" in r
