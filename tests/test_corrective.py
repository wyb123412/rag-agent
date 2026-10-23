"""Corrective RAG 的离线单测：相关性判定解析 + 重检索逻辑（用假 llm/searcher，不发请求）。"""
from rag.corrective import corrective_search, is_relevant


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, replies):
        self._replies = list(replies)

    def create(self, **kwargs):
        return _Resp(self._replies.pop(0))


class _FakeLLM:
    def __init__(self, replies):
        self.chat = type("C", (), {"completions": _Completions(replies)})()


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, query, mode="hybrid+rerank", top_k=4, qvec=None):
        self.calls.append(query)
        return {"documents": [f"doc-{query}"], "metadatas": [{"source": "s"}]}


def test_is_relevant():
    assert is_relevant("相关")
    assert is_relevant("判断：相关")
    assert not is_relevant("不相关")
    assert not is_relevant("")


def test_corrective_search_keeps_relevant():
    searcher = FakeSearcher()
    llm = _FakeLLM(["相关"])
    r = corrective_search(searcher, llm, "过拟合怎么解决？", mode="hybrid")
    assert r["documents"] == ["doc-过拟合怎么解决？"]
    assert searcher.calls == ["过拟合怎么解决？"]  # 判定相关，未重检索


def test_corrective_search_reretrieves_when_irrelevant():
    searcher = FakeSearcher()
    llm = _FakeLLM(["不相关", "过拟合 解决方法"])  # 第1次判定不相关，第2次返回改写后的 query
    r = corrective_search(searcher, llm, "过拟合怎么解决？", mode="hybrid")
    assert searcher.calls == ["过拟合怎么解决？", "过拟合 解决方法"]
    assert r["documents"] == ["doc-过拟合 解决方法"]
