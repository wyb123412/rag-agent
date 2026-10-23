"""rerank 模块的离线单测：空输入、请求失败降级、结果解析（不发真实请求）。"""
import requests

from rag.rerank import rerank


def test_empty_documents_returns_empty():
    assert rerank("问题", [], 3) == []


def test_failure_degrades_to_empty(monkeypatch):
    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("网络不可用")
    monkeypatch.setattr(requests, "post", boom)
    assert rerank("问题", ["文档"], 3) == []


class _Resp:
    def raise_for_status(self):
        pass

    def json(self):
        return {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]}


def test_parse_results_order(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    assert rerank("问题", ["甲", "乙"], 2) == [(1, 0.9), (0, 0.5)]
