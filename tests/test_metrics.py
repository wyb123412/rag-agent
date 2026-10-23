"""metrics 模块的离线单测。"""
from rag.metrics import recall_at_k, mrr, ndcg_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b"], ["a", "x", "y"], 3) == 0.5
    assert recall_at_k(["a"], ["a"], 1) == 1.0
    assert recall_at_k([], ["a"], 1) == 0.0


def test_mrr():
    assert mrr(["a"], ["x", "a"]) == 0.5
    assert mrr(["a"], ["a", "b"]) == 1.0
    assert mrr(["a"], ["x", "y"]) == 0.0


def test_ndcg_at_k():
    rel = {"a": 3, "b": 2, "c": 1}
    # 完美排序 DCG == IDCG → nDCG = 1
    assert abs(ndcg_at_k(rel, ["a", "b", "c"], 3) - 1.0) < 1e-9
    assert ndcg_at_k({}, ["a"], 1) == 0.0
