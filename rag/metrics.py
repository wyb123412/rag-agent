"""检索评测指标：Recall@k / MRR / nDCG@k（纯函数、无外部依赖，供 eval.py 与 beir_eval.py 复用）。"""
import math


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """相关文档中，出现在前 k 名结果里的比例。"""
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(rel & set(retrieved[:k])) / len(rel)


def mrr(relevant: list[str], retrieved: list[str], k: int | None = None) -> float:
    """第一个相关文档名次的倒数（MRR）；给定 k 则只在前 k 名内找。"""
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(relevant: dict[str, float], retrieved: list[str], k: int) -> float:
    """nDCG@k：relevant 为 {doc_id: 相关度分数}，retrieved 为排序后的 doc_id 列表。"""
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    if idcg == 0.0:
        return 0.0
    dcg = sum(relevant.get(doc, 0.0) / math.log2(i + 2) for i, doc in enumerate(retrieved[:k]))
    return dcg / idcg
