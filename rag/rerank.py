"""重排序：调用硅基流动 /v1/rerank 的 Cross-Encoder，对召回候选做精排。

召回阶段（向量/BM25）速度快但精度有限，重排阶段用更精细的相关性模型
把最相关的候选提到前面。这里用 BAAI/bge-reranker-v2-m3（在线 API）。
"""
import logging

import requests

from .config import RERANK_API_KEY, RERANK_BASE_URL, RERANK_MODEL

logger = logging.getLogger(__name__)


def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
    """对 documents 按与 query 的相关性打分并降序排序。

    返回 [(原下标, 相关性分数), ...]，最多 top_n 个。
    请求失败（网络/额度/超时）时返回空列表，让调用方降级为「不重排」。
    """
    if not documents:
        return []
    try:
        resp = requests.post(
            f"{RERANK_BASE_URL}/rerank",
            headers={"Authorization": f"Bearer {RERANK_API_KEY}"},
            json={"model": RERANK_MODEL, "query": query, "documents": documents},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        # 接口已按分数降序返回；只保留 (原下标, 分数)
        return [(r["index"], r["relevance_score"]) for r in results[:top_n]]
    except Exception as e:
        logger.warning("重排序请求失败，降级为不重排：%s", e)
        return []
