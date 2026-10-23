"""Corrective RAG：检索后对结果做相关性自检，不相关则改写查询重检索（最多一次），
降低「检索到 → 答非所问」的概率。相关性与改写的判定都走 LLM-as-Judge，失败自动降级。"""
import logging

from .config import LLM_MODEL, TOP_K, CORRECTIVE_RAG

logger = logging.getLogger(__name__)

_RELEVANCE_SYSTEM = ("你是检索相关性评测员。判断【参考资料】是否包含能回答【问题】的实质内容，"
                     "不要求措辞一致，只看有没有可支撑回答的信息。只输出两个词之一：相关 或 不相关。")

_REWRITE_SYSTEM = "你是检索查询改写助手。把用户问题改写成更适合从知识库检索的简短问法或关键词，只输出改写后的问法。"


def relevance_verdict(llm, question: str, documents: list[str]) -> str:
    """让 LLM 判断检索到的资料是否能支撑回答，返回原始判定词。"""
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _RELEVANCE_SYSTEM},
            {"role": "user", "content": f"问题：{question}\n\n参考资料：\n" + "\n\n".join(documents)},
        ],
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


def is_relevant(verdict: str) -> bool:
    """纯函数：判定「相关」，且排除「不相关」这类否定式。"""
    return "相关" in verdict and "不相关" not in verdict


def rewrite_question(llm, question: str) -> str:
    """改写问题，用于相关性不足时的重检索。失败退回原问题。"""
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": _REWRITE_SYSTEM}, {"role": "user", "content": question}],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or question).strip() or question


def corrective_search(searcher, llm, question: str, mode: str = "hybrid+rerank",
                      top_k: int = TOP_K, max_retries: int = 1) -> dict:
    """检索 + 相关性自检：判定不相关则改写重检索，最多 max_retries 次。任何异常都降级为原结果。"""
    r = searcher.search(question, mode=mode, top_k=top_k)
    if not r.get("documents"):
        return r
    for _ in range(max_retries):
        try:
            verdict = relevance_verdict(llm, question, r["documents"])
        except Exception as e:
            logger.warning("相关性自检失败，跳过：%s", e)
            break
        if is_relevant(verdict):
            break
        try:
            new_q = rewrite_question(llm, question)
        except Exception as e:
            logger.warning("改写查询失败，保留原结果：%s", e)
            break
        if new_q == question:
            break
        r2 = searcher.search(new_q, mode=mode, top_k=top_k)
        if not r2.get("documents"):
            break
        r = r2
    return r
