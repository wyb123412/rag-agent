"""查询改写：HyDE（假设文档）与多 Query 生成，在检索前改写问题以提升召回。"""
import json
import logging

from .config import LLM_MODEL, TOP_K
from .embed import embed_query
from .hybrid import search_multi

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = ("你是一个检索增强的查询改写助手。针对用户的问题，写出一段「假设能够回答该问题的文档片段」，"
                "内容通顺、信息点完整，长度不超过 150 字。只输出这段文档，不要解释。")

_MULTI_SYSTEM = ("你是一个查询改写助手。把用户的问题改写成 {n} 个不同的检索问法（同义或不同角度，更利于从知识库召回）。"
                 "输出一个 JSON 字符串数组，形如 [\"问法1\", \"问法2\"]。只输出数组，不要解释。")


def _chat_once(llm, system: str, user: str, temperature: float = 0.3) -> str:
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def hypothetical_document(llm, question: str) -> str:
    """HyDE：生成一段「假设答案文档」，用它替换 query 向量去检索。失败退回原问题。"""
    text = _chat_once(llm, _HYDE_SYSTEM, question)
    return text or question


def generate_queries(llm, question: str, n: int = 3) -> list[str]:
    """多 Query：生成 n 个改写问法（含原问题去重）。解析失败退回原问题。"""
    out = _chat_once(llm, _MULTI_SYSTEM.format(n=n), question)
    try:
        arr = json.loads(out.strip().removeprefix("```json").removesuffix("```").strip())
        if isinstance(arr, list):
            qs = [str(x).strip() for x in arr if str(x).strip()]
            return list(dict.fromkeys([question, *qs]))[: n + 1]
    except json.JSONDecodeError:
        logger.warning("多 Query 生成结果解析失败，退回原问题")
    return [question]


def hyde_search(searcher, llm, embed, question: str, mode: str = "hybrid+rerank", top_k: int = TOP_K) -> dict:
    """HyDE 检索：用假设文档的向量做向量腿，BM25 / 重排仍用原问题。"""
    hypo = hypothetical_document(llm, question)
    qvec = embed_query(embed, hypo)
    return searcher.search(question, mode=mode, top_k=top_k, qvec=qvec)


def multi_search(searcher, llm, question: str, mode: str = "hybrid+rerank", top_k: int = TOP_K) -> dict:
    """多 Query 检索：生成多个问法，分别检索后按 RRF 合并。"""
    queries = generate_queries(llm, question)
    return search_multi(searcher, queries, mode=mode, top_k=top_k)
