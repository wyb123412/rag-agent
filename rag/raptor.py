"""RAPTOR-lite：对切片向量做贪心聚类，把相似切片合并成「摘要块」再入库，支撑全局性问题。

原理参考 RAPTOR（递归式抽象聚类）：普通切片命中局部细节，摘要块命中跨段落的总结性内容，
检索时两类块一起参与召回。这里只做一层聚类摘要，零第三方依赖（纯 Python 余弦相似度）。
"""
import logging

from .config import LLM_MODEL

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = ("你是文档摘要助手。把下面若干段内容合并概括成一段连贯的中文摘要，"
                     "保留关键事实与结论，不超过 200 字。只输出摘要本身。")


def cosine(a, b) -> float:
    """两个向量的余弦相似度（纯 Python，长度不同按较短的对齐）。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def greedy_cluster(vectors, threshold: float = 0.6) -> list[list[int]]:
    """贪心聚类：每个向量归入第一个「与簇代表相似度 ≥ threshold」的已有簇，否则新建簇。

    返回簇的下标列表（按顺序，非全局最优但足够轻量）。
    """
    clusters: list[list[int]] = []
    for i in range(len(vectors)):
        placed = False
        for c in clusters:
            if cosine(vectors[i], vectors[c[0]]) >= threshold:
                c.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def summarize_chunks(llm, texts: list[str]) -> str:
    """把若干段文本合并摘要成一段，失败返回空串。"""
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": _SUMMARIZE_SYSTEM},
                  {"role": "user", "content": "\n\n---\n\n".join(texts)}],
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def build_summary_blocks(llm, chunks: list[str], vectors, threshold: float = 0.6) -> list[dict]:
    """对 chunks 聚类，只保留含 ≥2 块的簇，各生成一个摘要块。返回 [{"text":..., "members":[...]}]。"""
    blocks = []
    for cluster in greedy_cluster(vectors, threshold):
        if len(cluster) < 2:
            continue
        try:
            summary = summarize_chunks(llm, [chunks[i] for i in cluster])
        except Exception as e:
            logger.warning("摘要块生成失败，跳过该簇：%s", e)
            continue
        if summary:
            blocks.append({"text": summary, "members": cluster})
    return blocks
