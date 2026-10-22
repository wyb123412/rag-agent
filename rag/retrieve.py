"""模块 5：检索 —— 用问题向量去库里找最相关的切片。"""
from .config import TOP_K


def retrieve(collection, query_vector: list[float], top_k: int = TOP_K) -> dict:
    """返回与问题最相似的 top_k 个切片，带来源和相似度距离。"""
    res = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    # Chroma 返回的是「每个查询」一组的结果，这里只有一个查询，取第 0 组
    return {
        "documents": res["documents"][0],
        "metadatas": res["metadatas"][0],
        "distances": res["distances"][0],
    }
