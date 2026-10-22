"""模块 3：向量化 —— 调用在线 Embedding 接口，把文本变成向量。"""
from openai import OpenAI

from .config import get_embed_client, EMBED_MODEL

_BATCH = 32  # 每批最多送的条数，避免一次请求过大


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """把一组文本转成一组向量，返回顺序与输入一致。"""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        # 接口返回顺序可能被打乱，按 index 排回原序
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([d.embedding for d in ordered])
    return vectors


def embed_query(client: OpenAI, text: str) -> list[float]:
    """把单个问题转成向量。"""
    return embed_texts(client, [text])[0]
