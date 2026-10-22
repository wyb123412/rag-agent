"""混合检索：向量 + BM25（RRF 融合）+ 可选重排序，提供三种可切换的检索模式。

- vector         纯向量检索（等于原来的 retrieve）
- hybrid         向量 + BM25，用 RRF 融合两者名次
- hybrid+rerank  融合后再用 Cross-Encoder 精排（默认）

BM25 索引从 Chroma 已存切片懒加载构建一次并缓存，避免每次查询重建。
"""
import re

from rank_bm25 import BM25Okapi

from .config import TOP_K, RRF_K
from .embed import embed_query
from .retrieve import retrieve
from .rerank import rerank

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[一-鿿]+")


def tokenize(text: str) -> list[str]:
    """轻量分词：英文/数字词 + 中文二字 bigram，不引入额外分词依赖。"""
    tokens = [m.group().lower() for m in _WORD_RE.finditer(text)]
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class HybridSearcher:
    """封装 BM25 索引与三种检索模式，实例化后复用于多个查询。"""

    def __init__(self, collection, embed_client):
        self.collection = collection
        self.embed = embed_client
        self._ids = self._documents = self._metadatas = None
        self._bm25 = None

    def _ensure_index(self):
        """从 Chroma 里拉出全部切片，构建 BM25 索引（只做一次）。"""
        if self._documents is not None:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        self._ids = data["ids"]
        self._documents = data["documents"]
        self._metadatas = data["metadatas"]
        if self._documents:
            self._bm25 = BM25Okapi([tokenize(d) for d in self._documents])

    def search(self, query: str, mode: str = "hybrid+rerank", top_k: int = TOP_K) -> dict:
        """统一检索入口，返回 {"documents": [...], "metadatas": [...]}。"""
        qvec = embed_query(self.embed, query)

        if mode == "vector":
            return retrieve(self.collection, qvec, top_k=top_k)

        self._ensure_index()
        if not self._documents:  # 空库退化到纯向量
            return retrieve(self.collection, qvec, top_k=top_k)

        # 向量候选（直接查 collection 拿 ids，便于按 id 融合）
        vres = self.collection.query(
            query_embeddings=[qvec], n_results=top_k * 2, include=["documents", "metadatas"]
        )
        vec_ids = vres["ids"][0]

        # BM25 候选
        bm_scores = self._bm25.get_scores(tokenize(query))
        bm_top_idx = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[: top_k * 2]

        # RRF 融合：score = Σ 1/(k + rank)，对两个来源的名次求和
        rrf: dict[str, float] = {}
        for rank, cid in enumerate(vec_ids):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm_top_idx):
            cid = self._ids[idx]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

        id_to = {cid: i for i, cid in enumerate(self._ids)}
        docs = [self._documents[id_to[cid]] for cid in ranked_ids]
        metas = [self._metadatas[id_to[cid]] for cid in ranked_ids]

        if mode == "hybrid+rerank":
            pairs = rerank(query, docs, top_n=top_k)
            if pairs:  # 重排失败返回空，则保持融合后的顺序
                docs = [docs[i] for i, _ in pairs]
                metas = [metas[i] for i, _ in pairs]

        return {"documents": docs, "metadatas": metas}
