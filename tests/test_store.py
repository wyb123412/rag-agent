"""store + retrieve 的离线单测：用本地 Chroma + 假向量验证「入库 → 检索」闭环，无需 API key。"""
from rag.store import get_collection, reset_collection, upsert_chunks
from rag.retrieve import retrieve


def test_upsert_and_retrieve_roundtrip(tmp_path):
    coll = reset_collection(str(tmp_path))
    upsert_chunks(
        coll,
        ids=["a", "b"],
        chunks=["苹果是一种水果", "香蕉也是水果"],
        vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        metadatas=[{"source": "f1.md"}, {"source": "f2.md"}],
    )
    r = retrieve(coll, [1.0, 0.0, 0.0, 0.0], top_k=1)
    assert r["documents"] == ["苹果是一种水果"]
    assert r["metadatas"][0]["source"] == "f1.md"
    assert isinstance(r["distances"][0], float)


def test_reset_clears_old_data(tmp_path):
    coll = get_collection(str(tmp_path))
    upsert_chunks(coll, ["a"], ["旧数据"], [[1.0, 0.0, 0.0, 0.0]], [{"source": "old.md"}])
    reset_collection(str(tmp_path))
    assert get_collection(str(tmp_path)).count() == 0
