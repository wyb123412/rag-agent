"""模块 4：向量库 —— 用 Chroma 把「文本 + 向量 + 元信息」持久化。"""
import chromadb


def get_collection(persist_dir: str, name: str = "knowledge"):
    """拿到（或创建）一个持久化的 collection。cosine 表示用余弦相似度。"""
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def reset_collection(persist_dir: str, name: str = "knowledge"):
    """删除旧集合并新建一个空的，用于重新入库（避免维度/旧数据冲突）。"""
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(collection, ids: list[str], chunks: list[str],
                  vectors: list[list[float]], metadatas: list[dict]):
    """把切片写入向量库；upsert 表示 id 已存在就覆盖。"""
    collection.upsert(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
