"""模块 2：切片 —— 把长文本切成带重叠的小块。"""
from .config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按行切分后合并成不超过 chunk_size 的块，块与块之间保留 overlap 字符的重叠。

    重叠的作用：避免一句话被从中间硬切开，检索时丢掉上下文。
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: list[str] = []
    buf = ""
    for ln in lines:
        if buf and len(buf) + len(ln) + 1 > chunk_size:
            chunks.append(buf)
            # 把上一块末尾 overlap 个字符接回新块开头
            buf = buf[-overlap:] if overlap > 0 else ""
        buf = f"{buf}\n{ln}".strip()
    if buf:
        chunks.append(buf)
    return chunks
