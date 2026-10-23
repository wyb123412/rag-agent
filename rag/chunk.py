"""模块 2：切片 —— 把长文本切成带重叠的小块；支持按标题语义切片。"""
import re

from .config import CHUNK_SIZE, CHUNK_OVERLAP

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


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


def semantic_chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                        overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按 Markdown 标题切分，每块自带所属标题作为上下文；无标题则退回固定大小切片。

    好处：切块边界落在「章节」边界上，且每块自带上文标题，检索时不易丢上下文。
    """
    sections = _split_sections(text)
    chunks: list[str] = []
    for heading, body in sections:
        if not body:
            continue
        prefix = f"{heading}\n" if heading else ""
        if len(prefix) + len(body) <= chunk_size:
            chunks.append((prefix + body).strip())
            continue
        # 章节过大：内部按固定大小再切，每块仍带上标题前缀
        for sub in chunk_text(body, chunk_size=max(1, chunk_size - len(prefix)), overlap=overlap):
            chunks.append((prefix + sub).strip())
    return chunks or chunk_text(text, chunk_size, overlap)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按标题把文本切成 [(标题, 正文), ...]；代码块（``` 围住）里的 # 不当作标题。"""
    sections: list[tuple[str, str]] = []
    heading, body_lines, in_fence = "", [], False
    for ln in text.split("\n"):
        stripped = ln.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            body_lines.append(ln)
            continue
        match = None if in_fence else _HEADING_RE.match(stripped)
        if match:
            if body_lines or heading:
                sections.append((heading, "\n".join(body_lines).strip()))
            heading, body_lines = match.group(2).strip(), []
        else:
            body_lines.append(ln)
    if body_lines or heading:
        sections.append((heading, "\n".join(body_lines).strip()))
    return sections
