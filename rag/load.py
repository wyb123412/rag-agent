"""模块 1：文档加载 —— 读取目录下所有 Markdown / TXT / PDF / Word 文件。"""
from pathlib import Path

SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def load_documents(data_dir: str) -> list[dict]:
    """遍历 data_dir，返回 [{"path": 文件路径, "text": 全文}, ...]。"""
    docs = []
    for path in sorted(Path(data_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                text = _read_pdf(path)
            elif path.suffix.lower() == ".docx":
                text = _read_docx(path)
            else:
                text = path.read_text(encoding="utf-8")
        except Exception:
            continue  # 单个文件解析失败不影响其它文件
        if text and text.strip():
            docs.append({"path": str(path), "text": text})
    return docs
