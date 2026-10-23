"""模块 1：文档加载 —— 读取目录下所有 Markdown / TXT / PDF / Word / 图片文件。

图片（jpg/png 等）会先走 OCR 提取文字，再进入切片/向量化，让图片也能被检索。
"""
from pathlib import Path

from .ocr import IMAGE_SUFFIXES

SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"} | IMAGE_SUFFIXES


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _read_image(path: Path) -> str:
    from .ocr import ocr_image
    return ocr_image(path)


def list_document_paths(data_dir: str) -> list[str]:
    """只列出可入库文档的路径（不解析、不 OCR），用于工具描述里的文档名列表等场景。"""
    return [
        str(p) for p in sorted(Path(data_dir).rglob("*"))
        if p.is_file() and p.suffix.lower() in SUFFIXES
    ]


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
            elif path.suffix.lower() in IMAGE_SUFFIXES:
                text = _read_image(path)
            else:
                text = path.read_text(encoding="utf-8")
        except Exception:
            continue  # 单个文件解析失败不影响其它文件
        if text and text.strip():
            docs.append({"path": str(path), "text": text})
    return docs
