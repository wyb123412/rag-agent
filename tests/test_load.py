from pathlib import Path

from rag.load import load_documents


def test_load_md_and_txt(tmp_path):
    (tmp_path / "a.md").write_text("# 标题\n内容", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"not a real pdf")  # 无效 pdf 应被跳过
    docs = load_documents(str(tmp_path))
    names = sorted(Path(d["path"]).name for d in docs)
    assert names == ["a.md", "b.txt"]


def test_empty_file_skipped(tmp_path):
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    assert load_documents(str(tmp_path)) == []
