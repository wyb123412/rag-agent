"""tools 模块的离线单测：文档列举 / 查找 / 字数统计 / 工具分发，无需 key。"""
from rag import tools


def _write_docs(tmp_path):
    (tmp_path / "a.md").write_text("你好世界", encoding="utf-8")
    (tmp_path / "b.md").write_text("hello world again", encoding="utf-8")
    return tmp_path


def test_list_and_find(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "DATA_DIR", str(_write_docs(tmp_path)))
    assert tools.list_document_names() == ["a.md", "b.md"]
    assert tools._find_document("a.md")["text"] == "你好世界"
    assert tools._find_document("不存在.md") is None


def test_count_words_all_and_single(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "DATA_DIR", str(_write_docs(tmp_path)))
    all_out = tools._count_words({}, {})
    assert "合计：19 字" in all_out  # 你好世界=4 + helloworldagain=15
    one = tools._count_words({}, {"filename": "a.md"})
    assert "a.md：4 字" in one


def test_unknown_tool_returns_friendly_message():
    assert "未知工具" in tools.execute_tool("不存在的工具", {}, {})


def test_build_tool_schemas_has_three_tools(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "DATA_DIR", str(_write_docs(tmp_path)))
    names = {s["function"]["name"] for s in tools.build_tool_schemas()}
    assert names == {"search_knowledge_base", "summarize_document", "count_words"}
