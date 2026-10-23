"""知识图谱的离线单测：JSON 解析、建图、邻居扩展（不发请求）。"""
from rag.graph import build_graph, graph_context, parse_extraction_json


def test_parse_json_strips_code_fence():
    d = parse_extraction_json('```json\n{"entities": ["苹果"], "relations": [["苹果", "属于", "水果"]]}\n```')
    assert d["entities"] == ["苹果"]
    assert d["relations"] == [["苹果", "属于", "水果"]]


def test_parse_json_bad_returns_empty():
    assert parse_extraction_json("这不是 json") == {"entities": [], "relations": []}


def test_build_graph_with_fake_extract():
    def fake_extract(chunk):
        return {"entities": ["A", "B"], "relations": [["A", "依赖", "B"]]}

    graph = build_graph(["c0", "c1"], extract_fn=fake_extract)
    assert graph["entities"]["A"] == [0, 1]
    assert len(graph["triples"]) == 2


def test_graph_context_expands_neighbors():
    graph = {
        "entities": {"A": [0], "B": [0], "C": [1]},
        "triples": [("A", "依赖", "B", 0), ("B", "调用", "C", 1)],
    }
    ctx = graph_context(graph, ["A"], top_k=8)
    assert "A" in ctx and "B" in ctx
    assert "依赖" in ctx


def test_graph_context_no_match():
    graph = {"entities": {"A": [0]}, "triples": [("A", "依赖", "B", 0)]}
    assert graph_context(graph, ["无关"]) == ""
