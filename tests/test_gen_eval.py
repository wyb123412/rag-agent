"""生成质量评测（RAGAS 风格）的离线单测：二元判定与 0/1 数组解析。"""
from rag.gen_eval import parse_binary, parse_int_list


def test_parse_binary():
    assert parse_binary("忠实", "忠实")
    assert not parse_binary("不忠实", "忠实")
    assert parse_binary("相关", "相关")
    assert not parse_binary("不相关", "相关")


def test_parse_int_list():
    assert parse_int_list("[1,0,1]", 3) == [1, 0, 1]
    assert parse_int_list("```json\n[1,0]\n```", 2) == [1, 0]
    assert parse_int_list("不是json", 3) == []
    assert parse_int_list("[1]", 3) == [1, 0, 0]  # 长度不足补 0
