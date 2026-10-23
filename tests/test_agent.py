"""agent 模块的离线单测：参数解析与多轮记忆（跳过 __init__，不发任何请求）。"""
from rag.agent import Agent


def test_parse_args_valid():
    assert Agent._parse_args('{"query": "过拟合"}') == {"query": "过拟合"}


def test_parse_args_invalid_falls_back_to_empty():
    assert Agent._parse_args("不是 json") == {}
    assert Agent._parse_args("") == {}


def test_remember_and_reset():
    a = Agent.__new__(Agent)  # 跳过 __init__，不建客户端/集合
    a.history = []
    a.summary = ""
    a._remember("问题1", "答案1")
    assert a.history == [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "答案1"},
    ]
    a.reset()
    assert a.history == [] and a.summary == ""
