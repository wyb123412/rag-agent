"""answer_stream 的离线单测：用假 LLM 流验证「最终答案」与「工具调用→再回答」两条路径。"""
from rag.agent import Agent


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, *deltas):
        self.choices = [_Choice(d) for d in deltas]


class _TC:
    def __init__(self, index, id=None, name=None, args=None):
        self.index = index
        self.id = id
        self.function = None
        if name is not None or args is not None:
            self.function = type("F", (), {"name": name, "arguments": args})()


class _StreamLLM:
    """按调用次数依次返回预设的流。"""

    def __init__(self, streams):
        self._streams = list(streams)
        self.calls = 0
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kwargs):
        chunks = self._streams[self.calls]
        self.calls += 1
        return iter(chunks)


def _make_agent(streams):
    a = Agent.__new__(Agent)  # 跳过 __init__，不建客户端/集合
    a.llm = _StreamLLM(streams)
    a.searcher = None
    a.graph = None
    a.tools = []
    a.history = []
    a.summary = ""
    a.last_answer = ""
    a.last_sources = []
    return a


def test_answer_stream_final_answer():
    chunks = [
        _Chunk(_Delta(content="你好")),
        _Chunk(_Delta(content="，我是")),
        _Chunk(_Delta(content="助手")),
    ]
    a = _make_agent([chunks])
    assert "".join(a.answer_stream("hi")) == "你好，我是助手"
    assert a.last_answer == "你好，我是助手"
    assert a.history[-2:] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "你好，我是助手"},
    ]


def test_answer_stream_executes_tool(monkeypatch):
    # 第一个流：先给 id（function 为空），再给 function.name/arguments —— 模拟真实分片
    tool_stream = [
        _Chunk(_Delta(tool_calls=[_TC(0, id="call_1")])),
        _Chunk(_Delta(tool_calls=[_TC(0, name="count_words", args="{}")])),
    ]
    answer_stream = [_Chunk(_Delta(content="共 19 字"))]

    a = _make_agent([tool_stream, answer_stream])
    monkeypatch.setattr("rag.agent.execute_tool", lambda name, args, ctx: "工具结果：共 19 字")

    assert "".join(a.answer_stream("统计字数")) == "共 19 字"
    assert a.llm.calls == 2  # 第一次工具调用，第二次最终答案
