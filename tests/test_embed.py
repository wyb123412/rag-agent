"""embed 模块的离线单测：用假 client 验证「按 index 排回原序」与「32 条一批」。"""
from rag.embed import embed_texts


class _D:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class _Resp:
    def __init__(self, data):
        self.data = data


class FakeEmbedClient:
    """每次 create 依次返回预设的一批 data，并记录调用次数。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.embeddings = self

    def create(self, model, input):
        data = self._responses[self.calls]
        self.calls += 1
        return _Resp(data)


def test_reorders_by_index():
    # 接口把顺序打乱返回（index 1 在前、0 在后），函数应按 index 排回原序
    client = FakeEmbedClient([[_D(1, [1.0]), _D(0, [0.0])]])
    assert embed_texts(client, ["a", "b"]) == [[0.0], [1.0]]


def test_batches_into_32():
    client = FakeEmbedClient([
        [_D(i, [float(i)]) for i in range(32)],
        [_D(32, [32.0])],
    ])
    vecs = embed_texts(client, ["x"] * 33)
    assert len(vecs) == 33
    assert client.calls == 2  # 33 条被拆成 32 + 1 两批
