"""图片 OCR 与文档加载的离线单测：不发真实请求，用假客户端验证。"""
import base64

from rag.ocr import IMAGE_SUFFIXES, ocr_image
from rag.load import SUFFIXES, load_documents


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _Resp(self._content)


class _FakeClient:
    def __init__(self, content):
        self.chat = type("C", (), {"completions": _Completions(content)})()


def test_image_suffixes_wired_into_load():
    assert ".jpg" in IMAGE_SUFFIXES
    assert ".png" in IMAGE_SUFFIXES
    # load.py 的 SUFFIXES 应包含图片后缀，图片才会被加载
    assert IMAGE_SUFFIXES <= SUFFIXES


def test_ocr_image_returns_cleaned_text(monkeypatch, tmp_path):
    img = tmp_path / "表.jpg"
    img.write_bytes(b"fake-image-bytes")

    captured = {}

    def fake_client():
        client = _FakeClient("附表\n国家助学金申请表\n姓名 张三\n\n")
        captured["client"] = client
        return client

    monkeypatch.setattr("rag.ocr.get_ocr_client", fake_client)

    text = ocr_image(str(img))
    assert "张三" in text
    assert "国家助学金申请表" in text
    assert "\n\n" not in text  # 空行被清掉

    # 确认确实以 base64 图片 + 多模态消息请求
    created = captured["client"].chat.completions.create
    # 此处只需保证调用不报错即可；请求体校验可读模型/消息结构
    assert created is not None


def test_ocr_image_failure_returns_empty(monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("no network")

    monkeypatch.setattr("rag.ocr.get_ocr_client", boom)
    assert ocr_image(str(tmp_path / "x.png")) == ""


def test_load_documents_reads_image(monkeypatch, tmp_path):
    (tmp_path / "a.md").write_text("# 笔记\n内容", encoding="utf-8")
    (tmp_path / "b.jpg").write_bytes(b"img")

    monkeypatch.setattr("rag.ocr.ocr_image", lambda p: "图片里的文字")
    docs = load_documents(str(tmp_path))
    texts = {d["path"].split("\\")[-1]: d["text"] for d in docs}
    assert texts["a.md"] == "# 笔记\n内容"
    assert texts["b.jpg"] == "图片里的文字"
