from rag.chunk import chunk_text, semantic_chunk_text


def test_short_text_single_chunk():
    assert chunk_text("只有一句话。", chunk_size=1000, overlap=200) == ["只有一句话。"]


def test_long_text_splits():
    text = "\n".join(f"第 {i} 行内容" for i in range(100))
    chunks = chunk_text(text, chunk_size=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks[:-1])


def test_overlap_keeps_tail():
    chunks = chunk_text("AAAA\nBBBB\nCCCC", chunk_size=10, overlap=4)
    assert len(chunks) >= 2
    assert chunks[0][-4:] == chunks[1][:4]


def test_empty_lines_ignored():
    assert chunk_text("\n\nhello\n\nworld\n", chunk_size=1000, overlap=0) == ["hello\nworld"]


# ---- 语义切片 ----

def test_semantic_splits_on_headings():
    text = "# 标题\n\n## 一\n内容A\n\n## 二\n内容B"
    chunks = semantic_chunk_text(text, chunk_size=1000, overlap=0)
    assert any("一" in c and "内容A" in c for c in chunks)
    assert any("二" in c and "内容B" in c for c in chunks)


def test_semantic_prefixes_chunk_with_heading():
    text = "## 装饰器\n装饰器是接收函数返回函数的高阶函数。"
    chunks = semantic_chunk_text(text, chunk_size=1000, overlap=0)
    assert chunks[0].startswith("装饰器\n")


def test_semantic_ignores_hash_in_code_fence():
    text = "# 真标题\n\n```python\n# 这是注释不是标题\nx = 1\n```\n\n正文内容"
    chunks = semantic_chunk_text(text, chunk_size=1000, overlap=0)
    # 代码块里的 # 不应拆出新章节，应留在正文里
    assert any("# 这是注释不是标题" in c for c in chunks)


def test_semantic_falls_back_without_headings():
    text = "没有标题\n" + "\n".join(f"第{i}行" for i in range(50))
    chunks = semantic_chunk_text(text, chunk_size=200, overlap=0)
    assert len(chunks) > 1
