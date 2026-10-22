from rag.chunk import chunk_text


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
