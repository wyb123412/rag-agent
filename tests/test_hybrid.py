from rag.hybrid import tokenize


def test_tokenize_english_words():
    toks = tokenize("hello world 123")
    assert "hello" in toks and "world" in toks and "123" in toks


def test_tokenize_cjk_bigram():
    toks = tokenize("过拟合")
    assert "过拟" in toks and "拟合" in toks


def test_tokenize_mixed():
    toks = tokenize("python venv 虚拟环境")
    assert "python" in toks and "venv" in toks
    assert "虚拟" in toks and "拟环" in toks and "环境" in toks
