from pathlib import Path

import pytest

from rag.chunk import chunk_all, chunk_article
from rag.parse import Article, parse_articles

# five lines of three words each
ART = Article(7, "Sample title", "\n".join(["one two three"] * 5))


def test_short_article_is_one_chunk():
    chunks = chunk_article(ART, max_words=100)
    assert len(chunks) == 1
    assert chunks[0].body == ART.text


def test_long_article_is_split_between_lines():
    chunks = chunk_article(ART, max_words=7)  # two lines (6 words) fit, three (9) don't
    assert [c.body.count("\n") + 1 for c in chunks] == [2, 2, 1]
    assert all(c.n_words <= 7 for c in chunks)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_no_text_is_lost():
    chunks = chunk_article(ART, max_words=7)
    assert "\n".join(c.body for c in chunks) == ART.text


def test_header_and_metadata():
    chunk = chunk_article(ART, max_words=7)[0]
    assert chunk.article == 7
    assert chunk.text.startswith("Article 7 - Sample title\n")


def test_single_long_line_is_kept_whole():
    art = Article(1, "T", "word " * 30)
    (chunk,) = chunk_article(art, max_words=10)
    assert chunk.n_words == 30


RAW = Path("data/raw/ai_act_oj.html")


@pytest.mark.skipif(not RAW.exists(), reason="raw file not downloaded")
def test_real_file_loses_no_text():
    arts = parse_articles(RAW.read_text(encoding="utf-8"))
    chunks = chunk_all(arts)
    for art in arts:
        body = "\n".join(c.body for c in chunks if c.article == art.number)
        assert body == art.text