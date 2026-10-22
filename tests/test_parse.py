from pathlib import Path

import pytest

from rag.parse import parse_articles

SAMPLE = """
<div class="eli-subdivision" id="art_6">
  <p class="oj-ti-art">Article 6</p>
  <div class="eli-title" id="art_6.tit_1"><p class="oj-sti-art">Classification rules</p></div>
  <div id="006.001">
    <p class="oj-normal">1.   First paragraph:</p>
    <table><tbody><tr>
      <td><p class="oj-normal">(a)</p></td>
      <td><p class="oj-normal">first point;</p></td>
    </tr></tbody></table>
  </div>
  <div id="006.002"><p class="oj-normal">2.   Second paragraph.</p></div>
</div>
"""


def test_parse_sample_article():
    (art,) = parse_articles(SAMPLE)
    assert art.number == 6
    assert art.title == "Classification rules"
    assert art.text.split("\n") == [
        "1. First paragraph:",
        "(a) first point;",
        "2. Second paragraph.",
    ]


RAW = Path("data/raw/ai_act_oj.html")


@pytest.mark.skipif(not RAW.exists(), reason="raw file not downloaded")
def test_real_file_has_113_articles():
    articles = parse_articles(RAW.read_text(encoding="utf-8"))
    assert [a.number for a in articles] == list(range(1, 114))
    assert all(a.title and a.text for a in articles)