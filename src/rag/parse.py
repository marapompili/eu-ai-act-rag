import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

SKIP_CLASSES = {"oj-ti-art", "eli-title"}  # "Article 6" and the title: handled apart


@dataclass
class Article:
    number: int
    title: str
    text: str


def _clean(s: str) -> str:
    return " ".join(s.split())


def _lines(node):
    """Yield the text lines of a node, walking down the HTML tree."""
    for child in node.find_all(recursive=False):
        if SKIP_CLASSES & set(child.get("class") or []):
            continue
        if child.name == "p":
            yield _clean(child.get_text(" "))
        elif child.name == "tr":
            cells = child.find_all("td", recursive=False)
            if len(cells) == 2:  # label cell + body cell, e.g. "(a)" | "text"
                label = _clean(cells[0].get_text(" "))
                body = list(_lines(cells[1]))
                if body:
                    body[0] = f"{label} {body[0]}"
                    yield from body
            else:
                for cell in cells:
                    yield from _lines(cell)
        else:  # div, table, tbody...: keep descending
            yield from _lines(child)


def parse_articles(html: str) -> list[Article]:
    soup = BeautifulSoup(html, "lxml")
    articles = []
    for div in soup.find_all("div", id=re.compile(r"^art_\d+$")):
        title_div = div.find("div", class_="eli-title")
        title = _clean(title_div.get_text(" ")) if title_div else ""
        lines = [line for line in _lines(div) if line]
        articles.append(
            Article(int(div["id"].removeprefix("art_")), title, "\n".join(lines))
        )
    return articles
