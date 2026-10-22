from dataclasses import dataclass

from rag.parse import Article


@dataclass
class Chunk:
    article: int  # article number, used for citations
    index: int  # position of the chunk inside its article
    title: str
    body: str

    @property
    def text(self) -> str:
        """What gets embedded: header (context) + body."""
        return f"Article {self.article} - {self.title}\n{self.body}"

    @property
    def n_words(self) -> int:
        return len(self.body.split())


def chunk_article(article: Article, max_words: int = 250) -> list[Chunk]:
    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for line in article.text.split("\n"):
        n = len(line.split())
        if current and count + n > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(line)
        count += n
    if current:
        groups.append(current)
    return [
        Chunk(article.number, i, article.title, "\n".join(lines))
        for i, lines in enumerate(groups)
    ]


def chunk_all(articles: list[Article], max_words: int = 250) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, max_words)]