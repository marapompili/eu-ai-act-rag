import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunk import Chunk
from rag.index import INDEX_DIR


@dataclass
class Hit:
    score: float
    chunk: Chunk


class Retriever:
    def __init__(
        self, chunks: list[Chunk], vectors: np.ndarray, encode, query_prefix: str = ""
    ):
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self.chunks = chunks
        self.vectors = vectors
        self.encode = encode
        self.query_prefix = query_prefix  # applied to questions only, never to chunks

    @classmethod
    def load(cls, encode, index_dir: Path = INDEX_DIR, query_prefix: str = "") -> "Retriever":
        vectors = np.load(index_dir / "vectors.npy")
        meta = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        return cls([Chunk(**m) for m in meta], vectors, encode, query_prefix)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q = self.encode([self.query_prefix + query])[0]
        scores = self.vectors @ q  # cosine similarity: all vectors have norm 1
        top = np.argsort(-scores)[:k]
        return [Hit(float(scores[i]), self.chunks[i]) for i in top]