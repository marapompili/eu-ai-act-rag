import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag.chunk import Chunk

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_DIR = Path("data/index")


def load_model(name: str = MODEL_NAME):
    from sentence_transformers import SentenceTransformer  # heavy: import only when needed

    return SentenceTransformer(name)


def make_encoder(model):
    """Wrap a model into a function: list of texts -> matrix of unit-norm vectors."""

    def encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, normalize_embeddings=True, batch_size=32)

    return encode


def build_index(chunks: list[Chunk], encode, out_dir: Path = INDEX_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors = encode([c.text for c in chunks])
    np.save(out_dir / "vectors.npy", vectors)
    meta = [asdict(c) for c in chunks]
    (out_dir / "chunks.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    from rag.chunk import chunk_all
    from rag.parse import parse_articles

    html = Path("data/raw/ai_act_oj.html").read_text(encoding="utf-8")
    chunks = chunk_all(parse_articles(html))
    build_index(chunks, make_encoder(load_model()))
    print(f"Indexed {len(chunks)} chunks in {INDEX_DIR}/")