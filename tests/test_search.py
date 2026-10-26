import zlib

import numpy as np

from rag.chunk import Chunk
from rag.index import build_index
from rag.search import Retriever


def fake_encode(texts, dim=512):
    """Toy encoder: hashed bag of words, unit norm. Fast and deterministic."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, zlib.crc32(w.encode()) % dim] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


CHUNKS = [
    Chunk(5, 0, "Prohibited practices", "social scoring manipulation of behaviour"),
    Chunk(9, 0, "Risk management", "risk management system for high-risk systems"),
    Chunk(99, 0, "Penalties", "administrative fines for infringements"),
]


def make_retriever():
    return Retriever(CHUNKS, fake_encode([c.text for c in CHUNKS]), fake_encode)


def test_most_similar_chunk_is_first():
    hits = make_retriever().search("administrative fines", k=3)
    assert hits[0].chunk.article == 99


def test_scores_are_sorted_descending():
    scores = [h.score for h in make_retriever().search("risk management system")]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_corpus_returns_all():
    assert len(make_retriever().search("anything", k=10)) == 3


def test_build_and_load_roundtrip(tmp_path):
    build_index(CHUNKS, fake_encode, out_dir=tmp_path)
    loaded = Retriever.load(fake_encode, index_dir=tmp_path)
    assert loaded.chunks == CHUNKS
    assert loaded.search("social scoring", k=1)[0].chunk.article == 5