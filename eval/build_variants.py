import sys
from pathlib import Path

from rag.chunk import chunk_all
from rag.index import build_index, load_model, make_encoder
from rag.parse import parse_articles

arts = parse_articles(Path("data/raw/ai_act_oj.html").read_text(encoding="utf-8"))
encode = make_encoder(load_model())
for size in [int(s) for s in sys.argv[1:]]:
    chunks = chunk_all(arts, max_words=size)
    out = Path(f"data/index_{size}")
    build_index(chunks, encode, out_dir=out)
    print(f"max_words={size}: {len(chunks)} chunks -> {out}")
