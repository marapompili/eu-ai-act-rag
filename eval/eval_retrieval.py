import argparse
import json
import math
from pathlib import Path

from rag.index import load_model, make_encoder
from rag.search import Retriever

KS = (1, 3, 4, 5, 10)
BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
ap.add_argument("--prefix", default="none", choices=["none", "bge"])
args = ap.parse_args()

questions = [json.loads(l) for l in Path("eval/questions.jsonl").read_text().splitlines()]
if args.split != "all":
    questions = [q for q in questions if q["split"] == args.split]

retriever = Retriever.load(
    make_encoder(load_model()), query_prefix=BGE_PREFIX if args.prefix == "bge" else ""
)

rows, top1_ans, top1_unans = [], [], []
for q in questions:
    hits = retriever.search(q["question"], k=10)
    if not q["answerable"]:
        top1_unans.append(hits[0].score)
        continue
    top1_ans.append(hits[0].score)
    rank = next(
        (i + 1 for i, h in enumerate(hits) if h.chunk.article in q["gold_articles"]), None
    )
    rows.append((q, rank, [h.chunk.article for h in hits[:4]]))

n = len(rows)
print(f"split={args.split} prefix={args.prefix}: {n} answerable questions\n")
for q, rank, top in rows:
    flag = "" if rank and rank <= 4 else "   <-- check"
    print(f'{q["id"]}  gold={q["gold_articles"]}  rank={rank}  top4 articles={top}{flag}')

print()
for k in KS:
    c = sum(1 for _, r, _ in rows if r is not None and r <= k)
    lo, hi = wilson(c, n)
    print(f"hit@{k:<2} = {c}/{n} = {c / n:.2f}   (95% CI {lo:.2f}-{hi:.2f})")
mrr = sum(1 / r for _, r, _ in rows if r) / n
print(f"MRR    = {mrr:.3f}")

if top1_unans:
    print(
        f"\ntop-1 score, answerable: mean {sum(top1_ans) / len(top1_ans):.2f} | "
        f"unanswerable: mean {sum(top1_unans) / len(top1_unans):.2f} (n={len(top1_unans)})"
    )

Path("eval/results").mkdir(exist_ok=True)
Path(f"eval/results/retrieval_{args.split}_{args.prefix}.json").write_text(
    json.dumps({"split": args.split, "prefix": args.prefix, "n": n, "mrr": mrr,
                "ranks": {q["id"]: r for q, r, _ in rows}}, indent=1)
)
