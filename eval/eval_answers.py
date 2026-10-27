import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

from rag.generate import DEFAULT_MODEL, NOT_FOUND, answer, ask_ollama
from rag.index import load_model, make_encoder
from rag.search import Retriever

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def norm(s: str) -> str:
    s = s.lower().replace("\u00a0", " ")
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        s = s.replace(a, b)
    return " ".join(s.split())


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
ap.add_argument("--prefix", default="none", choices=["none", "bge"])
ap.add_argument("--index-dir", default="data/index")
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="only the first N questions (quick trial)")
ap.add_argument("--final", action="store_true", help="required to touch the test split")
args = ap.parse_args()

if args.split != "dev" and not args.final:
    raise SystemExit("The test split is looked at once, at the very end. Use --final if this is it.")

questions = [json.loads(l) for l in Path("eval/questions.jsonl").read_text().splitlines()]
if args.split != "all":
    questions = [q for q in questions if q["split"] == args.split]
if args.limit:
    questions = questions[: args.limit]

retriever = Retriever.load(
    make_encoder(load_model()),
    index_dir=Path(args.index_dir),
    query_prefix=BGE_PREFIX if args.prefix == "bge" else "",
)

try:  # warm-up: loads the model into memory, so the timings below are not skewed
    ask_ollama([{"role": "user", "content": "Say OK."}])
except httpx.HTTPError:
    raise SystemExit("Ollama is not reachable. Start it and try again.")

records = []
for q in questions:
    t0 = time.time()
    res = answer(q["question"], retriever, ask_ollama, k=args.k)
    secs = time.time() - t0
    text_n = norm(res.text)
    refused = norm(NOT_FOUND) in text_n
    found = [f for f in q["key_facts"] if norm(f) in text_n]
    records.append({
        "id": q["id"], "question": q["question"], "answerable": q["answerable"],
        "gold": q["gold_articles"], "retrieved": res.retrieved, "cited": res.cited,
        "unsupported": res.unsupported, "uncited": res.uncited, "refused": refused,
        "key_facts": q["key_facts"], "facts_found": found, "seconds": round(secs, 1),
        "answer": res.text,
    })
    print(f'{q["id"]} {secs:4.0f}s refused={refused} cited={res.cited} gold={q["gold_articles"]}', flush=True)

ans = [r for r in records if r["answerable"]]
una = [r for r in records if not r["answerable"]]
print(f"\n=== {args.split} | prefix={args.prefix} | index={args.index_dir} | k={args.k} ===")
if ans:
    print(f"answerable questions: {len(ans)}")
    print("  cited at least one article:    ", sum(bool(r["cited"]) for r in ans), "/", len(ans))
    print("  cited a gold article:          ", sum(bool(set(r["cited"]) & set(r["gold"])) for r in ans), "/", len(ans))
    print("  gold article was retrieved:    ", sum(bool(set(r["retrieved"]) & set(r["gold"])) for r in ans), "/", len(ans))
    print("  cited an article not provided: ", sum(bool(r["unsupported"]) for r in ans), "/", len(ans))
    print("  refused although answerable:   ", sum(r["refused"] for r in ans), "/", len(ans))
    cov = [len(r["facts_found"]) / len(r["key_facts"]) for r in ans if r["key_facts"]]
    print(f"  key facts found (rough, lower bound): mean {statistics.mean(cov):.2f}")
if una:
    print(f"unanswerable questions: {len(una)}")
    print("  correctly refused:             ", sum(r["refused"] for r in una), "/", len(una))
print(f"seconds per question: median {statistics.median(r['seconds'] for r in records):.0f}, "
      f"max {max(r['seconds'] for r in records):.0f}")

Path("eval/results").mkdir(exist_ok=True)
tag = f"{args.split}_{args.prefix}_{Path(args.index_dir).name}_k{args.k}"
with open(f"eval/results/answers_{tag}.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

lines = [f"# Answers ({tag})", "",
         f"Model: {DEFAULT_MODEL}, temperature 0, k={args.k}. Fill in the judgement lines.", ""]
for r in records:
    kind = "answerable" if r["answerable"] else "NOT answerable"
    lines += [f"## {r['id']} ({kind}): {r['question']}", "",
              f"- gold articles: {r['gold']} | retrieved: {r['retrieved']} | cited: {r['cited']} "
              f"| not provided: {r['unsupported']} | refused: {r['refused']}",
              f"- key facts found: {len(r['facts_found'])}/{len(r['key_facts'])}"
              + (f" (missing: {[f for f in r['key_facts'] if f not in r['facts_found']]})" if r["key_facts"] else ""),
              f"- time: {r['seconds']} s", ""]
    lines += [f"> {l}" if l.strip() else ">" for l in r["answer"].split("\n")]
    lines += ["", "**Judgement (correct / partial / wrong):** ", ""]
Path(f"eval/results/answers_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\nSaved: eval/results/answers_{tag}.md")
