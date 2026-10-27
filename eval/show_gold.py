import json
from pathlib import Path

from rag.parse import parse_articles


def norm(s):
    return " ".join(s.lower().replace("\u00a0", " ").split())


arts = {a.number: a for a in parse_articles(Path("data/raw/ai_act_oj.html").read_text(encoding="utf-8"))}
qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text().splitlines()]
test = [q for q in qs if q["split"] == "test"]

print("##### ANSWERABLE: does the passage answer the question?\n")
for q in (q for q in test if q["answerable"]):
    print("=" * 70)
    print(q["id"], q["question"], "| gold:", q["gold_articles"])
    for n in q["gold_articles"]:
        print(f"  Art. {n} - {arts[n].title}")
        shown = 0
        for line in arts[n].text.split("\n"):
            if any(f in norm(line) for f in q["key_facts"]):
                print("    >", line[:300])
                shown += 1
                if shown == 2:
                    break

print("\n##### NOT ANSWERABLE: do these words appear in the regulation?\n")
words = {"q29": ["cloud"], "q30": ["director"], "q31": ["fee", "cost"], "q33": ["speed", "vehicle"], "q34": ["austria"]}
for q in (q for q in test if not q["answerable"]):
    for w in words.get(q["id"], []):
        hits = [n for n, a in arts.items() if w in norm(a.text)]
        print(f'{q["id"]} {q["question"]}\n   "{w}" appears in {len(hits)} articles: {hits[:8]}')
