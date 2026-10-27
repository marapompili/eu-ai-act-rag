import json
import sys
from pathlib import Path

from rag.parse import parse_articles


def norm(s: str) -> str:
    s = s.lower().replace("\u00a0", " ")
    for a, b in (("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"')):
        s = s.replace(a, b)
    return " ".join(s.split())


articles = {
    a.number: a
    for a in parse_articles(Path("data/raw/ai_act_oj.html").read_text(encoding="utf-8"))
}
text = {n: norm(a.title + " " + a.text) for n, a in articles.items()}

if len(sys.argv) == 3 and sys.argv[1] == "--show":
    a = articles[int(sys.argv[2])]
    print(f"Article {a.number} - {a.title}\n{a.text}")
    raise SystemExit

questions = [json.loads(line) for line in Path("eval/questions.jsonl").read_text().splitlines()]
assert len({q["id"] for q in questions}) == len(questions), "duplicate ids"

problems = 0
for q in questions:
    if not q["answerable"]:
        continue
    for fact in q["key_facts"]:
        if not any(fact in text.get(n, "") for n in q["gold_articles"]):
            elsewhere = [n for n, t in text.items() if fact in t][:6]
            print(f'{q["id"]} gold={q["gold_articles"]}: "{fact}" not found; found in articles: {elsewhere}')
            problems += 1

n_ans = sum(q["answerable"] for q in questions)
print(f"\n{len(questions)} questions ({n_ans} answerable), "
      f"{sum(q['split'] == 'dev' for q in questions)} dev / {sum(q['split'] == 'test' for q in questions)} test")
print("phrases not found:", problems)
