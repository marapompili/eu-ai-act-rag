import json
from pathlib import Path


def load(tag):
    path = Path(f"eval/results/answers_dev_none_index_{tag}.jsonl")
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


a, b = load("k4"), load("k8")
same = 0
for i in a:
    x, y = a[i], b[i]
    if x["answer"] == y["answer"]:
        same += 1
        continue
    print("=" * 70)
    print(i, x["question"])
    print(f'cited {x["cited"]} -> {y["cited"]} | refused {x["refused"]} -> {y["refused"]} | '
          f'facts {len(x["facts_found"])} -> {len(y["facts_found"])} of {len(x["key_facts"])} | '
          f'seconds {x["seconds"]} -> {y["seconds"]}')
    print("--- k=4:", x["answer"][:700])
    print("--- k=8:", y["answer"][:700])
print("\nidentical answers:", same, "of", len(a))
