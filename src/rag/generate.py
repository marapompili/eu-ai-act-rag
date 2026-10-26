import re
import sys
from dataclasses import dataclass
from functools import partial

import httpx

from rag.search import Hit, Retriever

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
NOT_FOUND = "I cannot find this in the provided text."

SYSTEM_PROMPT = f"""You answer questions about the EU AI Act (Regulation (EU) 2024/1689).
Use ONLY the context given by the user. Do not use outside knowledge.
Answer in 2 to 5 sentences, or a short list, stating the actual content of the rules.
Never answer with only a citation or only point letters such as (a), (b).
After each statement, cite its article in square brackets, like [Art. 9] or [Art. 99(3)].
If the context does not contain the answer, reply exactly: {NOT_FOUND}
This is not legal advice."""


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    context = "\n\n".join(f"[Art. {h.chunk.article}]\n{h.chunk.text}" for h in hits)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ask_ollama(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    url: str = OLLAMA_URL,
    timeout: float = 180.0,
) -> str:
    response = httpx.post(
        f"{url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def extract_citations(text: str) -> list[int]:
    """Article numbers cited as [Art. N]. Limit: '[Art. 5, Art. 6]' is not recognised."""
    return sorted({int(n) for n in re.findall(r"\[Art\.\s*(\d+)[^\]]*\]", text)})


@dataclass
class Answer:
    question: str
    text: str
    retrieved: list[int]  # articles given to the model
    cited: list[int]  # articles the model cited
    unsupported: list[int]  # cited but never given: likely invented
    uncited: bool  # answered, but with no recognised [Art. N] citation


def answer(question: str, retriever: Retriever, llm, k: int = 4) -> Answer:
    hits = retriever.search(question, k=k)
    text = llm(build_messages(question, hits)).strip()
    retrieved = sorted({h.chunk.article for h in hits})
    cited = extract_citations(text)
    unsupported = [a for a in cited if a not in retrieved]
    uncited = not cited and text != NOT_FOUND
    return Answer(question, text, retrieved, cited, unsupported, uncited)

if __name__ == "__main__":
    from rag.index import load_model, make_encoder

    retriever = Retriever.load(make_encoder(load_model()))
    res = answer(" ".join(sys.argv[1:]), retriever, partial(ask_ollama))
    print(res.text)
    print("\nretrieved:", res.retrieved, "| cited:", res.cited, "| unsupported:", res.unsupported, "| uncited:", res.uncited)
