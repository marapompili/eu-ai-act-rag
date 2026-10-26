# EU AI Act RAG

A small question-answering assistant over the text of the EU AI Act
(Regulation (EU) 2024/1689). It retrieves the relevant articles, passes them to a
local language model, and answers with article citations.

Built as a learning and portfolio project: retrieval-augmented generation (RAG) with a
local LLM, a REST API, tests, and (planned) a container and a quantitative evaluation.

> **Not legal advice.** This is an experimental system. Answers can be incomplete or wrong
> (see [Known limitations](#known-limitations)). Always check the official text.

## Status

| Part | State |
|---|---|
| HTML parsing of the regulation (113 articles) | done, tested |
| Chunking by article / paragraph (245 chunks) | done, tested |
| Embeddings + cosine-similarity retrieval | done, tested |
| Answer generation with a local LLM (Ollama) and citation check | done, tested |
| REST API (FastAPI): `/health`, `/ask` | done, tested |
| Docker container | not started |
| Evaluation (test questions, metrics, error analysis) | not started; only informal notes in `eval/notes.md` |

## How it works

1. **Parse** (`src/rag/parse.py`): extracts the 113 articles from the EUR-Lex HTML.
2. **Chunk** (`src/rag/chunk.py`): splits long articles between paragraphs into chunks of at
   most 250 words. Each chunk keeps a header ("Article N - Title") and its article number,
   so answers can cite their source.
3. **Index** (`src/rag/index.py`): embeds every chunk with `BAAI/bge-small-en-v1.5`
   (sentence-transformers) and stores the unit-norm vectors in `data/index/`.
4. **Retrieve** (`src/rag/search.py`): embeds the question and ranks chunks by cosine
   similarity (a dot product, since vectors are normalised). With a few hundred chunks a
   NumPy matrix product is enough; no vector database is used.
5. **Generate** (`src/rag/generate.py`): sends the question and the top-k chunks to a local
   model (`llama3.2:3b` through Ollama, temperature 0), asks for citations like `[Art. 9]`,
   and checks which articles were cited and whether they were among the retrieved ones.

The encoder and the language model are passed in as parameters, so the tests use small fake
components and run in about a second without Ollama or the embedding model.

## Setup

Requirements: Python 3.12+, [Ollama](https://ollama.com), about 5 GB of free disk space.
Developed and tested on macOS (Apple silicon, 8 GB RAM).

```bash
git clone https://github.com/marapompili/eu-ai-act-rag.git
cd eu-ai-act-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
```

**Get the data (manual step).** EUR-Lex blocks automated downloads with a bot challenge, so
open https://eur-lex.europa.eu/eli/reg/2024/1689/oj in a browser (English, Official Journal
text of 12.7.2024) and save the page as *HTML only* to `data/raw/ai_act_oj.html`.
See `data/SOURCES.md`.

Build the index, then ask a question:

```bash
PYTHONPATH=src python -m rag.index
PYTHONPATH=src python -m rag.generate "What fines apply for non-compliance?"
```

Start the API (it loads the index and the embedding model once, at startup):

```bash
PYTHONPATH=src uvicorn rag.api:app --port 8000
curl -s -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What fines apply for non-compliance?"}'
```

Interactive documentation is at http://127.0.0.1:8000/docs. Invalid requests get a 422;
if Ollama is not running, `/ask` returns a 503.

Run the tests:

```bash
pytest -v
```

## Known limitations

These are things observed while building the project, not generic disclaimers.

- **Small model, uneven answers.** With `llama3.2:3b`, an early prompt produced answers that
  listed only point letters ("(a), (b) ...") instead of content. A later prompt fixed this
  on my test questions, but answers still shorten or paraphrase the text.
- **Mixed-up articles.** On a question about fines, the model attributed amounts from
  Art. 100 to Art. 99. The citation check does **not** catch this: it only detects
  citations of articles that were not provided to the model, not wrong attributions
  between provided articles.
- **Retrieval is not perfect.** For one question about informing people that they interact
  with an AI, the AI-literacy article ranked above the transparency article.
- **No "I don't know" from retrieval.** It always returns k chunks, even for off-topic
  questions; only the model refuses (it did for an off-topic test question).
- **Citation parsing is simple.** Only `[Art. N]` and `[Art. N(p)]` are recognised;
  `[Art. 5, Art. 6]` counts as article 5 only.
- **Small, informal testing so far.** The observations above come from a handful of
  questions on which the prompt was also adjusted, so they say nothing about overall
  accuracy. A proper evaluation is planned.
- **Original text only.** The regulation has been amended since publication; this project
  uses the original Official Journal text of 12.7.2024 and does not include later changes.
- **Speed.** About 20 to 45 seconds per answer on an 8 GB MacBook (single runs, noisy).
- **English only.**

## Project layout

```
src/rag/       parse, chunk, index, search, generate, api
tests/         pytest tests (fake encoder and fake LLM, no model needed)
eval/          working notes on errors; evaluation will go here
data/          SOURCES.md is versioned; raw/ and index/ are generated locally
```

## Data and license

Code: MIT (see `LICENSE`). The text of the regulation is not redistributed in this
repository: each user downloads it from EUR-Lex (source: European Union, EUR-Lex). See
`data/SOURCES.md`.
