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
| Docker image (API + embedding model; Ollama on the host) | done, tested manually |
| Evaluation | not started; only informal notes in `eval/notes.md` |

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

Requirements: Python 3.13 (the version used here), [Ollama](https://ollama.com), about 5 GB of free disk space.
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

Run it in Docker (the image contains the API and the embedding model; Ollama stays on the host,
because a 3B model inside a VM does not fit comfortably in 8 GB of RAM). Build the index on the host
first (`python -m rag.index`, see above) and make sure Ollama is running. Tested with
[Colima](https://github.com/abiosoft/colima) on macOS:

```bash
brew install colima docker
colima start --cpu 2 --memory 3 --disk 30 --vm-type vz
docker build -t eu-ai-act-rag .
docker run --rm -p 8000:8000 \
  -v "$PWD/data/index:/app/data/index:ro" \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  eu-ai-act-rag
```

The index is mounted read-only; the Ollama address comes from the `OLLAMA_URL` environment
variable. Stop the VM with `colima stop` when you are done, to free its RAM.

Image size: the first version was 9.8 GB because the default PyTorch wheel pulls about 4 GB of
CUDA libraries that are useless without an NVIDIA GPU. Installing the CPU-only wheel first brought
the image to 2.1 GB (disk usage as reported by `docker images`).

Run the tests:

```bash
pytest -v
```

## Evaluation

34 questions about the regulation: 27 answerable, 7 not answerable from the text (off-topic, or
plausible but absent), split into 12 for development and 22 for a final test. Files: `eval/`.

**How the questions were made.** Drafted with the help of an LLM (Claude). Every key phrase of
every gold article was checked automatically against the text. I checked by hand all test labels
(the 17 answerable and the 5 unanswerable ones: I found no wrong label); the development labels were
checked when the system disagreed with them.

**Protocol.** Experiments (query prefix, chunk size, number of chunks) were run on the development
split only. The test split was run once, with a configuration and rules written in `eval/notes.md`
before the run. Retrieval is scored per article: a hit means that some chunk of the right article is
among the top k. Answers were judged by me against the regulation text (correct / partial / wrong).
Intervals are 95% Wilson intervals; with 10 to 22 questions they are wide.

**Main configuration:** 250-word chunks, no query prefix, 4 chunks given to `llama3.2:3b`, prompt as in the repository.

| Test split (17 answerable) | Result | 95% CI |
|---|---|---|
| Retrieval: right article in the top 4 | 16/17 | 0.73-0.99 |
| Retrieval: right article first | 14/17 | 0.59-0.94 |
| Retrieval: MRR | 0.863 | |
| Answer judged correct | 12/17 | 0.47-0.87 |
| Answer judged correct or partial | 16/17 | 0.73-0.99 |
| Not-answerable questions correctly refused | 5/5 | 0.57-1.00 |
| Time per answer | median 12 s, max 31 s | |

Development split (10 answerable): 2 correct, 6 partial, 2 wrong; 2/2 refusals (both easy questions).

**What the errors look like.**
- *Incomplete lists* (the most frequent): on questions whose answer is a list, the 3B model summarises
  part of it and does not say so.
- *Retrieval miss*: one test question (the extra duties of the most powerful general-purpose models)
  never got its article into the top 4; the model refused instead of inventing, which is a safe failure.
- *Wrong citation, right figure*: one answer gave the right deadline but cited three articles that had
  never been given to the model. The automatic check "cited an article not provided" caught this one.
- *Invented detail* (development split): an answer said that fines for small companies are capped at 50%
  of the maximum, while the cited paragraph says the lower of a percentage and an amount.
- The automatic check "cited a gold article" is not a quality measure: it was wrong in both directions
  (14/17 on the test split, against 12 correct answers by my judgement).

**Other configurations.** On the development split, a query instruction for the embedding model and
chunk sizes of 100 and 150 words gave no difference that can be told apart from noise with 10 questions
(mean reciprocal rank 0.708 at 250 words, 0.765 at 150, 0.787 at 100), so I kept 250 words. With 8 chunks
instead of 4 the answers were more complete in 4 of 10 questions, but the model more often dropped the
citation or refused, and answers took about twice as long. On the test split, 8 chunks gave only automatic
numbers (an article cited in 12/17 answers against 16/17; median time 20 s); I did not judge those answers.

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
- **Original text only.** The regulation has been amended since publication; this project
  uses the original Official Journal text of 12.7.2024 and does not include later changes.
- **Speed.** On an 8 GB MacBook Air, with the model already loaded: median 12 s per answer (max 31 s)
  with 4 chunks, median 20 s with 8. Running the command-line script reloads the embedding model each
  time and takes longer.
- **Docker setup is only tested with Colima on one machine.** Docker Desktop was not tried. The
  container needs Ollama running on the host and an index built beforehand; it does not
  contain the language model.
- **Not a production service.** No authentication, no load testing, and each answer takes
  20 to 45 seconds on this hardware.
- **Small evaluation, one judge.** 17 answerable test questions, judged by me with a rubric written
  after the development judgements; the intervals above show how much that leaves open.
- **LLM-drafted questions.** I checked the test labels, but I only verified that the listed article
  answers the question, not that it is the only one that does.
- **English only.**

## Project layout

```
src/rag/       parse, chunk, index, search, generate, api
Dockerfile     image with the API and the embedding model
tests/         pytest tests (fake encoder and fake LLM, no model needed)
eval/          working notes on errors; evaluation will go here
data/          SOURCES.md is versioned; raw/ and index/ are generated locally
```

## Data and license

Code: MIT (see `LICENSE`). The text of the regulation is not redistributed in this
repository: each user downloads it from EUR-Lex (source: European Union, EUR-Lex). See
`data/SOURCES.md`.
