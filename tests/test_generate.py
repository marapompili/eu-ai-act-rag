from rag.chunk import Chunk
from rag.generate import NOT_FOUND, answer, build_messages, extract_citations
from rag.search import Hit, Retriever
from tests.test_search import fake_encode

CHUNKS = [
    Chunk(9, 0, "Risk management", "risk management system for high-risk systems"),
    Chunk(99, 0, "Penalties", "administrative fines for infringements"),
]


def make_retriever():
    return Retriever(CHUNKS, fake_encode([c.text for c in CHUNKS]), fake_encode)


def test_prompt_contains_question_context_and_labels():
    hits = [Hit(0.9, CHUNKS[0])]
    system, user = build_messages("What is required?", hits)
    assert system["role"] == "system" and "ONLY" in system["content"]
    assert "[Art. 9]" in user["content"]
    assert "risk management system" in user["content"]
    assert user["content"].endswith("Question: What is required?")


def test_extract_citations():
    assert extract_citations("A [Art. 9]. B [Art.5] and again [Art. 9].") == [5, 9]
    assert extract_citations(NOT_FOUND) == []


def test_answer_flags_invented_citation():
    fake_llm = lambda messages: "Fines exist [Art. 99]. Also [Art. 42]."
    res = answer("administrative fines", make_retriever(), fake_llm, k=2)
    assert res.cited == [42, 99]
    assert res.unsupported == [42]


def test_answer_passes_retrieved_context_to_llm():
    seen = {}

    def fake_llm(messages):
        seen["user"] = messages[1]["content"]
        return NOT_FOUND

    res = answer("administrative fines", make_retriever(), fake_llm, k=1)
    assert "[Art. 99]" in seen["user"]
    assert res.retrieved == [99] and res.cited == []

def test_answer_without_citation_is_flagged():
    res = answer("administrative fines", make_retriever(), lambda m: "Fines exist.", k=1)
    assert res.uncited


def test_refusal_is_not_flagged_as_uncited():
    res = answer("administrative fines", make_retriever(), lambda m: NOT_FOUND, k=1)
    assert not res.uncited

def test_citation_with_paragraph_number_is_recognised():
    assert extract_citations("A [Art. 99(4)] and [Art. 99(3)]. B [Art. 5]") == [5, 99]