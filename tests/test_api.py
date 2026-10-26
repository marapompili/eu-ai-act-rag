import httpx
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.generate import NOT_FOUND
from tests.test_generate import make_retriever


def client_with(llm):
    return TestClient(create_app(retriever=make_retriever(), llm=llm))


def test_health():
    with client_with(lambda m: NOT_FOUND) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chunks": 2}


def test_ask_returns_answer_and_citation_checks():
    with client_with(lambda m: "Fines exist [Art. 99]. Also [Art. 42].") as c:
        r = c.post("/ask", json={"question": "administrative fines", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["cited"] == [42, 99]
    assert body["unsupported"] == [42]
    assert body["retrieved"] == [9, 99]
    assert body["uncited"] is False


def test_invalid_requests_are_rejected():
    with client_with(lambda m: NOT_FOUND) as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={"question": "fines", "k": 99}).status_code == 422
        assert c.post("/ask", json={}).status_code == 422


def test_llm_down_gives_503():
    def broken(messages):
        raise httpx.ConnectError("no server")

    with client_with(broken) as c:
        r = c.post("/ask", json={"question": "administrative fines"})
    assert r.status_code == 503
