from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rag.generate import answer, ask_ollama
from rag.index import load_model, make_encoder
from rag.search import Retriever


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=8, description="chunks given to the model")


class AskResponse(BaseModel):
    answer: str
    retrieved: list[int]  # articles given to the model
    cited: list[int]  # articles the model cited
    unsupported: list[int]  # cited but not provided: likely invented
    uncited: bool  # answered without any recognised [Art. N] citation
    disclaimer: str = "Experimental system, not legal advice."


def create_app(retriever=None, llm=None) -> FastAPI:
    """Build the app. Tests pass fake components; production loads the real ones."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if retriever is None:
            try:
                app.state.retriever = Retriever.load(make_encoder(load_model()))
            except FileNotFoundError:
                raise RuntimeError("Index not found. Build it first: python -m rag.index")
        else:
            app.state.retriever = retriever
        app.state.llm = llm if llm is not None else ask_ollama
        yield

    app = FastAPI(title="EU AI Act RAG", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "chunks": len(request.app.state.retriever.chunks)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request):
        state = request.app.state
        try:
            res = answer(req.question, state.retriever, state.llm, k=req.k)
        except httpx.HTTPError:
            raise HTTPException(
                status_code=503, detail="The language model server (Ollama) is not reachable."
            )
        return AskResponse(
            answer=res.text,
            retrieved=res.retrieved,
            cited=res.cited,
            unsupported=res.unsupported,
            uncited=res.uncited,
        )

    return app


app = create_app()  