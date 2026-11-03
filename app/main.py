# -*- coding: utf-8 -*-
"""
FastAPI 应用入口：暴露 /health、/ask、/ingest 三个接口。
完整链路：提问 -> 向量检索 top-k -> LLM/分数重排 -> 结构化生成 -> 医疗安全校验 -> 返回。

运行（离线）：  uvicorn app.main:app --reload --port 8000
文档：          http://127.0.0.1:8000/docs
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .knowledge import build_store
from .rerank import rerank
from .generator import generate
from .safety import check, DISCLAIMER
from .schemas import AskRequest, AskResponse, Source

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-rag")

app = FastAPI(title="医疗知识库 RAG 问答服务", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STORE = None
ENTRIES = None


@app.on_event("startup")
def startup():
    global STORE, ENTRIES
    STORE, ENTRIES = build_store()
    logger.info("知识库索引构建完成：片段数=%d，模式=%s",
                len(STORE.chunks), "llm" if settings.has_llm() else "offline")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(STORE.chunks) if STORE else 0,
        "mode": "llm" if settings.has_llm() else "offline",
        "dense_embed": settings.has_dense_embed(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    top_k = req.top_k or settings.top_k
    hits = STORE.query(req.query, top_k)
    hits = rerank(req.query, hits)
    gen = generate(req.query, hits)
    safe = check(req.query, gen["answer"], gen["confidence"])

    sources = [
        Source(
            doc_id=c["doc_id"],
            title=c["title"],
            category=c.get("category", ""),
            snippet=c["text"][:120],
            score=round(s, 3),
        )
        for s, c in hits
    ]

    return AskResponse(
        query=req.query,
        answer=safe["answer"],
        confidence=gen["confidence"],
        category=gen["category"],
        sources=sources,
        needs_human_review=safe["needs_human_review"],
        safety_flags=safe["safety_flags"],
        disclaimer=DISCLAIMER,
        retrieved_count=len(hits),
        mode=gen["mode"],
    )


@app.post("/ingest")
def ingest(entry: dict):
    """简易入库：把一条新文档追加进知识库并重建索引（演示用，生产应接持久化）。"""
    global STORE, ENTRIES
    from .knowledge import load_entries, chunk_entry
    entries = load_entries()
    entries.append(entry)
    chunks = []
    for e in entries:
        chunks.extend(chunk_entry(e))
    STORE.build(chunks)
    ENTRIES = entries
    return {"ok": True, "chunks": len(STORE.chunks)}
