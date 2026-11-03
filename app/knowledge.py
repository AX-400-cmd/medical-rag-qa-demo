# -*- coding: utf-8 -*-
"""知识库加载、切块与索引构建。"""
import json
import os
import re

from .config import settings
from .vector_store import VectorStore

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(HERE), "data", "knowledge_base.json")


def load_entries():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def chunk_entry(entry, size=None, overlap=None):
    """按句累加切块，保留 overlap 重叠，避免句子被切断丢语义。"""
    size = size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = entry["content"]
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;\n])", text) if s.strip()]

    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) <= size:
            buf += s
        else:
            if buf:
                chunks.append(buf)
            buf = (buf[-overlap:] if overlap else "") + s
    if buf:
        chunks.append(buf)

    return [
        {
            "doc_id": entry["id"],
            "title": entry["title"],
            "category": entry.get("category", ""),
            "entry_level": entry.get("level", ""),
            "text": c,
            "chunk_idx": i,
        }
        for i, c in enumerate(chunks)
    ]


def build_store():
    """加载知识库 -> 切块 -> 向量化建索引。返回 (store, entries)。"""
    entries = load_entries()
    chunks = []
    for e in entries:
        chunks.extend(chunk_entry(e))
    store = VectorStore()
    store.build(chunks)
    return store, entries
