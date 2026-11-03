# -*- coding: utf-8 -*-
"""向量检索：把知识库切块后向量化建索引，查询时做余弦相似度 top-k。

离线模式下向量来自本地 TF-IDF；接入稠密 embedding 后逻辑完全一致，
只是向量维度与语义区分度更高。检索是“可解释”的：每个命中片段都带分数。
"""
import numpy as np

from .embeddings import make_embedder


class VectorStore:
    def __init__(self):
        self.embedder = make_embedder()
        self.chunks = []          # [{doc_id,title,category,entry_level,text,chunk_idx}]
        self.matrix = None        # numpy 矩阵，每行一个片段向量

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        vecs = self.embedder.embed(texts)
        self.matrix = np.array(vecs, dtype=float)
        return self

    def query(self, q: str, top_k: int = 4):
        if self.matrix is None or len(self.matrix) == 0:
            return []
        qv = np.array(self.embedder.embed([q])[0], dtype=float)
        sims = self.matrix @ qv
        qn = np.linalg.norm(qv) or 1.0
        norms = np.linalg.norm(self.matrix, axis=1)
        denom = norms * qn
        denom = np.where(denom == 0, 1.0, denom)
        sims = sims / denom
        k = min(top_k, len(sims))
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.chunks[i]) for i in idx if sims[i] > 0]
