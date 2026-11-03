# -*- coding: utf-8 -*-
"""
向量化模块：两种模式可切换。
  1) 稠密向量（Dense）：配置 EMBED_* 后调用兼容 OpenAI 的 /embeddings。
  2) 本地 TF-IDF 稀疏向量（默认离线）：用“字 + 2-gram”作特征，纯标准库实现，
     可解释、零下载、离线可跑。真实生产中可无缝替换为 BGE / m3e 等中文 embedding。

接口统一为 embed(texts) -> List[List[float]]，供 VectorStore 调用。
"""
import json
import math
import re
import urllib.request
from collections import defaultdict

from .config import settings


def tokenize(text: str):
    """极简中文分词：单字 + 相邻 2-gram，足以体现 TF-IDF 思想。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    tokens = list(text)
    for i in range(len(text) - 1):
        bigram = text[i:i + 2]
        if bigram.strip():
            tokens.append(bigram)
    return [t for t in tokens if t.strip()]


class LocalTFIDF:
    """本地 TF-IDF 向量器，首次 embed 时自动在语料上 fit。"""

    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def fit(self, documents):
        df = defaultdict(int)
        for doc in documents:
            for t in set(tokenize(doc)):
                df[t] += 1
        n = len(documents) or 1
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.vocab = {t: i for i, t in enumerate(self.idf.keys())}
        return self

    def transform(self, text: str):
        toks = tokenize(text)
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        vec = [0.0] * len(self.vocab)
        total = sum(tf.values()) or 1
        for t, c in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = (c / total) * self.idf[t]
        return vec

    def embed(self, texts):
        if not self.vocab:
            self.fit(texts)
        return [self.transform(t) for t in texts]


class APIEmbedder:
    """兼容 OpenAI embeddings 的稠密向量器。"""

    def embed(self, texts):
        url = settings.embed_base_url.rstrip("/")
        if not url.endswith("/embeddings"):
            url = url.rstrip("/") + "/embeddings"
        payload = {"model": settings.embed_model, "input": texts}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {settings.embed_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=40) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [d["embedding"] for d in data["data"]]


def make_embedder():
    """按配置返回向量器；未配稠密向量则默认本地 TF-IDF。"""
    if settings.has_dense_embed():
        return APIEmbedder()
    return LocalTFIDF()
