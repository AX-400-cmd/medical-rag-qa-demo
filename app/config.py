# -*- coding: utf-8 -*-
"""
配置中心：所有可调项都从环境变量读取，未配置时走安全的离线默认值。

两套可选外部能力（都不强制）：
  - LLM：配置了 LLM_BASE_URL + LLM_API_KEY 后，检索后的“生成”与“重排”会调用
         兼容 OpenAI 的对话接口；否则退回到基于检索片段的“接地回答”。
  - Embedding：配置了 EMBED_BASE_URL + EMBED_API_KEY 后，向量化使用稠密向量；
         否则使用本地 TF-IDF 稀疏向量（零依赖、可解释、离线可跑）。
"""
import os


class Settings:
    # —— LLM（可选）——
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

    # —— Embedding（可选，不配则用本地 TF-IDF）——
    embed_base_url: str = os.getenv("EMBED_BASE_URL", "")
    embed_api_key: str = os.getenv("EMBED_API_KEY", "")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # —— 检索 / 切分 ——
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "260"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))

    # —— 医疗安全阈值 ——
    # 检索置信度低于该值，强制“建议咨询专业医师”，不下结论。
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.18"))

    def has_llm(self) -> bool:
        return bool(self.llm_base_url and self.llm_api_key)

    def has_dense_embed(self) -> bool:
        return bool(self.embed_base_url and self.embed_api_key)


settings = Settings()
