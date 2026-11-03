# -*- coding: utf-8 -*-
"""接口与结构化输出的数据契约（Pydantic）。

JD 明确要求“结构化输出与校验”“人工接管/可追溯”——这里用强类型 Schema 把
每一次回答都框定为可校验、可溯源的结构，而不是自由文本。
"""
from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """溯源片段：让每一条回答都“有据可查”（可追溯）。"""
    doc_id: str = Field(..., description="来源文档 id")
    title: str = Field(..., description="来源标题")
    category: str = Field("", description="分类")
    snippet: str = Field("", description="片段摘要")
    score: float = Field(0.0, description="相关度（0~1）")


class AskRequest(BaseModel):
    query: str = Field(..., description="用户提问", min_length=1)
    top_k: Optional[int] = Field(None, description="检索片段数，覆盖默认配置")


class AskResponse(BaseModel):
    query: str = Field(..., description="原始问题")
    answer: str = Field(..., description="回答正文（已脱敏）")
    confidence: float = Field(..., description="置信度 0~1")
    category: str = Field("", description="问题分类：科普/用药/急救/营养等")
    sources: List[Source] = Field(default_factory=list, description="引用片段")
    needs_human_review: bool = Field(False, description="是否需要转人工/医师复核")
    safety_flags: List[str] = Field(default_factory=list, description="命中的安全规则")
    disclaimer: str = Field(..., description="医疗免责声明")
    retrieved_count: int = Field(0, description="实际检索到的片段数")
    mode: str = Field("offline", description="llm=调用大模型；offline=离线接地回答")
