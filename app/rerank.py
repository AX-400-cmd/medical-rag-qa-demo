# -*- coding: utf-8 -*-
"""
重排序（Rerank）：检索召回 top-k 后，再对片段做一次“相关性精排”，
降低噪声、把最该被引用的片段排到前面。

两档实现：
  - LLM  pointwise 重排：让模型对每段给 0~1 相关性打分（配了 LLM 才用）。
  - 分数兜底重排：无 LLM 时直接沿用检索分数（已按相似度排好）。
无论哪种，都保证离线也能产出稳定可用的排序。
"""
import re

from .config import settings
from .llm import chat


def rerank(query: str, hits):
    """hits: List[(score, chunk)] -> 重排后的 List[(score, chunk)]。"""
    if settings.has_llm():
        try:
            return _rerank_llm(query, hits)
        except Exception as e:  # 任何 LLM 异常都安全降级
            print(f"[rerank] LLM 重排失败，降级为分数重排：{e}")
    return _rerank_score(query, hits)


def _rerank_score(query, hits):
    # 检索分数本身已按相似度降序；做一层关键字重叠微调即可。
    def boost(score, chunk):
        overlap = sum(1 for kw in set(query) if kw in chunk["text"])
        return score + min(overlap, 5) * 0.01
    scored = [(boost(s, c), c) for s, c in hits]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _rerank_llm(query: str, hits):
    out = []
    for s, c in hits:
        prompt = (
            "你是检索相关性裁判。请判断下面【片段】与【问题】的相关程度，"
            "只回复一个 0 到 1 之间的小数（1 表示完全相关，0 表示无关），不要解释。\n"
            f"问题：{query}\n片段：{c['text'][:400]}"
        )
        try:
            raw = chat([{"role": "user", "content": prompt}], temperature=0.0)
            m = re.search(r"0?\.\d+|\d+", raw)
            rel = float(m.group(0)) if m else s
        except Exception:
            rel = s
        out.append((float(rel), c))
    out.sort(key=lambda x: x[0], reverse=True)
    return out
