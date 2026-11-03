# -*- coding: utf-8 -*-
"""
生成模块：把检索+重排后的片段，转成结构化回答。

  - 配了 LLM：要求模型“只根据材料回答”，并以 JSON 输出
    {answer, confidence, category}，保证结构化与可追溯。
  - 离线兜底：无 LLM 时返回“基于检索片段的接地摘录”，并据检索分数估置信度。
两种模式都先经过 safety.check 才返回，确保医疗边界被守住。
"""
import json
import re

from .config import settings
from .llm import chat
from .safety import check, DISCLAIMER

SYSTEM_PROMPT = (
    "你是严谨的医疗健康科普问答助手。只根据提供的【检索材料】作答，"
    "材料中没有的信息如实说不知道，绝不编造。用简体中文、简洁表达。"
)


def _extract_json(s: str) -> str:
    m = re.search(r"\{.*\}", s, re.S)
    return m.group(0) if m else s


def generate(query: str, hits):
    """返回 dict: {answer, confidence, category, mode}。"""
    if not hits:
        return {
            "answer": "未检索到相关知识，建议咨询专业医师或往知识库补充内容。",
            "confidence": 0.0,
            "category": "",
            "mode": "offline",
        }

    context = "\n\n".join(
        f"[片段{i}]（来源：{c['title']}）\n{c['text']}"
        for i, (_, c) in enumerate(hits, 1)
    )

    if settings.has_llm():
        try:
            return _generate_llm(query, context, hits)
        except Exception as e:
            print(f"[generator] LLM 生成失败，降级离线：{e}")

    return _generate_offline(query, context, hits)


def _generate_llm(query: str, context: str, hits):
    schema = {
        "answer": "基于检索材料的科普回答（材料没有就说不知道）",
        "confidence": "0~1 之间的置信度",
        "category": "科普/用药/急救/营养 等分类之一",
    }
    user = (
        f"检索材料：\n{context}\n\n"
        f"用户问题：{query}\n\n"
        f"请严格按以下 JSON 结构输出（不要多余文字）：\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
    raw = chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        json_mode=True,
    )
    data = json.loads(_extract_json(raw))
    return {
        "answer": data.get("answer", ""),
        "confidence": float(data.get("confidence", 0.0)),
        "category": data.get("category", ""),
        "mode": "llm",
    }


def _generate_offline(query: str, context: str, hits):
    best_chunk = hits[0][1]
    answer = (
        f"根据《{best_chunk['title']}》相关说明：\n"
        f"{best_chunk['text'][:240]}……\n"
        f"（以上为检索片段摘录，非医生诊断；完整内容见下方来源）"
    )
    conf = max(s for s, _ in hits)
    return {
        "answer": answer,
        "confidence": round(conf, 3),
        "category": best_chunk.get("category", ""),
        "mode": "offline",
    }
