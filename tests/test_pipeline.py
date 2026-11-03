# -*- coding: utf-8 -*-
"""
离线管线自测：不依赖任何 API Key，验证 检索 -> 重排 -> 生成 -> 安全 全链路。
运行：  py -3.12 tests/test_pipeline.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.knowledge import build_store
from app.rerank import rerank
from app.generator import generate
from app.safety import check, DISCLAIMER


def run():
    store, entries = build_store()
    assert len(store.chunks) > 0, "知识库切块为空"
    print(f"[ok] 加载知识库：{len(entries)} 篇文档 -> {len(store.chunks)} 个片段")

    # 1) 普通科普问题
    q = "高血压平时要注意什么"
    hits = store.query(q, top_k=3)
    assert hits, "检索未返回结果"
    hits = rerank(q, hits)
    gen = generate(q, hits)
    safe = check(q, gen["answer"], gen["confidence"])
    assert gen["answer"], "生成回答为空"
    assert safe["answer"], "安全模块返回为空"
    print(f"[ok] 科普问答：mode={gen['mode']} confidence={gen['confidence']} "
          f"needs_human={safe['needs_human_review']}")
    print(f"     答案节选：{gen['answer'][:60]}…")

    # 2) 紧急情形应强制转人工
    s2 = check("突然胸痛呼吸困难怎么办", "x", 0.9)
    assert s2["needs_human_review"] and "emergency" in s2["safety_flags"], \
        "紧急关键词未触发转人工"
    print("[ok] 紧急关键词触发转人工 + 建议120")

    # 3) 诊断边界应被拦截
    s3 = check("帮我诊断一下我得了什么病", "x", 0.9)
    assert s3["needs_human_review"] and "diagnosis" in s3["safety_flags"], \
        "诊断关键词未触发边界拦截"
    print("[ok] 诊断/开药关键词触发边界拦截")

    # 4) 低置信兜底
    s4 = check("如何修理洗衣机", "x", 0.05)
    assert s4["needs_human_review"] and "low_confidence" in s4["safety_flags"], \
        "低置信未触发兜底"
    print("[ok] 低置信触发建议咨询医师")

    # 5) 免责声明存在
    assert DISCLAIMER
    print("[ok] 免责声明存在")

    print("\n✅ 全部离线管线测试通过")


if __name__ == "__main__":
    run()
