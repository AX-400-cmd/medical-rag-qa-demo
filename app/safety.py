# -*- coding: utf-8 -*-
"""
医疗安全边界（Medical Guardrails）：这是本项目的核心设计点，直接对应 JD 里
“医疗场景严格边界 / 人工接管 / 可追溯”的要求。

三层防护：
  1) 紧急危重识别：命中即强制转人工 + 建议拨打 120，绝不给自行处理建议。
  2) 诊断/用药边界：涉及“帮我诊断/开药/我得了什么病”一律不下结论，转医师。
  3) 低置信兜底：检索置信度低于阈值时，不替用户拍板，建议专业核实。
  4) 隐私脱敏：对身份证号、手机号、显式姓名做脱敏，避免泄露。
无论哪条命中，回答都会附带统一免责声明，保证“可追溯、可接管”。
"""
import re

DISCLAIMER = "本内容仅作健康科普参考，不能替代执业医师的诊断与治疗建议。"

# 紧急/危重信号（命中即转人工）
EMERGENCY_KW = [
    "胸痛", "呼吸困难", "大出血", "昏迷", "中风", "抽搐", "休克", "中毒",
    "自杀", "溺水", "窒息", "心梗", "猝死", "大出血不止", "意识丧失",
]

# 诊断/开药边界（命中即不自行结论）
DIAGNOSIS_KW = [
    "帮我诊断", "帮我确诊", "开药", "开处方", "我得了", "我是不是得了",
    "我是什么病", "该吃什么药", "给我药方", "帮我开",
]


def redact(text: str) -> str:
    """对常见隐私做脱敏。"""
    text = re.sub(r"\d{17}[\dXx]", "[身份证号已脱敏]", text)
    text = re.sub(r"1[3-9]\d{9}", "[手机号已脱敏]", text)
    text = re.sub(r"(姓名[：:]\s*)[^\s,，。；]{1,4}", r"\1[姓名已脱敏]", text)
    return text


def check(query: str, answer: str, confidence: float, threshold: float = None):
    """返回安全 verdict：needs_human_review / safety_flags / refusal_reason / answer(已脱敏)。"""
    from .config import settings
    if threshold is None:
        threshold = settings.confidence_threshold

    flags = []
    needs = False
    reason = ""

    if any(k in query for k in EMERGENCY_KW):
        needs = True
        flags.append("emergency")
        reason = "检测到紧急/危重情形，已转人工并建议立即就医（急救电话 120）。"

    if any(k in query for k in DIAGNOSIS_KW):
        needs = True
        flags.append("diagnosis")
        reason = (reason + " ") if reason else ""
        reason += "涉及诊断/用药，超出健康科普边界，建议咨询执业医师。"

    if confidence < threshold:
        needs = True
        flags.append("low_confidence")
        if not reason:
            reason = "检索置信度偏低，建议咨询专业医师核实，避免误判。"

    # 安全答案策略：命中“紧急/诊断”时，绝不吐出无关检索片段，直接给出安全拒答；
    # 仅低置信时，不给确定结论、建议核实；其余才放行模型/检索生成的回答（先脱敏）。
    if "emergency" in flags or "diagnosis" in flags:
        safe_answer = redact(reason)
    elif "low_confidence" in flags:
        safe_answer = redact("检索到的相关内容有限，未给出确定结论；" + reason)
    else:
        safe_answer = redact(answer)
    return {
        "needs_human_review": needs,
        "safety_flags": flags,
        "refusal_reason": reason,
        "answer": safe_answer,
    }
