# -*- coding: utf-8 -*-
"""极简 LLM 客户端：仅用标准库 urllib 调用兼容 OpenAI 的 /chat/completions。

不引入 openai SDK，保证离线无 key 时整个项目零额外依赖也能跑。
配置方式（环境变量）：
    LLM_BASE_URL=https://你的服务/v1/chat/completions
    LLM_API_KEY=sk-xxxx
    LLM_MODEL=你的模型名
"""
import json
import urllib.request

from .config import settings


def chat(messages, temperature: float = 0.2, json_mode: bool = False) -> str:
    """messages: [{"role":"system"/"user"/"assistant", "content": "..."}]
    返回模型文本。json_mode=True 时要求模型返回 JSON 对象。"""
    url = settings.llm_base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]
