"""DeepSeek's OpenAI-compatible API; LangGraph owns all orchestration."""

import os

from langchain_openai import ChatOpenAI


def deepseek_model():
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("真实模型模式需要设置 DEEPSEEK_API_KEY 环境变量")
    return ChatOpenAI(
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
        timeout=60,
        max_retries=2,
        max_tokens=1500,
    )
