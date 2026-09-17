"""DeepSeek's OpenAI-compatible API; LangGraph owns all orchestration."""

import os

from langchain_openai import ChatOpenAI


def deepseek_model():
    # 不读取或打印密钥；仅从进程环境获取，以避免把凭据写进 checkpoint 或仓库。
    # Never read from files or print the key; use process env to keep it out of checkpoints and Git.
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise ValueError("真实模型模式需要设置 DEEPSEEK_API_KEY 环境变量")
    # DeepSeek 兼容 OpenAI Chat Completions 协议；LangGraph 仍负责工作流编排。
    # DeepSeek is OpenAI Chat Completions compatible; LangGraph still owns orchestration.
    return ChatOpenAI(
        api_key=key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        # 研究演示追求可复现性；temperature=0 降低同输入的随机差异。
        # The demo favors reproducibility; temperature=0 reduces variance for identical input.
        timeout=60,
        max_retries=2,
        max_tokens=1500,
    )
