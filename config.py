from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8765/v1")
        api_key = os.getenv("LLM_API_KEY", "change-me")
        model = os.getenv("LLM_MODEL", "")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "256"))
        timeout = float(os.getenv("LLM_TIMEOUT", "60"))

        if not model:
            raise ValueError(
                "LLM_MODEL is required. Copy .env.example to .env and set it."
            )

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
