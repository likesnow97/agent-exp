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
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8765/v1")
        api_key = os.getenv("LLM_API_KEY", "change-me")
        model = os.getenv("LLM_MODEL", "")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

        if not model:
            raise ValueError("LLM_MODEL is required. Copy .env.example to .env and set it.")

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
