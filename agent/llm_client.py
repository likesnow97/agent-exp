from __future__ import annotations

from openai import OpenAI

from config import Settings


class LLMClient:
    """Thin client for any OpenAI-compatible API, including vLLM."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
        )

    def chat(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""
