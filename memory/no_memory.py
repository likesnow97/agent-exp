from __future__ import annotations

from .base import BaseMemory, MemoryItem


class NoMemory(BaseMemory):
    name = "no_memory"

    def add(self, item: MemoryItem) -> None:
        return None

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        return []

    def reset(self) -> None:
        return None
