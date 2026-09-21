from __future__ import annotations

from collections import deque

from .base import BaseMemory, MemoryItem


class ShortTermMemory(BaseMemory):
    name = "short_term"

    def __init__(self, max_items: int = 6) -> None:
        self.items: deque[MemoryItem] = deque(maxlen=max_items)

    def add(self, item: MemoryItem) -> None:
        self.items.append(item)

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        items = list(self.items)
        return items[-top_k:]

    def reset(self) -> None:
        self.items.clear()
