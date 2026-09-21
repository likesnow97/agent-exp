from __future__ import annotations

from .base import BaseMemory, MemoryItem
from .semantic import SemanticMemory


class SkillMemory(BaseMemory):
    """Stores reusable procedures rather than raw dialogue."""

    name = "skill"

    def __init__(self) -> None:
        self.backend = SemanticMemory()

    def add(self, item: MemoryItem) -> None:
        if item.kind == "skill":
            self.backend.add(item)

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        return self.backend.retrieve(query, top_k=top_k)

    def reset(self) -> None:
        self.backend.reset()
