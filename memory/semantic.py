from __future__ import annotations

import math
import re
from collections import Counter

from .base import BaseMemory, MemoryItem


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0) for k in a)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class SemanticMemory(BaseMemory):
    """Simple local vector-like retrieval without external embedding models."""

    name = "semantic"

    def __init__(self) -> None:
        self.items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> bool:
        self.items.append(item)
        return True

    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        query_vec = Counter(_tokens(query))
        scored = [
            (_cosine(query_vec, Counter(_tokens(item.text))), idx, item)
            for idx, item in enumerate(self.items)
        ]
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [item for score, _, item in scored[:top_k] if score > 0]

    def reset(self) -> None:
        self.items.clear()
