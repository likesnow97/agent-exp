from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryItem:
    text: str
    kind: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    name = "base"

    @abstractmethod
    def add(self, item: MemoryItem) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
