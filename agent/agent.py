from __future__ import annotations

from dataclasses import dataclass

from memory import BaseMemory, MemoryItem
from .llm_client import LLMClient


SYSTEM_PROMPT = """You are a concise task-solving agent.
Use retrieved memory only when it is relevant.
If memory conflicts with the current user request, follow the current request.
"""


@dataclass
class AgentResult:
    query: str
    retrieved_memory: list[MemoryItem]
    answer: str
    interaction_written: bool


class MemoryAgent:
    def __init__(self, llm: LLMClient, memory: BaseMemory, top_k: int = 3) -> None:
        self.llm = llm
        self.memory = memory
        self.top_k = top_k

    def seed_memory(self, items: list[MemoryItem]) -> list[tuple[MemoryItem, bool]]:
        return [(item, self.memory.add(item)) for item in items]

    @staticmethod
    def _format_memory(items: list[MemoryItem]) -> str:
        if not items:
            return "(no relevant memory)"
        return "\n".join(f"- [{item.kind}] {item.text}" for item in items)

    def run(self, query: str) -> AgentResult:
        # 1) Read: retrieve memory relevant to the current query.
        retrieved = self.memory.retrieve(query, top_k=self.top_k)

        # 2) Use: inject retrieved memory into the LLM prompt.
        prompt = f"""Current task:
{query}

Retrieved memory:
{self._format_memory(retrieved)}

Answer the current task concisely.
"""

        # 3) Think/answer: call the same LLM for every memory condition.
        answer = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        # 4) Write: propose the completed interaction as new memory.
        # Each memory implementation decides whether this type should be stored.
        written = self.memory.add(
            MemoryItem(
                text=f"Task: {query}\nAnswer: {answer}",
                kind="interaction",
            )
        )

        return AgentResult(
            query=query,
            retrieved_memory=retrieved,
            answer=answer,
            interaction_written=written,
        )
