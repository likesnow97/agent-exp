from __future__ import annotations

import json
from pathlib import Path

from agent import LLMClient, MemoryAgent
from config import Settings
from memory import (
    EpisodicMemory,
    NoMemory,
    SemanticMemory,
    ShortTermMemory,
    SkillMemory,
)
from experiments.tasks import CASES


MEMORY_FACTORIES = {
    "no_memory": NoMemory,
    "short_term": lambda: ShortTermMemory(max_items=6),
    "semantic": SemanticMemory,
    "episodic": EpisodicMemory,
    "skill": SkillMemory,
}


def main() -> None:
    settings = Settings.from_env()
    llm = LLMClient(settings)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_results: list[dict] = []

    for memory_name, factory in MEMORY_FACTORIES.items():
        print(f"\n=== Memory: {memory_name} ===")

        for case in CASES:
            memory = factory()
            agent = MemoryAgent(llm=llm, memory=memory, top_k=3)
            agent.seed_memory(case.seed_memory)

            result = agent.run(case.query)

            row = {
                "memory": memory_name,
                "case": case.name,
                "query": result.query,
                "retrieved_memory": [
                    {"kind": item.kind, "text": item.text}
                    for item in result.retrieved_memory
                ],
                "answer": result.answer,
            }
            all_results.append(row)

            print(f"\nCase: {case.name}")
            print("Retrieved:")
            if row["retrieved_memory"]:
                for item in row["retrieved_memory"]:
                    print(f"  - [{item['kind']}] {item['text']}")
            else:
                print("  (none)")
            print(f"Answer: {result.answer}")

    output_path = results_dir / "memory_compare.json"
    output_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
