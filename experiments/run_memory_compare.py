from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from agent import LLMClient, MemoryAgent
from config import Settings
from memory import (
    BaseMemory,
    EpisodicMemory,
    NoMemory,
    SemanticMemory,
    ShortTermMemory,
    SkillMemory,
)
from experiments.tasks import CASES, ExperimentCase


MemoryFactory = Callable[[], BaseMemory]

MEMORY_FACTORIES: dict[str, MemoryFactory] = {
    "no_memory": NoMemory,
    "short_term": lambda: ShortTermMemory(max_items=6),
    "semantic": SemanticMemory,
    "episodic": EpisodicMemory,
    "skill": SkillMemory,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare how the same LLM agent behaves with different memory modules."
    )
    parser.add_argument(
        "--memory",
        choices=["all", *MEMORY_FACTORIES.keys()],
        default="all",
        help="Run one memory type or all memory types.",
    )
    parser.add_argument(
        "--case",
        choices=["all", *(case.name for case in CASES)],
        default="all",
        help="Run one experiment case or all cases.",
    )
    return parser.parse_args()


def selected_memories(name: str) -> list[tuple[str, MemoryFactory]]:
    if name == "all":
        return list(MEMORY_FACTORIES.items())
    return [(name, MEMORY_FACTORIES[name])]


def selected_cases(name: str) -> list[ExperimentCase]:
    if name == "all":
        return CASES
    return [case for case in CASES if case.name == name]


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    llm = LLMClient(settings)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    all_results: list[dict] = []

    for memory_name, factory in selected_memories(args.memory):
        print(f"\n=== Memory: {memory_name} ===")

        for case in selected_cases(args.case):
            memory = factory()
            agent = MemoryAgent(llm=llm, memory=memory, top_k=3)

            seed_results = agent.seed_memory(case.seed_memory)
            result = agent.run(case.query)

            row = {
                "memory": memory_name,
                "case": case.name,
                "seed_writes": [
                    {
                        "kind": item.kind,
                        "text": item.text,
                        "accepted": accepted,
                    }
                    for item, accepted in seed_results
                ],
                "query": result.query,
                "retrieved_memory": [
                    {"kind": item.kind, "text": item.text}
                    for item in result.retrieved_memory
                ],
                "answer": result.answer,
                "interaction_written": result.interaction_written,
            }
            all_results.append(row)

            print(f"\nCase: {case.name}")
            print("Seed writes:")
            for item in row["seed_writes"]:
                print(
                    f"  - accepted={item['accepted']} "
                    f"[{item['kind']}] {item['text']}"
                )

            print("Retrieved:")
            if row["retrieved_memory"]:
                for item in row["retrieved_memory"]:
                    print(f"  - [{item['kind']}] {item['text']}")
            else:
                print("  (none)")

            print(f"Answer: {result.answer}")
            print(f"Interaction stored after answer: {result.interaction_written}")

    output_path = results_dir / "memory_compare.json"
    output_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
