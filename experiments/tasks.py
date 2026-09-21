from __future__ import annotations

from dataclasses import dataclass

from memory import MemoryItem


@dataclass(frozen=True)
class ExperimentCase:
    name: str
    seed_memory: list[MemoryItem]
    query: str


CASES = [
    ExperimentCase(
        name="recent_preference",
        seed_memory=[
            MemoryItem("The user's preferred editor is Vim.", kind="fact"),
            MemoryItem("The user later switched to VS Code.", kind="fact"),
        ],
        query="Which editor does the user currently prefer?",
    ),
    ExperimentCase(
        name="semantic_fact",
        seed_memory=[
            MemoryItem("The project API runs on port 8765.", kind="fact"),
            MemoryItem("The database backup runs every Sunday.", kind="fact"),
            MemoryItem("The experiment server has one RTX 4090 GPU.", kind="fact"),
        ],
        query="Which port should I use to call the project API?",
    ),
    ExperimentCase(
        name="past_episode",
        seed_memory=[
            MemoryItem(
                "Episode: Installing package X failed because build isolation hid a required local dependency. "
                "Disabling build isolation fixed the installation.",
                kind="episode",
            ),
            MemoryItem(
                "Episode: A model server failed to start because port 8000 was already occupied.",
                kind="episode",
            ),
        ],
        query="I see the same package build failure again. What previous experience is relevant?",
    ),
    ExperimentCase(
        name="reusable_skill",
        seed_memory=[
            MemoryItem(
                "Skill: Restart vLLM service. Steps: stop the old process; verify the port is free; "
                "start vLLM with the configured model, host, port and API key; then call /v1/models to verify it.",
                kind="skill",
            ),
            MemoryItem(
                "Skill: Create a Python virtual environment. Steps: create environment; activate it; install dependencies.",
                kind="skill",
            ),
        ],
        query="How should I restart and verify the vLLM service?",
    ),
]
