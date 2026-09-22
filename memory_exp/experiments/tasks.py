from __future__ import annotations

from dataclasses import dataclass

from memory import MemoryItem


# frozen=True：
# 创建后不允许修改字段。
#
# 这里把每个实验 case 当作固定配置，
# 可以避免实验运行过程中被意外改掉。
@dataclass(frozen=True)
class ExperimentCase:
    # case 名称
    name: str

    # 实验开始前预先写入 Agent 的 Memory。
    #
    # list[MemoryItem]：
    # 表示这是一个由多个 MemoryItem 组成的列表。
    seed_memory: list[MemoryItem]

    # 本次实验真正要问 Agent 的问题
    query: str


# CASES 是一个普通 Python list。
# 其中每个元素都是一个 ExperimentCase 对象。
CASES = [
    ExperimentCase(
        name="recent_preference",

        # [] 创建 list。
        # list 中放两个 MemoryItem 对象。
        seed_memory=[
            MemoryItem(
                "The user's preferred editor is Vim.",
                kind="fact",
            ),
            MemoryItem(
                "The user later switched to VS Code.",
                kind="fact",
            ),
        ],

        query="Which editor does the user currently prefer?",
    ),

    ExperimentCase(
        name="semantic_fact",
        seed_memory=[
            MemoryItem(
                "The project API runs on port 8765.",
                kind="fact",
            ),
            MemoryItem(
                "The database backup runs every Sunday.",
                kind="fact",
            ),
            MemoryItem(
                "The experiment server has one RTX 4090 GPU.",
                kind="fact",
            ),
        ],
        query="Which port should I use to call the project API?",
    ),

    ExperimentCase(
        name="past_episode",
        seed_memory=[
            MemoryItem(
                # Python 会自动拼接相邻的字符串字面量。
                #
                # 下面虽然写成两行：
                # "AAA "
                # "BBB"
                #
                # 实际得到的是一个完整字符串：
                # "AAA BBB"
                "Episode: Installing package X failed because "
                "build isolation hid a required local dependency. "
                "Disabling build isolation fixed the installation.",
                kind="episode",
            ),
            MemoryItem(
                "Episode: A model server failed to start because "
                "port 8000 was already occupied.",
                kind="episode",
            ),
        ],
        query=(
            # 圆括号中的相邻字符串也会自动拼接。
            "I see the same package build failure again. "
            "What previous experience is relevant?"
        ),
    ),

    ExperimentCase(
        name="reusable_skill",
        seed_memory=[
            MemoryItem(
                "Skill: Restart vLLM service. Steps: "
                "stop the old process; verify the port is free; "
                "start vLLM with the configured model, host, port "
                "and API key; then call /v1/models to verify it.",
                kind="skill",
            ),
            MemoryItem(
                "Skill: Create a Python virtual environment. "
                "Steps: create environment; activate it; "
                "install dependencies.",
                kind="skill",
            ),
        ],
        query="How should I restart and verify the vLLM service?",
    ),
]
