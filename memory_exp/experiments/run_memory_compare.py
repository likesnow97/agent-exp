from __future__ import annotations

# argparse：读取命令行参数，例如 --memory no_memory
import argparse

# json：把实验结果保存成 JSON 文件
import json

# Path：更方便地处理文件夹和文件路径
from pathlib import Path

# Callable：类型标注，表示“可以被调用的对象”
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


# Callable[[], BaseMemory] 可以拆开理解：
#
# Callable[参数列表, 返回类型]
#
# [] 表示“不需要参数”
# BaseMemory 表示“返回一个 BaseMemory 对象”
#
# 因此 MemoryFactory 表示：
# “一个不需要参数、调用后能创建 Memory 的东西”。
#
# Python 中“类本身”也是可调用对象，例如：
# NoMemory()
MemoryFactory = Callable[[], BaseMemory]


# dict[str, MemoryFactory]：
# key 是 str，value 是 MemoryFactory。
MEMORY_FACTORIES: dict[str, MemoryFactory] = {
    "no_memory": NoMemory,

    # lambda 是匿名函数。
    #
    # 下面等价于：
    #
    # def create_short_term():
    #     return ShortTermMemory(max_items=6)
    #
    # 使用 lambda 是为了让所有 value 都保持“调用后创建 Memory”的形式。
    "short_term": lambda: ShortTermMemory(max_items=6),

    "semantic": SemanticMemory,
    "episodic": EpisodicMemory,
    "skill": SkillMemory,
}


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Compare how the same LLM agent behaves "
            "with different memory modules."
        )
    )

    # MEMORY_FACTORIES.keys()
    # 得到所有 dict key。
    #
    # 前面的 * 是“可迭代对象解包”：
    #
    # ["all", *["a", "b"]]
    # 等价于：
    # ["all", "a", "b"]
    parser.add_argument(
        "--memory",
        choices=["all", *MEMORY_FACTORIES.keys()],
        default="all",
        help="Run one memory type or all memory types.",
    )

    # *(case.name for case in CASES)
    #
    # (case.name for case in CASES)
    # 是生成器表达式；
    # 前面的 * 再把它展开到 list 中。
    parser.add_argument(
        "--case",
        choices=[
            "all",
            *(case.name for case in CASES),
        ],
        default="all",
        help="Run one experiment case or all cases.",
    )

    return parser.parse_args()


def selected_memories(
    name: str,
) -> list[tuple[str, MemoryFactory]]:
    """根据命令行参数决定运行哪些 Memory。"""

    # 返回类型：
    # list[tuple[str, MemoryFactory]]
    #
    # 即：
    # [
    #   ("no_memory", NoMemory),
    #   ("semantic", SemanticMemory),
    #   ...
    # ]

    if name == "all":
        # dict.items() 返回 (key, value)。
        # list(...) 把它转换成普通 list。
        return list(MEMORY_FACTORIES.items())

    # MEMORY_FACTORIES[name]：
    # 使用 key 从 dict 中取出对应的 value。
    return [
        (name, MEMORY_FACTORIES[name])
    ]


def selected_cases(name: str) -> list[ExperimentCase]:
    """根据命令行参数决定运行哪些实验任务。"""

    if name == "all":
        return CASES

    # 这是“带过滤条件的列表推导式”：
    #
    # [表达式 for 元素 in 列表 if 条件]
    #
    # 等价于：
    #
    # result = []
    # for case in CASES:
    #     if case.name == name:
    #         result.append(case)
    # return result
    return [
        case
        for case in CASES
        if case.name == name
    ]


def main() -> None:
    # 1. 读取命令行参数
    args = parse_args()

    # 2. 从 .env 创建配置对象
    settings = Settings.from_env()

    # 3. 创建调用 vLLM 的客户端
    llm = LLMClient(settings)

    # Path("results") 创建一个 Path 对象，
    # 不是立即创建文件夹。
    results_dir = Path("results")

    # exist_ok=True：
    # 如果 results 已经存在，不报错。
    results_dir.mkdir(exist_ok=True)

    # list[dict]：
    # 这是一个列表，其中每个元素都是 dict。
    all_results: list[dict] = []

    # selected_memories(...) 中每个元素都是二元组：
    # (memory_name, factory)
    #
    # for memory_name, factory in ...
    # 属于“元组解包”。
    for memory_name, factory in selected_memories(
        args.memory
    ):
        print(f"\n=== Memory: {memory_name} ===")

        for case in selected_cases(args.case):
            # factory 是可调用对象。
            #
            # 可能是：
            # NoMemory
            # SemanticMemory
            # 或 lambda
            #
            # 统一使用 factory() 创建 Memory 实例。
            memory = factory()

            agent = MemoryAgent(
                llm=llm,
                memory=memory,
                top_k=3,
            )

            # 先写入实验预设记忆。
            #
            # 返回值类似：
            # [
            #   (MemoryItem(...), True),
            #   (MemoryItem(...), False),
            # ]
            seed_results = agent.seed_memory(
                case.seed_memory
            )

            # 正式执行一次 Agent。
            result = agent.run(case.query)

            # dict 使用：
            # {
            #     "key": value,
            # }
            row = {
                "memory": memory_name,
                "case": case.name,

                # 这里是列表推导式。
                #
                # for item, accepted in seed_results
                # 同样属于元组解包：
                # 每次把二元组拆成两个变量。
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
                    {
                        "kind": item.kind,
                        "text": item.text,
                    }
                    for item in result.retrieved_memory
                ],

                "answer": result.answer,
                "interaction_written": (
                    result.interaction_written
                ),
            }

            all_results.append(row)

            print(f"\nCase: {case.name}")
            print("Seed writes:")

            # row["seed_writes"]：
            # 通过 key 从 dict 中取 value。
            for item in row["seed_writes"]:
                # item 本身也是 dict，
                # 所以继续用 item["accepted"] 等方式访问。
                print(
                    f"  - accepted={item['accepted']} "
                    f"[{item['kind']}] {item['text']}"
                )

            print("Retrieved:")

            # 非空 list 在 if 中为 True，
            # 空 list 为 False。
            if row["retrieved_memory"]:
                for item in row["retrieved_memory"]:
                    print(
                        f"  - [{item['kind']}] "
                        f"{item['text']}"
                    )
            else:
                print("  (none)")

            print(f"Answer: {result.answer}")

            print(
                "Interaction stored after answer: "
                f"{result.interaction_written}"
            )

    # Path 对象支持 "/" 运算符拼接路径。
    #
    # Path("results") / "memory_compare.json"
    # 得到：
    # results/memory_compare.json
    output_path = (
        results_dir / "memory_compare.json"
    )

    # json.dumps(...)：
    # Python 对象 -> JSON 字符串。
    #
    # ensure_ascii=False：
    # 中文保持原样。
    #
    # indent=2：
    # 使用 2 个空格缩进，方便阅读。
    output_path.write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved results to {output_path}")


# Python 文件被直接运行时：
# __name__ == "__main__"
#
# 如果只是被其他文件 import，
# 这里就不会执行。
if __name__ == "__main__":
    main()
