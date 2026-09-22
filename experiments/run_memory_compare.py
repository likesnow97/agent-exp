from __future__ import annotations

# argparse：读取命令行参数，例如 --memory no_memory
import argparse

# json：把实验结果保存成 JSON 文件
import json

# Path：更方便地处理文件夹和文件路径
from pathlib import Path

# Callable：这里只用于类型标注，表示“一个可以被调用的对象”
from typing import Callable

# LLMClient：负责调用你自己部署的 OpenAI-compatible / vLLM API
# MemoryAgent：真正执行“检索记忆 -> 调用 LLM -> 写入记忆”的 Agent
from agent import LLMClient, MemoryAgent

# Settings：从 .env 中读取模型 API 地址、Key、模型名等配置
from config import Settings

# 导入不同类型的 Memory
from memory import (
    BaseMemory,
    EpisodicMemory,
    NoMemory,
    SemanticMemory,
    ShortTermMemory,
    SkillMemory,
)

# CASES：预先定义好的实验任务
# ExperimentCase：单个实验任务的数据结构
from experiments.tasks import CASES, ExperimentCase


# 定义一种类型：
# MemoryFactory 表示“调用后会创建一个 BaseMemory 对象的函数/类”
#
# 例如：
# NoMemory() -> NoMemory 对象
# SemanticMemory() -> SemanticMemory 对象
MemoryFactory = Callable[[], BaseMemory]


# 把命令行中的 memory 名称映射到具体 Memory 实现。
#
# 这样执行：
#   python -m experiments.run_memory_compare --memory semantic
#
# 就能通过 "semantic" 找到 SemanticMemory。
MEMORY_FACTORIES: dict[str, MemoryFactory] = {
    "no_memory": NoMemory,

    # lambda 的作用是创建 ShortTermMemory 时顺便指定最多保存 6 条记忆
    "short_term": lambda: ShortTermMemory(max_items=6),

    "semantic": SemanticMemory,
    "episodic": EpisodicMemory,
    "skill": SkillMemory,
}


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Compare how the same LLM agent behaves with different memory modules."
    )

    # --memory 用来指定运行哪一种 Memory。
    #
    # 例如：
    #   --memory no_memory
    #   --memory short_term
    #   --memory semantic
    #
    # 默认 all，表示全部运行。
    parser.add_argument(
        "--memory",
        choices=["all", *MEMORY_FACTORIES.keys()],
        default="all",
        help="Run one memory type or all memory types.",
    )

    # --case 用来指定运行哪一个实验任务。
    #
    # 例如：
    #   --case reusable_skill
    #
    # 默认 all，表示全部 case 都运行。
    parser.add_argument(
        "--case",
        choices=["all", *(case.name for case in CASES)],
        default="all",
        help="Run one experiment case or all cases.",
    )

    return parser.parse_args()


def selected_memories(name: str) -> list[tuple[str, MemoryFactory]]:
    """根据命令行参数决定本次实验要运行哪些 Memory。"""

    # 如果用户没有指定某一种，就返回全部 Memory
    if name == "all":
        return list(MEMORY_FACTORIES.items())

    # 否则只返回用户指定的那一种
    return [(name, MEMORY_FACTORIES[name])]


def selected_cases(name: str) -> list[ExperimentCase]:
    """根据命令行参数决定本次实验要运行哪些任务。"""

    # all：运行 tasks.py 中定义的所有 CASES
    if name == "all":
        return CASES

    # 否则只保留名称匹配的那个 case
    return [case for case in CASES if case.name == name]


def main() -> None:
    # ---------------------------------------------------------
    # 1. 初始化实验
    # ---------------------------------------------------------

    # 读取 --memory、--case 等命令行参数
    args = parse_args()

    # 从 .env 读取：
    # LLM_BASE_URL
    # LLM_API_KEY
    # LLM_MODEL
    # temperature 等配置
    settings = Settings.from_env()

    # 创建 LLM 客户端。
    # 后续 Agent 就通过这个对象向 vLLM API 发送请求。
    llm = LLMClient(settings)

    # 确保 results/ 文件夹存在
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 所有实验结果最终都会先收集到这个列表中，
    # 最后统一保存到 JSON 文件。
    all_results: list[dict] = []

    # ---------------------------------------------------------
    # 2. 遍历需要测试的 Memory
    # ---------------------------------------------------------

    # memory_name：
    #   "no_memory" / "short_term" / "semantic" ...
    #
    # factory：
    #   一个可以创建对应 Memory 对象的函数或类
    for memory_name, factory in selected_memories(args.memory):
        print(f"\n=== Memory: {memory_name} ===")

        # -----------------------------------------------------
        # 3. 对当前 Memory 运行所有指定的实验任务
        # -----------------------------------------------------

        for case in selected_cases(args.case):

            # 每个 case 都重新创建一个干净的 Memory。
            # 这样不同 case 之间不会互相污染记忆。
            memory = factory()

            # 创建 Agent。
            #
            # top_k=3 表示：
            # 每次最多从 Memory 中取 3 条相关记忆给 LLM。
            agent = MemoryAgent(
                llm=llm,
                memory=memory,
                top_k=3,
            )

            # -------------------------------------------------
            # 4. 先给 Agent 写入实验预设的 Memory
            # -------------------------------------------------

            # case.seed_memory 来自 experiments/tasks.py。
            #
            # 例如 semantic_fact 会提前提供：
            # "The project API runs on port 8765."
            #
            # seed_results 会记录：
            #   这条记忆是否被当前 Memory 接受。
            #
            # NoMemory：
            #   accepted=False
            #
            # SemanticMemory：
            #   accepted=True
            seed_results = agent.seed_memory(case.seed_memory)

            # -------------------------------------------------
            # 5. 正式让 Agent 处理问题
            # -------------------------------------------------

            # agent.run() 内部会做：
            #
            # query
            #   ↓
            # memory.retrieve()
            #   ↓
            # 把检索到的 Memory 放进 Prompt
            #   ↓
            # 调用 LLM
            #   ↓
            # 得到 answer
            #   ↓
            # 尝试把本次 interaction 再写入 Memory
            result = agent.run(case.query)

            # -------------------------------------------------
            # 6. 整理本次实验结果
            # -------------------------------------------------

            row = {
                # 使用了哪种 Memory
                "memory": memory_name,

                # 当前 case 名称
                "case": case.name,

                # 初始化时，每条 seed memory 是否成功写入
                "seed_writes": [
                    {
                        "kind": item.kind,
                        "text": item.text,
                        "accepted": accepted,
                    }
                    for item, accepted in seed_results
                ],

                # Agent 收到的问题
                "query": result.query,

                # Agent 在回答之前真正检索出来的 Memory
                "retrieved_memory": [
                    {
                        "kind": item.kind,
                        "text": item.text,
                    }
                    for item in result.retrieved_memory
                ],

                # LLM 最终回答
                "answer": result.answer,

                # 回答完成以后，
                # 当前 Memory 是否接受了这次新的 interaction
                "interaction_written": result.interaction_written,
            }

            # 把本次结果放进总结果列表
            all_results.append(row)

            # -------------------------------------------------
            # 7. 把关键过程打印到终端，方便直接观察
            # -------------------------------------------------

            print(f"\nCase: {case.name}")

            # 查看预设 Memory 有没有真正写进去
            print("Seed writes:")
            for item in row["seed_writes"]:
                print(
                    f"  - accepted={item['accepted']} "
                    f"[{item['kind']}] {item['text']}"
                )

            # 查看 Agent 回答当前问题时，
            # 到底从 Memory 中取出了哪些内容
            print("Retrieved:")

            if row["retrieved_memory"]:
                for item in row["retrieved_memory"]:
                    print(
                        f"  - [{item['kind']}] "
                        f"{item['text']}"
                    )
            else:
                print("  (none)")

            # LLM 最终回答
            print(f"Answer: {result.answer}")

            # 查看本轮 interaction 是否被继续存入 Memory
            print(
                "Interaction stored after answer: "
                f"{result.interaction_written}"
            )

    # ---------------------------------------------------------
    # 8. 保存所有实验结果
    # ---------------------------------------------------------

    output_path = results_dir / "memory_compare.json"

    # ensure_ascii=False：
    # 如果以后结果里包含中文，不会被转成 \uXXXX。
    #
    # indent=2：
    # 让 JSON 更容易人工阅读。
    output_path.write_text(
        json.dumps(
            all_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved results to {output_path}")


# 只有直接运行当前模块时才执行 main()。
#
# python -m experiments.run_memory_compare
#           ↓
#         main()
if __name__ == "__main__":
    main()
