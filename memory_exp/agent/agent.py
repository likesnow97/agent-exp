from __future__ import annotations

# dataclass 用来快速定义“主要用于存数据”的类。
# 它会自动帮我们生成 __init__、__repr__ 等方法。
from dataclasses import dataclass

from memory import BaseMemory, MemoryItem

# 前面的 "." 表示相对导入：
# 从当前 agent 包里的 llm_client.py 导入 LLMClient。
from .llm_client import LLMClient


SYSTEM_PROMPT = """You are a concise task-solving agent.
Use retrieved memory only when it is relevant.
If memory conflicts with the current user request, follow the current request.
"""


# @dataclass 是 Python 的“装饰器”。
# 可以理解为：在类定义完成后，再自动帮这个类补充一些常用功能。
#
# 有了它，我们可以直接写：
#   result = AgentResult(query="hello", ...)
# 而不用手写 __init__。
@dataclass
class AgentResult:
    # “变量名: 类型”属于类型标注。
    # Python 运行时通常不会强制检查，但 IDE / 类型检查器可以利用它。
    query: str

    # list[MemoryItem] 表示：
    # 这是一个列表，里面每个元素都应该是 MemoryItem。
    retrieved_memory: list[MemoryItem]

    answer: str
    interaction_written: bool


class MemoryAgent:
    def __init__(
        self,
        llm: LLMClient,
        memory: BaseMemory,
        top_k: int = 3,
    ) -> None:
        # self 表示“当前这个对象本身”。
        # 下面是在对象里保存三个属性。
        self.llm = llm
        self.memory = memory
        self.top_k = top_k

    def seed_memory(
        self,
        items: list[MemoryItem],
    ) -> list[tuple[MemoryItem, bool]]:
        # 返回类型：
        # list[tuple[MemoryItem, bool]]
        #
        # 可以拆开理解：
        # tuple[MemoryItem, bool]
        #   ↓
        # 一个二元组，例如：
        # (某条记忆, True)
        #
        # list[tuple[...]]
        #   ↓
        # 很多个这样的二元组组成的列表。
        #
        # 下面这句属于“列表推导式”：
        #
        # [(表达式) for item in items]
        #
        # 等价于：
        #
        # result = []
        # for item in items:
        #     accepted = self.memory.add(item)
        #     result.append((item, accepted))
        # return result
        return [
            (item, self.memory.add(item))
            for item in items
        ]

    @staticmethod
    def _format_memory(items: list[MemoryItem]) -> str:
        # @staticmethod 表示这个函数虽然放在类里，
        # 但它不需要访问 self，也不依赖某个具体 Agent 对象。

        if not items:
            return "(no relevant memory)"

        # join() 用指定字符串连接多个字符串。
        #
        # "\n".join(["A", "B"])
        # 得到：
        # A
        # B
        #
        # f"...{变量}..." 是 f-string，用来把变量直接插入字符串。
        return "\n".join(
            f"- [{item.kind}] {item.text}"
            for item in items
        )

    def run(self, query: str) -> AgentResult:
        # -----------------------------------------------------
        # 1. 从 Memory 中读取与当前 query 相关的内容
        # -----------------------------------------------------
        retrieved = self.memory.retrieve(
            query,
            top_k=self.top_k,
        )

        # -----------------------------------------------------
        # 2. 把检索到的 Memory 放进 Prompt
        # -----------------------------------------------------

        # f"""..."""：
        # 三引号允许写多行字符串；
        # 前面的 f 表示其中可以使用 {变量/表达式}。
        prompt = f"""Current task:
{query}

Retrieved memory:
{self._format_memory(retrieved)}

Answer the current task concisely.
"""

        # -----------------------------------------------------
        # 3. 调用 LLM
        # -----------------------------------------------------
        answer = self.llm.chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        # -----------------------------------------------------
        # 4. 把本轮交互尝试写回 Memory
        # -----------------------------------------------------

        # MemoryItem(...) 是创建一个 MemoryItem 对象。
        #
        # 这里用的是“关键字参数”：
        # text=...
        # kind=...
        #
        # 比 MemoryItem(x, y) 更容易看出每个参数的含义。
        written = self.memory.add(
            MemoryItem(
                text=f"Task: {query}\nAnswer: {answer}",
                kind="interaction",
            )
        )

        # 返回一个 AgentResult 对象，而不是普通 dict，
        # 后续可以通过：
        #   result.answer
        #   result.retrieved_memory
        # 来访问字段。
        return AgentResult(
            query=query,
            retrieved_memory=retrieved,
            answer=answer,
            interaction_written=written,
        )
