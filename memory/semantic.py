from __future__ import annotations

import math
import re

# Counter 用来统计元素出现次数。
# 例如：
# Counter(["api", "port", "api"])
# -> {"api": 2, "port": 1}
from collections import Counter

from .base import BaseMemory, MemoryItem


# re.compile()：提前编译一个正则表达式。
#
# r"..." 前面的 r 表示 raw string（原始字符串），
# 这样反斜杠不会被 Python 字符串本身再次转义。
#
# 这个表达式会匹配：
# 1. 英文字母 / 数字 / 下划线组成的 token
# 2. 单个中文字符
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9_]+|[\u4e00-\u9fff]"
)


def _tokens(text: str) -> list[str]:
    # findall(text) 会返回所有匹配到的 token。
    #
    # 下面是“列表推导式”：
    #
    # [表达式 for 变量 in 可迭代对象]
    #
    # 等价于：
    #
    # result = []
    # for token in _TOKEN_RE.findall(text):
    #     result.append(token.lower())
    # return result
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text)
    ]


def _cosine(
    a: Counter[str],
    b: Counter[str],
) -> float:
    """计算两个词频向量的余弦相似度。"""

    # 空 Counter 在布尔判断中等价于 False。
    if not a or not b:
        return 0.0

    # sum(...) 把内部产生的数全部加起来。
    #
    # b.get(k, 0)：
    # 如果 b 中有 key=k，就返回对应值；
    # 否则返回默认值 0。
    dot = sum(
        a[k] * b.get(k, 0)
        for k in a
    )

    # 这里的：
    # (v * v for v in a.values())
    #
    # 是“生成器表达式”，它和列表推导式很像，
    # 但不会一次性创建整个列表。
    na = math.sqrt(
        sum(v * v for v in a.values())
    )
    nb = math.sqrt(
        sum(v * v for v in b.values())
    )

    # A if condition else B
    # 是 Python 的条件表达式（三元表达式）。
    #
    # 如果 na 和 nb 都不为 0，就计算相似度；
    # 否则返回 0。
    return dot / (na * nb) if na and nb else 0.0


class SemanticMemory(BaseMemory):
    """使用轻量词频相似度实现的 Semantic Memory。"""

    name = "semantic"

    def __init__(self) -> None:
        # [] 创建空列表。
        # list[MemoryItem] 是类型标注。
        self.items: list[MemoryItem] = []

    def add(self, item: MemoryItem) -> bool:
        self.items.append(item)
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[MemoryItem]:
        # 把 query 转成“token -> 出现次数”的词频表示。
        query_vec = Counter(_tokens(query))

        # scored 中每个元素都是三元组：
        #
        # (相似度, 原始下标, MemoryItem)
        #
        # enumerate(self.items)：
        # 同时得到“下标”和“元素”：
        #
        # 0, item0
        # 1, item1
        # ...
        scored = [
            (
                _cosine(
                    query_vec,
                    Counter(_tokens(item.text)),
                ),
                idx,
                item,
            )
            for idx, item in enumerate(self.items)
        ]

        # sort()：对当前列表原地排序。
        #
        # key=lambda x: (x[0], x[1])
        #
        # lambda 是匿名函数，等价于：
        #
        # def sort_key(x):
        #     return (x[0], x[1])
        #
        # x[0] 是相似度，x[1] 是原始下标。
        #
        # reverse=True 表示从大到小排序。
        scored.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True,
        )

        # scored[:top_k]：
        # 列表切片，只取前 top_k 个结果。
        #
        # for score, _, item in ...：
        # 这是“元组解包”。
        #
        # "_" 通常表示：
        # 这个位置有值，但这里不需要使用。
        #
        # 最后的 if score > 0：
        # 属于列表推导式里的过滤条件。
        return [
            item
            for score, _, item in scored[:top_k]
            if score > 0
        ]

    def reset(self) -> None:
        self.items.clear()
