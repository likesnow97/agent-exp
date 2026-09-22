from __future__ import annotations

# deque（双端队列）适合维护一个“固定长度的最近记录窗口”。
# 当达到 maxlen 后继续 append，新元素进入时最旧元素会自动被删除。
from collections import deque

from .base import BaseMemory, MemoryItem


# 继承 BaseMemory：
# ShortTermMemory 必须实现 add / retrieve / reset。
class ShortTermMemory(BaseMemory):
    name = "short_term"

    def __init__(self, max_items: int = 6) -> None:
        # deque[MemoryItem] 是类型标注：
        # 表示这个 deque 中保存的是 MemoryItem。
        #
        # maxlen=max_items：
        # 队列最多保存 max_items 条。
        self.items: deque[MemoryItem] = deque(
            maxlen=max_items
        )

    def add(self, item: MemoryItem) -> bool:
        # append()：把元素加到队尾。
        self.items.append(item)

        # True 表示这条 Memory 被成功保存。
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[MemoryItem]:
        # 当前 Short-term Memory 不考虑 query 的语义，
        # 只返回最近的 top_k 条。

        # deque 转成普通 list，方便使用切片。
        items = list(self.items)

        # Python 切片：
        # items[-top_k:]
        #
        # -top_k 表示“从倒数第 top_k 个开始”。
        #
        # 例如：
        # items = [A, B, C, D]
        # top_k = 2
        # items[-2:] -> [C, D]
        return items[-top_k:]

    def reset(self) -> None:
        # clear()：原地清空 deque。
        self.items.clear()
