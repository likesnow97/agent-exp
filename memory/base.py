from __future__ import annotations

# ABC = Abstract Base Class，抽象基类。
# abstractmethod 用来规定：
# 子类必须实现某个方法。
from abc import ABC, abstractmethod

# dataclass：减少“数据类”的样板代码。
# field：用于给 dataclass 字段配置更复杂的默认值。
from dataclasses import dataclass, field

# Any 表示“这里允许任意类型”。
from typing import Any


@dataclass
class MemoryItem:
    # 每条 Memory 最核心的文本内容
    text: str

    # 默认值为 "generic"。
    # 创建对象时如果不传 kind，就自动使用这个值。
    kind: str = "generic"

    # dict[str, Any]：
    # key 必须是 str，value 可以是任意类型。
    #
    # 为什么不用：
    #   metadata: dict = {}
    #
    # 因为 {} 是可变对象，多个实例可能错误地共享同一个 dict。
    #
    # field(default_factory=dict)
    # 表示每创建一个 MemoryItem，都重新调用 dict() 创建一个新字典。
    metadata: dict[str, Any] = field(default_factory=dict)


# BaseMemory(ABC)：
# BaseMemory 继承 ABC，因此可以定义抽象接口。
#
# 它的作用不是直接拿来运行，
# 而是要求 NoMemory、ShortTermMemory 等子类遵守统一接口。
class BaseMemory(ABC):
    name = "base"

    # @abstractmethod：
    # 子类必须实现 add()。
    # 如果某个子类没有实现，通常不能直接实例化。
    @abstractmethod
    def add(self, item: MemoryItem) -> bool:
        """保存一条记忆，并返回是否成功接受这条记忆。"""

        # raise 表示主动抛出异常。
        #
        # 这里理论上不会真正执行，
        # 因为具体子类应该覆盖（override）这个方法。
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[MemoryItem]:
        """根据 query 检索最多 top_k 条记忆。"""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """清空当前 Memory。"""
        raise NotImplementedError
