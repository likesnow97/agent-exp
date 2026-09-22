from __future__ import annotations

import os

from dataclasses import dataclass
from dotenv import load_dotenv


# 读取项目根目录中的 .env，
# 并把里面的变量加载进当前进程环境。
load_dotenv()


# frozen=True 表示：
# Settings 对象创建后，其字段不应该再被修改。
#
# 例如：
# settings.model = "xxx"
# 会报错。
@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str

    # “字段: 类型 = 默认值”
    # 表示这些参数如果没有显式提供，
    # 就使用右侧默认值。
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "Settings":
        # @classmethod 表示：
        # 这个方法接收的第一个参数不是 self，而是类本身 cls。
        #
        # 因此最后可以写：
        # return cls(...)
        #
        # 相当于创建一个 Settings(...) 对象。

        # os.getenv("变量名", "默认值")
        # 用于读取环境变量。
        #
        # 如果变量不存在，就使用第二个参数作为默认值。
        base_url = os.getenv(
            "LLM_BASE_URL",
            "http://127.0.0.1:8765/v1",
        )

        api_key = os.getenv(
            "LLM_API_KEY",
            "change-me",
        )

        model = os.getenv(
            "LLM_MODEL",
            "",
        )

        # os.getenv() 返回的是字符串，
        # 所以需要用 float() / int() 转换成数字。
        temperature = float(
            os.getenv("LLM_TEMPERATURE", "0.0")
        )

        max_tokens = int(
            os.getenv("LLM_MAX_TOKENS", "256")
        )

        timeout = float(
            os.getenv("LLM_TIMEOUT", "60")
        )

        # 空字符串在 if 判断中等价于 False。
        if not model:
            # raise：主动抛出异常，并停止当前流程。
            raise ValueError(
                "LLM_MODEL is required. "
                "Copy .env.example to .env and set it."
            )

        # cls(...) 等价于 Settings(...)，
        # 但使用 cls 可以让 classmethod 更容易被子类复用。
        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
