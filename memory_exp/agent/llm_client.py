from __future__ import annotations

from openai import OpenAI

from config import Settings


class LLMClient:
    """负责调用 OpenAI-compatible API，包括 vLLM。"""

    def __init__(self, settings: Settings) -> None:
        # 把配置对象保存到当前实例中。
        self.settings = settings

        # OpenAI(...) 创建一个 API 客户端对象。
        #
        # 这里虽然使用 openai SDK，
        # 但 base_url 指向的是你自己部署的 vLLM 服务，
        # 因此真正处理请求的是你的 vLLM。
        self.client = OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout,
        )

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        # 参数列表中的单独一个 "*" 有特殊作用：
        #
        # 它表示 "*" 后面的参数必须通过“关键字参数”传入。
        #
        # 正确：
        # chat(
        #     system_prompt="...",
        #     user_prompt="..."
        # )
        #
        # 不允许：
        # chat("...", "...")

        # client.chat.completions.create(...)
        # 对应 OpenAI-compatible API 的 chat completions 接口。
        response = self.client.chat.completions.create(
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,

            # messages 是一个 list，
            # 每个元素都是一个 dict。
            #
            # dict 使用：
            # {"key": value}
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        # response.choices[0]
        # 表示取返回结果列表中的第一个候选答案。
        #
        # .message.content
        # 继续读取这个答案中的文本内容。
        #
        # A or ""
        # 表示：
        # 如果 content 是 None 或空值，就返回空字符串，
        # 从而保证这个函数最终返回 str。
        return response.choices[0].message.content or ""
