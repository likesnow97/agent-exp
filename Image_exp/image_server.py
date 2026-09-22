from __future__ import annotations

import os
import subprocess


# ============================================================
# 配置区
# ============================================================
#
# 以后通常只需要修改这里，不需要在命令行传一大堆参数。

# Qwen-Image-2.1 本地模型路径
MODEL_PATH = "../../modlib/model/Qwen/Qwen-Image-2.1/"

# 对外暴露给 API 的模型名称
SERVED_MODEL_NAME = "qwen-image-2.1"

# 指定使用哪张物理 GPU。
#
# 例如：
#   "0"   -> 只使用 GPU 0
#   "1"   -> 只使用 GPU 1
#   "0,1" -> 让进程看到 GPU 0 和 GPU 1
#
# 注意：
# CUDA_VISIBLE_DEVICES="1" 后，
# vLLM 内部看到的这张卡会变成 cuda:0。
GPU_DEVICES = "1"

# API 服务地址
HOST = "127.0.0.1"
PORT = 8091

# API Key。
# 如果只在本机测试，可以先使用这个简单值。
API_KEY = "change-me"

# vLLM-Omni 可选配置。
# 第一轮实验先保持关闭，避免引入额外变量。
STEP_EXECUTION = False
MAX_NUM_SEQS: int | None = None


def build_command() -> list[str]:
    """
    构造最终执行的 vLLM 命令。

    返回值类似：

    [
        "vllm",
        "serve",
        ".../Qwen-Image-2.1/",
        "--omni",
        ...
    ]
    """

    command = [
        "vllm",
        "serve",
        MODEL_PATH,
        "--omni",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--api-key",
        API_KEY,
    ]

    if STEP_EXECUTION:
        command.append("--step-execution")

    if MAX_NUM_SEQS is not None:
        command.extend(
            [
                "--max-num-seqs",
                str(MAX_NUM_SEQS),
            ]
        )

    return command


def main() -> None:
    # os.environ 是当前 Python 进程的环境变量。
    #
    # copy()：
    # 复制一份环境变量给 vLLM 子进程，
    # 避免直接修改当前 Python 进程本身的环境。
    env = os.environ.copy()

    # 这一行等价于在终端执行：
    #
    # CUDA_VISIBLE_DEVICES=1 vllm serve ...
    #
    # vLLM 因此只能看到 GPU_DEVICES 指定的显卡。
    env["CUDA_VISIBLE_DEVICES"] = GPU_DEVICES

    command = build_command()

    print("=" * 60)
    print("Qwen-Image-2.1 vLLM-Omni server")
    print("=" * 60)
    print(f"Model:       {MODEL_PATH}")
    print(f"GPU:         {GPU_DEVICES}")
    print(f"Model name:  {SERVED_MODEL_NAME}")
    print(f"API:         http://{HOST}:{PORT}")
    print("=" * 60)

    print("\nStarting:")
    print(" ".join(command))
    print()

    # subprocess.run()：
    # 从当前 Python 程序启动另一个进程。
    #
    # env=env：
    # 把上面设置好的 CUDA_VISIBLE_DEVICES
    # 一并传给 vLLM。
    #
    # check=True：
    # 如果 vLLM 启动失败，会立即抛出异常，
    # 方便我们看到真实错误。
    subprocess.run(
        command,
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
