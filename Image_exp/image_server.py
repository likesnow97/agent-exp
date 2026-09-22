from __future__ import annotations

import argparse
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start Qwen-Image-2.1 with vLLM-Omni."
    )

    # 与之前 memory_exp/server/llm_server.py 保持相同风格：
    # --model 可以填写 Hugging Face 模型名，也可以填写本地模型目录。
    parser.add_argument(
        "--model",
        required=True,
        help="Model name or local model path.",
    )

    parser.add_argument(
        "--served-model-name",
        default="qwen-image-2.1",
        help="Model name exposed by the API.",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8091,
    )

    parser.add_argument(
        "--api-key",
        default="change-me",
    )

    # 这两个参数默认不启用。
    # 需要时再传给 vLLM-Omni，避免第一轮实验引入额外配置。
    parser.add_argument(
        "--step-execution",
        action="store_true",
        help="Enable vLLM-Omni step execution mode.",
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Optional maximum number of concurrent sequences.",
    )

    args = parser.parse_args()

    # Qwen-Image-2.1 是 diffusion 图像生成模型，
    # 因此需要 --omni 启用 vLLM-Omni 模式。
    command = [
        "vllm",
        "serve",
        args.model,
        "--omni",
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
    ]

    if args.step_execution:
        command.append("--step-execution")

    if args.max_num_seqs is not None:
        command.extend(
            [
                "--max-num-seqs",
                str(args.max_num_seqs),
            ]
        )

    print("Starting:", " ".join(command))

    # check=True：
    # 如果 vllm serve 启动失败，Python 会抛出异常，
    # 而不是静默结束。
    subprocess.run(
        command,
        check=True,
    )


if __name__ == "__main__":
    main()
