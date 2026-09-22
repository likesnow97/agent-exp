from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline


def parse_args() -> argparse.Namespace:
    """读取命令行参数。"""

    parser = argparse.ArgumentParser(
        description="使用 Qwen-Image-2.1 根据文本生成图片。"
    )

    # 可以填写 Hugging Face 模型名，也可以填写服务器上的本地模型路径。
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2.1",
        help="Hugging Face 模型名或本地模型目录。",
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="用于生成图片的文本提示词。",
    )

    parser.add_argument(
        "--output",
        default="Image_exp/output.png",
        help="生成图片的保存路径。",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="生成图片宽度。",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="生成图片高度。",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="扩散推理步数。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，用于复现实验结果。",
    )

    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="启用 CPU offload，降低显存占用，但速度会变慢。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("当前未检测到 CUDA GPU。")

    print(f"Loading model: {args.model}")

    # QwenImage21Pipeline 是 Diffusers 为 Qwen-Image-2.1 提供的官方 Pipeline。
    # bfloat16 可以显著降低显存占用。
    pipe = QwenImage21Pipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    if args.cpu_offload:
        # 将暂时不用的模型组件卸载到 CPU，以减少 GPU 显存占用。
        pipe.enable_model_cpu_offload()
    else:
        # 默认把整个 Pipeline 放到 GPU。
        pipe = pipe.to("cuda")

    # 固定随机种子后，相同配置下更容易复现实验结果。
    generator = torch.Generator(
        device="cuda"
    ).manual_seed(args.seed)

    print("Generating image...")

    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        generator=generator,
    )

    # Pipeline 返回的 images 是图片列表，这里保存第一张。
    image = result.images[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(output_path)

    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
