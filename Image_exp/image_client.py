from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively generate images with a "
            "Qwen-Image-2.1 vLLM-Omni server."
        )
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8091",
        help=(
            "Server address. Both http://host:port "
            "and http://host:port/v1 are accepted."
        ),
    )

    parser.add_argument(
        "--api-key",
        default="change-me",
        help="API key configured when starting the server.",
    )

    parser.add_argument(
        "--model",
        default="qwen-image-2.1",
        help="Served model name.",
    )

    parser.add_argument(
        "--size",
        default="1024x1024",
        help='Image size, for example "1024x1024".',
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of diffusion inference steps.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Initial random seed.",
    )

    parser.add_argument(
        "--output-dir",
        default="Image_exp/outputs",
        help="Directory used to save generated images.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds.",
    )

    return parser.parse_args()


def build_endpoint(base_url: str) -> str:
    """Build /v1/images/generations from either base URL style."""

    base_url = base_url.rstrip("/")

    if base_url.endswith("/v1"):
        return base_url + "/images/generations"

    return base_url + "/v1/images/generations"


def generate_image(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    steps: int,
    seed: int,
    timeout: float,
) -> bytes:
    """Call the image generation API and return PNG bytes."""

    endpoint = build_endpoint(base_url)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "num_inference_steps": steps,
        "seed": seed,
        "n": 1,
        "response_format": "b64_json",
    }

    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text}"
        )

    data = response.json()
    image_b64 = data["data"][0]["b64_json"]

    return base64.b64decode(image_b64)


def build_output_path(
    output_dir: Path,
    index: int,
    seed: int,
) -> Path:
    """Create a unique file name for every generated image."""

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return output_dir / (
        f"{index:03d}_{timestamp}_seed{seed}.png"
    )


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Qwen-Image-2.1 interactive client")
    print("输入 prompt 生成图片。")
    print("输入 exit / quit / q 退出。")
    print()

    index = 1

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExit.")
            break

        if not prompt:
            continue

        if prompt.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("Exit.")
            break

        # 每成功生成一次，seed 自动 +1。
        # 因此连续输入同一 prompt 时也能得到不同结果。
        current_seed = args.seed + index - 1

        print(
            f"Generating: size={args.size}, "
            f"steps={args.steps}, "
            f"seed={current_seed}"
        )

        try:
            image_bytes = generate_image(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                size=args.size,
                steps=args.steps,
                seed=current_seed,
                timeout=args.timeout,
            )

            output_path = build_output_path(
                output_dir,
                index=index,
                seed=current_seed,
            )

            output_path.write_bytes(
                image_bytes
            )

            print(
                f"Saved: {output_path}\n"
            )

            index += 1

        except (
            requests.RequestException,
            RuntimeError,
            KeyError,
            ValueError,
            TypeError,
        ) as exc:
            print(
                f"Generation failed: {exc}\n"
            )


if __name__ == "__main__":
    main()
