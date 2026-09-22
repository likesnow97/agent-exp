# Image_exp

用于测试 **Qwen-Image-2.1 + vLLM-Omni** 的文本生成图片实验。

## 文件

```text
Image_exp/
├── image_server.py      # 启动常驻 vLLM-Omni 服务
├── image_client.py      # 循环输入 prompt 并生成图片
├── generate_image.py    # 之前的 Diffusers 直接推理版本，保留作备用
├── pyproject.toml
└── outputs/             # 自动生成，不提交到 Git
```

## 1. 创建客户端环境

进入目录：

```bash
cd Image_exp
uv sync
```

当前 `pyproject.toml` 只管理客户端依赖。

> Qwen-Image-2.1 是新模型，vLLM-Omni 需要使用支持该模型的版本，并与 vLLM 的 major/minor 版本匹配，因此没有在这里硬编码 vLLM-Omni 版本。

## 2. 准备 vLLM-Omni

当前官方 vLLM-Omni 要求 Python 3.12。

如果你的环境已经可以执行：

```bash
vllm serve --help
```

并且帮助中包含：

```text
--omni
```

则可以直接进入下一步。

## 3. 启动 Qwen-Image-2.1

例如模型已经下载到本地：

```bash
uv run python image_server.py \
  --model /path/to/Qwen-Image-2.1 \
  --served-model-name qwen-image-2.1 \
  --host 127.0.0.1 \
  --port 8091 \
  --api-key change-me
```

也可以直接使用 Hugging Face 模型名：

```bash
uv run python image_server.py \
  --model Qwen/Qwen-Image-2.1
```

`image_server.py` 实际执行：

```text
vllm serve <model> --omni ...
```

模型只加载一次，之后客户端可以连续发送生成请求。

### 可选：step execution

```bash
uv run python image_server.py \
  --model /path/to/Qwen-Image-2.1 \
  --step-execution \
  --max-num-seqs 8
```

## 4. 检查服务

```bash
curl http://127.0.0.1:8091/health
```

查看模型：

```bash
curl http://127.0.0.1:8091/v1/models \
  -H "Authorization: Bearer change-me"
```

## 5. 循环生成图片

新开一个终端：

```bash
cd Image_exp
uv run python image_client.py
```

然后持续输入：

```text
Prompt> 一只坐在实验室里的橘猫
Saved: Image_exp/outputs/001_..._seed42.png

Prompt> 东京雨夜的街道
Saved: Image_exp/outputs/002_..._seed43.png

Prompt> exit
```

模型服务不会重新加载。

每成功生成一张图片，默认 seed 自动增加：

```text
42
43
44
...
```

## 6. 修改生成参数

例如：

```bash
uv run python image_client.py \
  --base-url http://127.0.0.1:8091 \
  --api-key change-me \
  --model qwen-image-2.1 \
  --size 1024x1024 \
  --steps 40 \
  --seed 100
```

主要参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--base-url` | `http://127.0.0.1:8091` | 服务地址 |
| `--model` | `qwen-image-2.1` | API 暴露的模型名 |
| `--size` | `1024x1024` | 图片尺寸 |
| `--steps` | `40` | diffusion inference steps |
| `--seed` | `42` | 第一张图片的随机种子 |
| `--output-dir` | `Image_exp/outputs` | 输出目录 |
| `--timeout` | `600` | 单次请求超时 |

## 调用链

```text
image_server.py
    ↓
vllm serve Qwen-Image-2.1 --omni
    ↓
模型常驻 GPU
    ↓
image_client.py
    ↓
POST /v1/images/generations
    ↓
base64 PNG
    ↓
outputs/*.png
    ↓
继续输入下一个 prompt
```
