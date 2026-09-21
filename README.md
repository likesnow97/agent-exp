# agent-exp

A minimal project for understanding how different memory types change an LLM agent's behavior.

## Memory types

- No Memory
- Short-term Memory
- Semantic / Vector-like Memory
- Episodic Memory
- Skill Memory

The experiment keeps the LLM and tasks fixed and changes only the memory module.

## Project structure

```text
agent-exp/
├── agent/          # Agent loop and LLM client
├── memory/         # Pluggable memory implementations
├── experiments/    # Tasks and comparison runner
├── server/         # Optional vLLM launcher
├── results/        # Experiment outputs
├── config.py
├── pyproject.toml
└── .env.example
```

## 1. Create the uv environment

```bash
uv sync
```

If you also want this project to install vLLM itself:

```bash
uv sync --extra server
```

## 2. Configure the LLM API

```bash
cp .env.example .env
```

```env
LLM_BASE_URL=http://127.0.0.1:8765/v1
LLM_API_KEY=change-me
LLM_MODEL=your-served-model-name
LLM_TEMPERATURE=0.2
```

## 3. Start vLLM

```bash
vllm serve YOUR_MODEL \
  --host 127.0.0.1 \
  --port 8765 \
  --api-key change-me
```

Or:

```bash
uv run python server/llm_server.py \
  --model YOUR_MODEL \
  --port 8765 \
  --api-key change-me
```

## 4. Run the comparison

```bash
uv run python -m experiments.run_memory_compare
```

Results are saved to:

```text
results/memory_compare.json
```

## Core flow

```text
Task
  ↓
Memory.retrieve()
  ↓
Retrieved memory
  ↓
Prompt construction
  ↓
vLLM / OpenAI-compatible API
  ↓
Answer
  ↓
Memory.add()
```

The code intentionally avoids LangChain so the memory mechanism remains easy to inspect.
