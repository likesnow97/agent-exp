# agent-exp

A minimal project for understanding how different memory types change an LLM agent's behavior.

## Memory types

| Memory | What it keeps | Retrieval |
|---|---|---|
| No Memory | nothing | none |
| Short-term | recent items | latest Top-k |
| Semantic | facts/interactions | local similarity retrieval |
| Episodic | only `episode` items | similar past experiences |
| Skill | only `skill` items | similar reusable procedures |

The experiment keeps the **LLM, prompt style and tasks fixed** and changes only the memory module.

## Project structure

```text
agent-exp/
├── agent/
│   ├── agent.py
│   └── llm_client.py
├── memory/
│   ├── base.py
│   ├── no_memory.py
│   ├── short_term.py
│   ├── semantic.py
│   ├── episodic.py
│   └── skill_memory.py
├── experiments/
│   ├── tasks.py
│   └── run_memory_compare.py
├── server/
│   └── llm_server.py
├── results/
├── config.py
├── pyproject.toml
└── .env.example
```

## 1. Create the uv environment

```bash
uv sync
```

If you also want this project to install vLLM:

```bash
uv sync --extra server
```

## 2. Configure the API

```bash
cp .env.example .env
```

Example:

```env
LLM_BASE_URL=http://127.0.0.1:8765/v1
LLM_API_KEY=change-me
LLM_MODEL=agent-model
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=256
LLM_TIMEOUT=60
```

`LLM_MODEL` must match the model name exposed by your vLLM server.

## 3. Start vLLM

Directly:

```bash
vllm serve YOUR_MODEL \
  --served-model-name agent-model \
  --host 127.0.0.1 \
  --port 8765 \
  --api-key change-me
```

Or use the included launcher:

```bash
uv run python server/llm_server.py \
  --model YOUR_MODEL \
  --port 8765 \
  --api-key change-me
```

## 4. Run experiments

Run every memory type and every case:

```bash
uv run python -m experiments.run_memory_compare
```

Run one memory type:

```bash
uv run python -m experiments.run_memory_compare --memory episodic
```

Run one task:

```bash
uv run python -m experiments.run_memory_compare --case reusable_skill
```

Combine both:

```bash
uv run python -m experiments.run_memory_compare \
  --memory skill \
  --case reusable_skill
```

Results are saved to:

```text
results/memory_compare.json
```

## What to observe

Each run prints three key stages:

```text
seed memory
    ↓
Memory.add()       # what this memory accepts
    ↓
Memory.retrieve()  # what it recalls for the current task
    ↓
prompt + LLM API
    ↓
answer
    ↓
Memory.add()       # whether the new interaction is stored
```

This makes the difference between memory types visible without LangChain or another agent framework.

## Note on Semantic Memory

The current implementation uses a lightweight local lexical similarity function instead of an embedding model. This keeps the experiment dependency-light and makes retrieval easy to inspect. It can later be replaced by sentence-transformers, FAISS, Chroma or another vector backend without changing the Agent interface.
