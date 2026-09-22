# agent-exp

一个用于**直观理解不同 Memory 如何影响 LLM Agent 行为**的最小实验项目。

项目固定：
- 同一个 LLM
- 同一套任务
- 同一套 Agent 主流程
- 同一种 Prompt 结构

只替换不同的 Memory 模块，从而观察：

```text
记忆写入
  ↓
记忆检索
  ↓
检索结果注入 Prompt
  ↓
LLM 生成回答
  ↓
新的交互是否再次写入 Memory
```

---

## 1. 当前支持的 Memory

| Memory | 保存内容 | 检索方式 | 主要用途 |
|---|---|---|---|
| No Memory | 不保存 | 不检索 | 作为 baseline |
| Short-term Memory | 最近若干条记忆 | 最近 Top-k | 模拟工作记忆 / 对话上下文 |
| Semantic Memory | fact、interaction 等内容 | 本地相似度检索 | 模拟知识型长期记忆 |
| Episodic Memory | 仅保存 `episode` | 检索相似历史经历 | 利用过去任务经验 |
| Skill Memory | 仅保存 `skill` | 检索相似操作流程 | 复用已有技能 / procedure |

当前 Semantic Memory 使用的是**轻量本地文本相似度**，暂时不依赖 embedding 模型和向量数据库，方便先理解机制。

后续可以替换为：
- sentence-transformers
- FAISS
- Chroma
- Milvus
- 其他 embedding / vector database

而无需修改 Agent 主逻辑。

---

## 2. 项目结构

```text
agent-exp/
├── agent/
│   ├── agent.py              # Agent 主流程：retrieve → prompt → LLM → write
│   └── llm_client.py         # 调用 OpenAI-compatible API
│
├── memory/
│   ├── base.py               # Memory 统一接口
│   ├── no_memory.py
│   ├── short_term.py
│   ├── semantic.py
│   ├── episodic.py
│   └── skill_memory.py
│
├── experiments/
│   ├── tasks.py              # 实验任务
│   └── run_memory_compare.py # Memory 对比实验入口
│
├── server/
│   └── llm_server.py         # 可选：启动 vLLM
│
├── results/                  # 实验结果
├── config.py                 # API 配置读取
├── pyproject.toml            # uv 项目配置
├── .env.example
└── README.md
```

---

# 3. 环境要求

建议环境：

```text
Python >= 3.11
uv
```

如果你要在本机 / 服务器上直接运行 vLLM，还需要：

```text
Linux
NVIDIA GPU
CUDA
vLLM
```

如果 vLLM 已经在另一套环境或另一台机器运行，则本项目本身只需要能够访问它暴露的 OpenAI-compatible API。

---

# 4. 使用 uv 创建环境

克隆仓库：

```bash
git clone https://github.com/likesnow97/agent-exp.git
cd agent-exp
```

安装 Agent 实验所需依赖：

```bash
uv sync
```

这会安装：

```text
openai
python-dotenv
```

如果还希望由当前项目环境安装 vLLM：

```bash
uv sync --extra server
```

> 如果你的 vLLM 已经单独部署，不需要安装 `server` extra。

---

# 5. 配置 LLM API

首先复制配置文件：

```bash
cp .env.example .env
```

默认配置：

```env
LLM_BASE_URL=http://127.0.0.1:8765/v1
LLM_API_KEY=change-me
LLM_MODEL=agent-model
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=256
LLM_TIMEOUT=60
```

参数含义：

| 参数 | 含义 |
|---|---|
| `LLM_BASE_URL` | vLLM 暴露的 OpenAI-compatible API 地址 |
| `LLM_API_KEY` | API Key |
| `LLM_MODEL` | API 中暴露的模型名称 |
| `LLM_TEMPERATURE` | 生成随机性，实验默认设为 0 |
| `LLM_MAX_TOKENS` | 单次最大生成 token |
| `LLM_TIMEOUT` | API 请求超时时间 |

其中最需要注意的是：

```text
LLM_MODEL
```

必须与 vLLM 中：

```text
--served-model-name
```

保持一致。

---

# 6. 启动 vLLM

## 方法 A：直接使用 vLLM 命令

例如：

```bash
vllm serve YOUR_MODEL \
  --served-model-name agent-model \
  --host 127.0.0.1 \
  --port 8765 \
  --api-key change-me
```

其中：

```text
YOUR_MODEL
```

可以是 Hugging Face 模型名，也可以是服务器上的本地模型路径。

例如：

```bash
vllm serve /path/to/model \
  --served-model-name agent-model \
  --host 127.0.0.1 \
  --port 8765 \
  --api-key change-me
```

---

## 方法 B：使用项目中的 llm_server.py

```bash
uv run python server/llm_server.py \
  --model YOUR_MODEL \
  --served-model-name agent-model \
  --host 127.0.0.1 \
  --port 8765 \
  --api-key change-me
```

还可以控制：

```bash
--gpu-memory-utilization 0.90
--max-model-len 8192
```

完整示例：

```bash
uv run python server/llm_server.py \
  --model /path/to/model \
  --served-model-name agent-model \
  --port 8765 \
  --api-key change-me \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

---

# 7. 检查 vLLM 是否启动成功

可以请求：

```bash
curl http://127.0.0.1:8765/v1/models \
  -H "Authorization: Bearer change-me"
```

如果能够看到：

```text
agent-model
```

说明服务已经可用。

同时确认 `.env` 中：

```env
LLM_MODEL=agent-model
```

---

# 8. 运行全部 Memory 对比实验

```bash
uv run python -m experiments.run_memory_compare
```

程序会依次运行：

```text
No Memory
Short-term Memory
Semantic Memory
Episodic Memory
Skill Memory
```

并在相同任务上比较它们的行为。

---

# 9. 只运行一种 Memory

例如只测试 Episodic Memory：

```bash
uv run python -m experiments.run_memory_compare \
  --memory episodic
```

可选值：

```text
no_memory
short_term
semantic
episodic
skill
all
```

例如：

```bash
uv run python -m experiments.run_memory_compare --memory semantic
```

---

# 10. 当前实验任务

项目目前包含 4 个简单 case。

### 1. recent_preference

测试 Agent 是否能够根据记忆判断用户最新偏好。

```text
Vim
→ 后来切换到 VS Code
→ 问：现在更偏好哪个编辑器？
```

### 2. semantic_fact

测试 Agent 是否能够从若干事实中找到相关信息。

例如：

```text
项目 API 运行在 8765 端口
→ 问：调用项目 API 应该使用哪个端口？
```

### 3. past_episode

测试 Episodic Memory 是否能够调用过去的失败经验。

例如已有经验：

```text
某个 package 构建失败
原因：build isolation 隔离了本地依赖
解决：关闭 build isolation
```

然后再次询问相似错误。

### 4. reusable_skill

测试 Skill Memory 是否能够复用已有操作流程。

例如保存：

```text
Restart vLLM
1. 停止旧进程
2. 检查端口
3. 启动 vLLM
4. 请求 /v1/models 验证
```

然后询问如何重新启动 vLLM。

---

# 11. 只运行一个实验任务

例如：

```bash
uv run python -m experiments.run_memory_compare \
  --case reusable_skill
```

可选 case：

```text
recent_preference
semantic_fact
past_episode
reusable_skill
all
```

---

# 12. 同时指定 Memory 和任务

这是最适合调试的方法。

例如：

```bash
uv run python -m experiments.run_memory_compare \
  --memory episodic \
  --case past_episode
```

或者：

```bash
uv run python -m experiments.run_memory_compare \
  --memory skill \
  --case reusable_skill
```

这样可以非常直接地观察某一种 Memory 的行为。

---

# 13. 如何看实验输出

每个实验主要关注三部分：

### ① Seed writes

表示最初提供给 Agent 的 Memory 是否被当前 Memory 模块接受。

例如：

```text
accepted=True
[episode] ...
```

说明 Episodic Memory 成功保存了这条 episode。

如果：

```text
accepted=False
```

说明这种 Memory 不接受该类型的信息。

---

### ② Retrieved

表示面对当前任务时，Memory 实际检索出了什么。

例如：

```text
Retrieved:
  - [episode] Installing package X failed because ...
```

这一步最能体现：

```text
Agent 到底“想起了什么”
```

---

### ③ Answer

检索出的 Memory 会被加入 Prompt：

```text
Current task
+
Retrieved memory
+
LLM
→ Answer
```

因此可以直接比较：

```text
No Memory
vs
Short-term
vs
Semantic
vs
Episodic
vs
Skill
```

最终回答有什么区别。

---

# 14. 实验结果保存位置

所有结果会保存到：

```text
results/memory_compare.json
```

其中包含：

```text
memory
case
seed_writes
query
retrieved_memory
answer
interaction_written
```

因此后续可以继续做：

- success rate
- retrieval accuracy
- memory hit rate
- token cost
- latency
- 不同 memory 的效果对比

---

# 15. Agent 的核心 Memory 工作流

核心代码位于：

```text
agent/agent.py
```

整个过程可以简化为：

```text
当前任务 query
      ↓
memory.retrieve(query)
      ↓
retrieved memory
      ↓
注入 Prompt
      ↓
调用 vLLM API
      ↓
生成 answer
      ↓
memory.add(new interaction)
```

因此理解这个项目时，建议重点阅读：

```text
memory/base.py
      ↓
memory/*.py
      ↓
agent/agent.py
      ↓
experiments/run_memory_compare.py
```

---

# 16. Memory 统一接口

所有 Memory 都继承：

```text
BaseMemory
```

主要只有三个接口：

```python
add(item)
retrieve(query, top_k)
reset()
```

因此以后新增 Memory 时，只需要实现这三个接口。

例如可以继续添加：

```text
Vector Memory
Summary Memory
Hierarchical Memory
Graph Memory
Hybrid Memory
Reflection Memory
Long-term User Memory
Experience Replay Memory
```

而不需要修改 Agent 主流程。

---

# 17. 推荐的学习顺序

如果目标是理解 Agent Memory，建议按下面顺序运行：

```text
1. No Memory
2. Short-term Memory
3. Semantic Memory
4. Episodic Memory
5. Skill Memory
```

重点观察：

```text
Memory 保存了什么？
        ↓
为什么检索出这条内容？
        ↓
这条内容如何进入 Prompt？
        ↓
它是否改变了 Agent 的回答？
```

这个项目的目的不是追求复杂实现，而是先把 **Agent 如何写入、检索和使用 Memory** 这件事看清楚。
