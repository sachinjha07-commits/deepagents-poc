# Deep Agent Studio

A Streamlit chatbot that puts every feature from the notebooks in `deepagents/`
behind one conversational UI.

```bash
uv run streamlit run streamlit_app.py
```

Then open http://localhost:8501.

`.env` needs `OPENAI_API_KEY` (and/or `GROQ_API_KEY`) plus `TAVILY_API_KEY` for
web search. The app tells you which variables are missing and lets you switch off
the features that need them.

## What is wired up, and where it came from

| Feature | Notebook | In the app |
|---|---|---|
| `create_deep_agent`, custom Tavily tool, system prompt | `1-basicdeepagent` | Sidebar → Model, Capabilities → *Web search* |
| Planning with `write_todos` | planning pillar | Sidebar → *Planning*; live plan in the **Plan** tab |
| Virtual filesystem (`ls`/`read_file`/`write_file`/`edit_file`) | 1 & 3 | **Files** tab — browse and download whatever the agent wrote |
| `AGENTS.md` as always-on memory | `2-contextengineering` | Sidebar → *Load AGENTS.md as memory*; content in the **Context** tab |
| Agent Skills with progressive disclosure | `2-contextengineering` | Sidebar → *Agent Skills* + which skills to expose |
| Code interpreter (QuickJS `eval`) | `2-contextengineering` | Sidebar → *Code interpreter* |
| `StateBackend` / `FilesystemBackend` / `StoreBackend` | `3-backends` | Sidebar → Backend (switchable at runtime) |
| Subagents via the `task` tool, per-subagent model override | `4-subagent` | Sidebar → Subagents; nested calls shown indented in the trace |
| Structured subagent output (`response_format`) | `4-subagent` | Sidebar → *Structured findings* |
| Multi-turn memory (`MemorySaver` + `thread_id`) | — | Sidebar → Conversation / **New thread** |

## Layout

* **Left** — the chat. Each answer streams token by token, with a collapsible
  trace of every tool call, tool result and subagent delegation behind it.
* **Right** — three tabs: **Plan** (the live todo list), **Files** (the agent's
  virtual filesystem, whichever backend is active) and **Context** (active
  configuration, the loaded `AGENTS.md`, the available skills).

## Backends in practice

| Backend | Files live in | Try this |
|---|---|---|
| `state` | this thread's LangGraph state | Ask for a file, then hit **New thread** — it's gone |
| `filesystem` | real disk under `root_dir` (defaults to `deepagents/`) | Ask for a file, then look for it in your editor |
| `store` | an `InMemoryStore` shared by every thread this session | Write a file, hit **New thread**, ask the agent to read it back |

## Code map

```
streamlit_app.py       UI: chat, streaming, trace, Plan/Files/Context tabs
app/config.py          paths, model catalog, AgentConfig (one switch per feature)
app/agent_builder.py   assembles create_deep_agent from a config
app/context.py         loads AGENTS.md + skills, seeds them into a backend
app/runner.py          flattens the LangGraph stream into UI events; state/file readers
app/tools.py           the Tavily internet_search tool
```

Notes:

* `deepagents` 0.7.x no longer attaches `TodoListMiddleware` by default, so the
  planning pillar is added explicitly in `app/agent_builder.py`.
* With `StateBackend`, `AGENTS.md` and the skill files are re-seeded on every
  turn (state is per-thread); with `StoreBackend` they are seeded once; with
  `FilesystemBackend` they are read straight from disk.
* The `filesystem` backend gives the agent real write access under `root_dir` —
  point it at a scratch directory if that makes you nervous.
