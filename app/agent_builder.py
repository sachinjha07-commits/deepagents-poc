"""Assemble a deep agent from an :class:`~app.config.AgentConfig`.

This is the one place where every notebook feature is wired together:

* `create_deep_agent(...)`                     — notebook 1
* custom Tavily tool                           — notebook 1 / 4
* `TodoListMiddleware` (the `write_todos` plan) — planning pillar
* built-in filesystem tools (`ls`/`read_file`/`write_file`/`edit_file`)
* `memory=[AGENTS.md]`                         — notebook 2
* `skills=[...]`                               — notebook 2
* `CodeInterpreterMiddleware`                  — notebook 2
* `StateBackend` / `FilesystemBackend` / `StoreBackend` — notebook 3
* `subagents=[...]` + `response_format`        — notebook 4
* `checkpointer=MemorySaver()`                 — multi-turn conversation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from app.config import (
    DISK_AGENTS_MD,
    DISK_SKILLS_DIR,
    STORE_NAMESPACE,
    VIRTUAL_AGENTS_MD,
    VIRTUAL_SKILLS_DIR,
    AgentConfig,
)
from app.context import build_context_files
from app.tools import internet_search


class ResearchFindings(BaseModel):
    """Structured findings from a research task (notebook 4)."""

    summary: str = Field(description="Summary of findings")
    confidence: float = Field(description="Confidence score from 0 to 1")
    sources: list[str] = Field(description="List of source URLs")


@dataclass
class AgentBundle:
    """A built agent plus everything the UI needs to talk to it."""

    graph: Any
    config: AgentConfig
    seed_files: dict[str, Any] = field(default_factory=dict)
    store: BaseStore | None = None
    backend: Any = None
    features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tool_names: list[str] = field(default_factory=list)

    @property
    def needs_file_seeding(self) -> bool:
        """StateBackend files live in one thread's state, so re-seed each turn."""
        return self.config.backend == "state" and bool(self.seed_files)


def _build_backend(cfg: AgentConfig, store: BaseStore | None):
    """Notebook 3: the same agent code, three different storage layers."""
    from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend

    if cfg.backend == "filesystem":
        root = Path(cfg.filesystem_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return FilesystemBackend(root_dir=str(root), virtual_mode=True)
    if cfg.backend == "store":
        return StoreBackend(store=store, namespace=lambda _rt: STORE_NAMESPACE)
    return StateBackend()


def _memory_paths(cfg: AgentConfig) -> list[str]:
    if not cfg.memory:
        return []
    return [DISK_AGENTS_MD if cfg.backend == "filesystem" else VIRTUAL_AGENTS_MD]


def _skill_sources(cfg: AgentConfig) -> list[str] | None:
    if not cfg.skills or not cfg.skill_names:
        return None
    return [DISK_SKILLS_DIR if cfg.backend == "filesystem" else VIRTUAL_SKILLS_DIR]


def _seed_store(store: BaseStore, files: dict[str, Any]) -> None:
    """StoreBackend keys *are* the virtual paths, so seed them verbatim."""
    for path, file_data in files.items():
        store.put(STORE_NAMESPACE, path, dict(file_data))


def _build_subagents(cfg: AgentConfig) -> list[dict[str, Any]]:
    """Notebook 4: a specialised researcher with its own context window."""
    if not cfg.subagents:
        return []

    research_tools = [internet_search] if cfg.web_search else []
    researcher: dict[str, Any] = {
        "name": "research-agent",
        "description": (
            "Delegate in-depth research here. Give it a self-contained question; "
            "it works in its own context window and returns only its final answer."
        ),
        "system_prompt": (
            "You are a great researcher. Search the web, cross-check claims across "
            "sources, and return a dense, well-organised summary with source URLs. "
            "Write bulky raw material to files rather than into your answer."
        ),
        "tools": research_tools,
    }
    if cfg.subagent_model and cfg.subagent_model != cfg.model:
        researcher["model"] = cfg.subagent_model
    if cfg.structured_research:
        researcher["response_format"] = ResearchFindings

    critic: dict[str, Any] = {
        "name": "critique-agent",
        "description": (
            "Delegate review of a draft or a finding here. Returns concrete, "
            "actionable criticism — no rewriting."
        ),
        "system_prompt": (
            "You are a rigorous reviewer. Read the draft the supervisor gives you "
            "(use read_file when it points at a file), then list concrete problems: "
            "unsupported claims, missing context, structural issues. Be specific "
            "and brief. Do not rewrite the draft."
        ),
    }
    return [researcher, critic]


def _build_middleware(cfg: AgentConfig, notes: list[str]) -> list[Any]:
    middleware: list[Any] = []

    if cfg.planning:
        # `write_todos` is langchain's TodoListMiddleware; deepagents 0.7.x no
        # longer attaches it by default, so the planning pillar is opt-in here.
        from langchain.agents.middleware import TodoListMiddleware

        middleware.append(TodoListMiddleware())

    if cfg.code_interpreter:
        try:
            from langchain_quickjs import CodeInterpreterMiddleware

            middleware.append(CodeInterpreterMiddleware())
        except Exception as exc:  # pragma: no cover - optional dependency
            notes.append(f"Code interpreter unavailable ({exc}).")

    return middleware


def build_agent(
    cfg: AgentConfig,
    *,
    checkpointer: BaseCheckpointSaver,
    store: BaseStore | None = None,
) -> AgentBundle:
    """Build the deep agent described by ``cfg``."""
    from deepagents import create_deep_agent

    notes: list[str] = []
    features: list[str] = []

    backend = _build_backend(cfg, store)
    features.append(f"backend={cfg.backend}")

    tools = [internet_search] if cfg.web_search else []
    tool_names = ["internet_search"] if cfg.web_search else []

    middleware = _build_middleware(cfg, notes)
    if cfg.planning:
        features.append("planning (write_todos)")
        tool_names.append("write_todos")
    tool_names += ["ls", "read_file", "write_file", "edit_file"]
    if cfg.code_interpreter and any(
        type(m).__name__ == "CodeInterpreterMiddleware" for m in middleware
    ):
        features.append("code interpreter (JS sandbox)")
        tool_names.append("eval")

    memory = _memory_paths(cfg)
    if memory:
        features.append("memory (AGENTS.md)")
    skills = _skill_sources(cfg)
    if skills:
        features.append(f"skills ({len(cfg.skill_names)})")

    subagents = _build_subagents(cfg)
    if subagents:
        features.append(f"subagents ({', '.join(s['name'] for s in subagents)})")
        tool_names.append("task")
    if cfg.structured_research:
        features.append("structured subagent output")

    # Seed AGENTS.md + skill files into the virtual filesystem for the backends
    # that have no disk of their own.
    seed_files: dict[str, Any] = {}
    if cfg.backend != "filesystem" and (cfg.memory or cfg.skills):
        seed_files = build_context_files(
            include_memory=cfg.memory,
            skill_names=cfg.skill_names if cfg.skills else (),
            agents_md_virtual_path=VIRTUAL_AGENTS_MD,
            skills_virtual_root=VIRTUAL_SKILLS_DIR,
        )
    if cfg.backend == "store" and store is not None and seed_files:
        # The store outlives every thread, so seeding once is enough.
        _seed_store(store, seed_files)
        seed_files = {}
        notes.append("AGENTS.md and skills were seeded into the store (cross-thread).")

    graph = create_deep_agent(
        model=cfg.model,
        tools=tools,
        system_prompt=cfg.system_prompt,
        middleware=middleware,
        subagents=subagents or None,
        skills=skills,
        memory=memory or None,
        backend=backend,
        checkpointer=checkpointer,
        store=store if cfg.backend == "store" else None,
    )

    return AgentBundle(
        graph=graph,
        config=cfg,
        seed_files=seed_files,
        store=store,
        backend=backend,
        features=features,
        notes=notes,
        tool_names=tool_names,
    )
