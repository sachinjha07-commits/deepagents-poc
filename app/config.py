"""Paths, model catalog and the AgentConfig that drives the whole app.

Every switch here maps to something demonstrated in the notebooks:

| Notebook                    | Config field(s)                                  |
|-----------------------------|--------------------------------------------------|
| 1-basicdeepagent            | model, system_prompt, web_search                 |
| 2-contextengineering        | memory (AGENTS.md), skills, code_interpreter     |
| 3-backends                  | backend, filesystem_root                         |
| 4-subagent                  | subagents, subagent_model, structured_research   |
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEPAGENTS_DIR = PROJECT_ROOT / "deepagents"
PROJECTS_DIR = DEEPAGENTS_DIR / "projects"
SKILLS_DIR = DEEPAGENTS_DIR / "skills"
AGENTS_MD_PATH = PROJECTS_DIR / "AGENTS.md"
ENV_PATH = PROJECT_ROOT / ".env"

# Virtual paths used inside the agent's filesystem (StateBackend / StoreBackend).
VIRTUAL_AGENTS_MD = "/projects/AGENTS.md"
VIRTUAL_SKILLS_DIR = "/skills/"

# FilesystemBackend resolves paths relative to root_dir, so the same files are
# reachable without a leading slash.
DISK_AGENTS_MD = "projects/AGENTS.md"
DISK_SKILLS_DIR = "skills/"

STORE_NAMESPACE = ("memories",)

# --- Model catalog -----------------------------------------------------------
# The notebooks use "openai:gpt-5.4" / "openai:gpt-5.5" for the supervisor and a
# Groq model for a subagent override. Anything init_chat_model understands works;
# the sidebar also accepts a free-text "provider:model" string.
MODEL_CHOICES: list[str] = [
    "openai:gpt-5.4",
    "openai:gpt-5.5",
    "openai:gpt-5.4-mini",
    "openai:gpt-5-mini",
    "openai:gpt-4.1-mini",
    "groq:qwen/qwen3-32b",
    "groq:llama-3.3-70b-versatile",
]
DEFAULT_MODEL = "openai:gpt-5.4-mini"
DEFAULT_SUBAGENT_MODEL = "openai:gpt-5.4-mini"

BACKEND_CHOICES = ["state", "filesystem", "store"]
BACKEND_LABELS = {
    "state": "StateBackend — files live in this thread's LangGraph state",
    "filesystem": "FilesystemBackend — files are real files on disk",
    "store": "StoreBackend — files persist across every thread in this session",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a deep research assistant built with the deepagents library.\n"
    "Plan before you act, offload bulky material to files, delegate deep dives to "
    "subagents, and cite your sources.\n"
    "Answer conversationally — the user is chatting with you in a UI that shows "
    "your todos, your files and your tool calls alongside the transcript."
)

REQUIRED_KEYS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}


@dataclass(frozen=True)
class AgentConfig:
    """A complete, hashable description of the agent to build."""

    model: str = DEFAULT_MODEL
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Notebook 3: where the virtual filesystem physically lives.
    backend: str = "state"
    filesystem_root: str = str(DEEPAGENTS_DIR)

    # Notebook 1 + 2: capabilities bolted onto the agent.
    planning: bool = True
    web_search: bool = True
    code_interpreter: bool = False
    memory: bool = True
    skills: bool = True

    # Notebook 4: delegation.
    subagents: bool = True
    subagent_model: str = DEFAULT_SUBAGENT_MODEL
    structured_research: bool = False

    # Long-conversation hygiene.
    recursion_limit: int = 100

    # Which skills to expose (names of directories under deepagents/skills).
    skill_names: tuple[str, ...] = field(default_factory=tuple)

    def signature(self) -> str:
        """Stable key: when this changes the agent is rebuilt."""
        return json.dumps(asdict(self), sort_keys=True, default=str)

    @property
    def provider(self) -> str:
        return self.model.split(":", 1)[0] if ":" in self.model else "openai"

    def missing_api_keys(self) -> list[str]:
        """API keys the current selection needs but the environment lacks."""
        needed: set[str] = set()
        for model in (self.model, self.subagent_model if self.subagents else ""):
            if not model:
                continue
            provider = model.split(":", 1)[0]
            key = REQUIRED_KEYS.get(provider)
            if key:
                needed.add(key)
        if self.web_search:
            needed.add("TAVILY_API_KEY")
        return sorted(k for k in needed if not os.getenv(k))
