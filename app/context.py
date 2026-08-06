"""Context engineering helpers (notebook 2).

Two kinds of context are loaded from disk and handed to the agent:

* **Memory** — `deepagents/projects/AGENTS.md`, always injected into the system
  prompt via `memory=[...]`. Durable "who you are / how you behave" context that
  lives in a versioned file instead of a hard-coded string.
* **Skills** — `deepagents/skills/<name>/{SKILL.md,instructions.md,examples.md}`,
  surfaced through `skills=[...]` with progressive disclosure: the agent sees
  each skill's name + description up front and `read_file`s the details only when
  a skill is actually relevant.

For StateBackend/StoreBackend these files have to be *seeded* into the agent's
virtual filesystem (they are not on the agent's disk). FilesystemBackend reads
them straight from `root_dir`, so nothing needs seeding there.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.config import AGENTS_MD_PATH, SKILLS_DIR

# Files inside a skill directory worth seeding into a virtual filesystem.
SKILL_FILE_GLOBS = ("*.md", "*.py", "*.txt", "*.json")


@dataclass(frozen=True)
class SkillInfo:
    """A skill discovered on disk, as the sidebar shows it."""

    name: str
    description: str
    directory: Path

    @property
    def files(self) -> list[Path]:
        found: list[Path] = []
        for pattern in SKILL_FILE_GLOBS:
            found.extend(sorted(self.directory.glob(pattern)))
        return found


def _parse_frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        data = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def discover_skills(skills_dir: Path = SKILLS_DIR) -> list[SkillInfo]:
    """Every `<skills_dir>/<name>/SKILL.md` on disk, with its frontmatter."""
    if not skills_dir.is_dir():
        return []
    skills: list[SkillInfo] = []
    for child in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
        skill_md = child / "SKILL.md"
        if not skill_md.is_file():
            continue
        meta = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        skills.append(
            SkillInfo(
                name=str(meta.get("name") or child.name),
                description=str(meta.get("description") or "").strip(),
                directory=child,
            )
        )
    return skills


def load_agents_md(path: Path = AGENTS_MD_PATH) -> str:
    """The AGENTS.md context file, or '' if it is missing."""
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def build_context_files(
    *,
    include_memory: bool,
    skill_names: tuple[str, ...] | list[str],
    agents_md_virtual_path: str,
    skills_virtual_root: str,
) -> dict[str, Any]:
    """Virtual-path -> FileData mapping to seed into a non-disk backend.

    Mirrors notebook 2's `create_file_data` seeding, but seeds *all* of a skill's
    supporting files (not just SKILL.md) so `read_file` on `instructions.md` /
    `examples.md` actually resolves.
    """
    from deepagents.backends.utils import create_file_data

    files: dict[str, Any] = {}

    if include_memory:
        content = load_agents_md()
        if content:
            files[agents_md_virtual_path] = create_file_data(content)

    wanted = set(skill_names)
    root = skills_virtual_root.rstrip("/")
    for skill in discover_skills():
        if skill.directory.name not in wanted and skill.name not in wanted:
            continue
        for file_path in skill.files:
            virtual = f"{root}/{skill.directory.name}/{file_path.name}"
            files[virtual] = create_file_data(file_path.read_text(encoding="utf-8"))

    return files
