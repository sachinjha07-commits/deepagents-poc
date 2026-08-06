"""Turn a deep-agent run into a stream of UI events, and inspect its state.

`stream_turn` flattens LangGraph's `("updates", "messages")` stream (with
`subgraphs=True`, so subagent activity is visible too) into simple dicts the
Streamlit layer can render as it goes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from app.config import STORE_NAMESPACE
from app.agent_builder import AgentBundle

# Files a FilesystemBackend listing should never show.
_SKIP_DIRS = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "node_modules"}
_TEXT_SUFFIXES = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".csv", ".ipynb", ""}
_MAX_LISTED_FILES = 300


def message_text(message: Any) -> str:
    """Plain text of a message, tolerating LangChain's block content format."""
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def thread_config(bundle: AgentBundle, thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": bundle.config.recursion_limit,
    }


def stream_turn(
    bundle: AgentBundle,
    user_text: str,
    thread_id: str,
) -> Iterator[dict[str, Any]]:
    """Run one conversational turn, yielding events as they happen.

    Event types: ``token``, ``tool_call``, ``tool_result``, ``assistant_message``,
    ``todos``, ``files``, ``structured``, ``error``.
    """
    payload: dict[str, Any] = {"messages": [{"role": "user", "content": user_text}]}
    if bundle.needs_file_seeding:
        # StateBackend keeps files in *this thread's* state, so AGENTS.md and the
        # skill files have to ride along on every invoke (notebook 2).
        payload["files"] = bundle.seed_files

    config = thread_config(bundle, thread_id)
    # Maps a `task` tool-call id to the subagent it launched, so nested events
    # can be labelled with the subagent's name.
    delegations: dict[str, str] = {}
    current_subagent: str | None = None

    try:
        for namespace, mode, chunk in bundle.graph.stream(
            payload,
            config=config,
            stream_mode=["updates", "messages"],
            subgraphs=True,
        ):
            depth = len(namespace or ())

            if mode == "messages":
                message, meta = chunk
                # "messages" also streams ToolMessages; only the model's own
                # AI chunks belong in the visible answer.
                if not message.__class__.__name__.startswith("AI"):
                    continue
                if (meta or {}).get("langgraph_node") not in (None, "model"):
                    continue
                text = message_text(message)
                if text:
                    yield {
                        "type": "token",
                        "text": text,
                        "depth": depth,
                        "agent": current_subagent if depth else None,
                    }
                continue

            if not isinstance(chunk, dict):
                continue

            for _node, update in chunk.items():
                if not isinstance(update, dict):
                    continue

                if "todos" in update:
                    yield {"type": "todos", "todos": update["todos"] or []}
                if "files" in update:
                    yield {"type": "files", "files": update["files"] or {}}
                if update.get("structured_response") is not None:
                    # `response_format` output — from the supervisor (depth 0) or
                    # from a subagent running in its own context (depth > 0).
                    yield {
                        "type": "structured",
                        "value": update["structured_response"],
                        "depth": depth,
                        "agent": current_subagent if depth else None,
                    }

                for message in update.get("messages", []) or []:
                    tool_calls = getattr(message, "tool_calls", None) or []
                    for call in tool_calls:
                        name = call.get("name", "?")
                        args = call.get("args", {}) or {}
                        if name == "task":
                            current_subagent = str(
                                args.get("subagent_type") or args.get("agent") or "subagent"
                            )
                            delegations[str(call.get("id"))] = current_subagent
                        yield {
                            "type": "tool_call",
                            "name": name,
                            "args": args,
                            "id": str(call.get("id") or ""),
                            "depth": depth,
                            "agent": current_subagent if depth else None,
                        }

                    if message.__class__.__name__ == "ToolMessage":
                        call_id = str(getattr(message, "tool_call_id", "") or "")
                        if call_id in delegations:
                            current_subagent = None
                        yield {
                            "type": "tool_result",
                            "name": getattr(message, "name", "") or "tool",
                            "content": message_text(message),
                            "id": call_id,
                            "depth": depth,
                        }
                    elif message.__class__.__name__.startswith("AI") and not tool_calls:
                        text = message_text(message)
                        if text and depth == 0:
                            yield {"type": "assistant_message", "text": text}
    except Exception as exc:  # surfaced in the chat instead of crashing the app
        yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}


# --- State / file inspection -------------------------------------------------


def get_state_values(bundle: AgentBundle, thread_id: str) -> dict[str, Any]:
    try:
        return dict(bundle.graph.get_state(thread_config(bundle, thread_id)).values or {})
    except Exception:
        return {}


def get_todos(bundle: AgentBundle, thread_id: str) -> list[dict[str, Any]]:
    todos = get_state_values(bundle, thread_id).get("todos") or []
    return [dict(t) if not isinstance(t, dict) else t for t in todos]


def _file_content(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("content", ""))
    return str(value)


def list_files(bundle: AgentBundle, thread_id: str) -> dict[str, str]:
    """Every file the agent can currently see, whatever the backend."""
    kind = bundle.config.backend

    if kind == "state":
        files = get_state_values(bundle, thread_id).get("files") or {}
        return {path: _file_content(value) for path, value in sorted(files.items())}

    if kind == "store" and bundle.store is not None:
        try:
            items = bundle.store.search(STORE_NAMESPACE, limit=_MAX_LISTED_FILES)
        except Exception:
            return {}
        return {
            str(item.key): _file_content(item.value)
            for item in sorted(items, key=lambda i: str(i.key))
        }

    if kind == "filesystem":
        root = Path(bundle.config.filesystem_root).expanduser()
        if not root.is_dir():
            return {}
        out: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if len(out) >= _MAX_LISTED_FILES:
                break
            if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
                continue
            if any(part in _SKIP_DIRS or part.startswith(".") for part in path.parts):
                continue
            virtual = "/" + path.relative_to(root).as_posix()
            try:
                out[virtual] = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
        return out

    return {}


def pretty(value: Any, limit: int = 4000) -> str:
    """Compact JSON for tool args / results, truncated for display."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, indent=2, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > limit:
        return text[:limit] + f"\n… [{len(text) - limit} more characters]"
    return text
