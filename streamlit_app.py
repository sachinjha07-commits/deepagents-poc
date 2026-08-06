"""Deep Agent Studio — a conversational chatbot exercising every feature from
the notebooks in `deepagents/`.

Run it with:

    uv run streamlit run streamlit_app.py

What is wired up:

* **Planning** — the `write_todos` tool; the live plan is shown in the Plan tab.
* **Virtual filesystem** — `ls` / `read_file` / `write_file` / `edit_file`, with
  the resulting files browsable in the Files tab.
* **Backends** — StateBackend, FilesystemBackend and StoreBackend, switchable at
  runtime (notebook 3).
* **Context engineering** — `AGENTS.md` loaded as memory, plus Agent Skills with
  progressive disclosure (notebook 2).
* **Subagents** — a researcher and a critic with their own context windows,
  optional model override and optional structured output (notebook 4).
* **Custom tools** — Tavily web search (notebook 1).
* **Code interpreter** — the QuickJS sandbox middleware (notebook 2).
* **Conversation memory** — a `MemorySaver` checkpointer keyed by thread id.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app.agent_builder import AgentBundle, build_agent
from app.config import (
    AGENTS_MD_PATH,
    BACKEND_CHOICES,
    BACKEND_LABELS,
    DEEPAGENTS_DIR,
    DEFAULT_MODEL,
    DEFAULT_SUBAGENT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    ENV_PATH,
    MODEL_CHOICES,
    AgentConfig,
)
from app.context import discover_skills, load_agents_md
from app.runner import get_todos, list_files, pretty, stream_turn

load_dotenv(ENV_PATH)

st.set_page_config(page_title="Deep Agent Studio", page_icon="🧠", layout="wide")

TOOL_ICONS = {
    "write_todos": "📋",
    "write_file": "💾",
    "edit_file": "✏️",
    "read_file": "📖",
    "ls": "📂",
    "internet_search": "🌐",
    "task": "🤖",
    "eval": "⚙️",
}
TODO_ICONS = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}

EXAMPLE_PROMPTS = {
    "Who are you?": "Who are you, what is in your memory, and which skills and subagents do you have available?",
    "Plan + write a file": "Plan this with todos first: research what an LLM gateway is, save your raw notes to /notes/llm-gateway-notes.md, then give me a 5-bullet summary.",
    "Delegate research": "Delegate to your research subagent: research recent advances in LLM routing, then critique the result with your critique subagent before answering.",
    "Use a skill": "How do I build a LangGraph graph with conditional routing and memory? Show a minimal example.",
    "Write a report": "Write me a short report on deep agent architecture and save it under /reports/.",
}


# --- Session state -----------------------------------------------------------


def _init_state() -> None:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore

    st.session_state.setdefault("checkpointer", MemorySaver())
    # One store for the whole session: StoreBackend files survive every thread.
    st.session_state.setdefault("store", InMemoryStore())
    st.session_state.setdefault("thread_id", f"chat-{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("bundle", None)
    st.session_state.setdefault("bundle_sig", None)
    st.session_state.setdefault("build_error", None)


def _get_bundle(cfg: AgentConfig) -> AgentBundle | None:
    """Build the agent, reusing the previous one while the config is unchanged."""
    if st.session_state.bundle_sig == cfg.signature() and st.session_state.bundle:
        return st.session_state.bundle
    try:
        bundle = build_agent(
            cfg,
            checkpointer=st.session_state.checkpointer,
            store=st.session_state.store,
        )
    except Exception as exc:
        st.session_state.build_error = f"{type(exc).__name__}: {exc}"
        st.session_state.bundle = None
        st.session_state.bundle_sig = None
        return None
    st.session_state.build_error = None
    st.session_state.bundle = bundle
    st.session_state.bundle_sig = cfg.signature()
    return bundle


def _new_thread() -> None:
    st.session_state.thread_id = f"chat-{uuid.uuid4().hex[:8]}"
    st.session_state.messages = []


# --- Sidebar -----------------------------------------------------------------


def sidebar_config(skills) -> AgentConfig:
    st.sidebar.title("🧠 Deep Agent Studio")
    st.sidebar.caption("Every feature from the `deepagents/` notebooks, in one chat.")

    if not ENV_PATH.is_file():
        st.sidebar.warning(f"No .env found at {ENV_PATH}")

    # --- Model (notebook 1) ---
    st.sidebar.subheader("Model")
    choices = [*MODEL_CHOICES, "custom…"]
    default_index = choices.index(DEFAULT_MODEL) if DEFAULT_MODEL in choices else 0
    picked = st.sidebar.selectbox("Supervisor model", choices, index=default_index)
    model = (
        st.sidebar.text_input("Custom model string", value=DEFAULT_MODEL, key="custom_model")
        if picked == "custom…"
        else picked
    )

    # --- Backend (notebook 3) ---
    st.sidebar.subheader("Backend")
    backend = st.sidebar.radio(
        "Where the agent's files live",
        BACKEND_CHOICES,
        format_func=lambda b: b,
        help="\n\n".join(f"**{k}** — {v}" for k, v in BACKEND_LABELS.items()),
    )
    st.sidebar.caption(BACKEND_LABELS[backend])
    filesystem_root = str(DEEPAGENTS_DIR)
    if backend == "filesystem":
        filesystem_root = st.sidebar.text_input(
            "root_dir (real disk — the agent can write here)",
            value=str(DEEPAGENTS_DIR),
        )

    # --- Capabilities (notebooks 1 & 2) ---
    st.sidebar.subheader("Capabilities")
    planning = st.sidebar.toggle(
        "Planning (`write_todos`)", value=True, help="The todo-list pillar of a deep agent."
    )
    web_search = st.sidebar.toggle(
        "Web search (Tavily)", value=True, help="The custom `internet_search` tool."
    )
    code_interpreter = st.sidebar.toggle(
        "Code interpreter (QuickJS)",
        value=False,
        help="Adds langchain-quickjs' sandboxed `eval` tool.",
    )

    # --- Context engineering (notebook 2) ---
    st.sidebar.subheader("Context engineering")
    memory = st.sidebar.toggle(
        "Load `AGENTS.md` as memory",
        value=AGENTS_MD_PATH.is_file(),
        disabled=not AGENTS_MD_PATH.is_file(),
        help=str(AGENTS_MD_PATH),
    )
    skills_on = st.sidebar.toggle("Agent Skills", value=bool(skills), disabled=not skills)
    skill_names: list[str] = []
    if skills_on and skills:
        skill_names = st.sidebar.multiselect(
            "Skills to expose",
            [s.directory.name for s in skills],
            default=[s.directory.name for s in skills],
        )

    # --- Subagents (notebook 4) ---
    st.sidebar.subheader("Subagents")
    subagents = st.sidebar.toggle("Enable subagents (`task` tool)", value=True)
    subagent_model = DEFAULT_SUBAGENT_MODEL
    structured_research = False
    if subagents:
        sub_choices = [f"same as supervisor ({model})", *MODEL_CHOICES]
        sub_picked = st.sidebar.selectbox("Researcher model override", sub_choices, index=0)
        subagent_model = model if sub_picked.startswith("same as supervisor") else sub_picked
        structured_research = st.sidebar.toggle(
            "Structured findings (`response_format`)",
            value=False,
            help="The researcher returns summary / confidence / sources as a Pydantic model.",
        )

    # --- Prompt + thread ---
    with st.sidebar.expander("System prompt"):
        system_prompt = st.text_area("system_prompt", value=DEFAULT_SYSTEM_PROMPT, height=180)

    st.sidebar.subheader("Conversation")
    st.sidebar.caption(f"thread_id: `{st.session_state.thread_id}`")
    st.sidebar.button("🧵 New thread", use_container_width=True, on_click=_new_thread)

    return AgentConfig(
        model=model,
        system_prompt=system_prompt,
        backend=backend,
        filesystem_root=filesystem_root,
        planning=planning,
        web_search=web_search,
        code_interpreter=code_interpreter,
        memory=memory,
        skills=skills_on,
        subagents=subagents,
        subagent_model=subagent_model,
        structured_research=structured_research,
        skill_names=tuple(skill_names),
    )


# --- Rendering helpers -------------------------------------------------------


def render_trace(trace: list[dict], container) -> None:
    """Render a finished turn's tool activity."""
    for event in trace:
        kind = event["type"]
        indent = "&nbsp;" * 4 * min(event.get("depth", 0), 3)
        if kind == "tool_call":
            icon = TOOL_ICONS.get(event["name"], "🔧")
            agent = f" _(via {event['agent']})_" if event.get("agent") else ""
            container.markdown(f"{indent}{icon} **{event['name']}**{agent}", unsafe_allow_html=True)
            if event["args"]:
                container.code(pretty(event["args"], 1200), language="json")
        elif kind == "tool_result":
            container.markdown(f"{indent}↳ `{event['name']}` result", unsafe_allow_html=True)
            container.code(pretty(event["content"], 1500))
        elif kind == "structured":
            agent = event.get("agent") or "subagent"
            container.markdown(f"{indent}🧾 structured findings from **{agent}**", unsafe_allow_html=True)
            value = event["value"]
            container.json(value.model_dump() if hasattr(value, "model_dump") else value)


def render_plan(bundle: AgentBundle) -> None:
    if not bundle.config.planning:
        st.caption("Planning is off — enable `write_todos` in the sidebar.")
        return
    todos = get_todos(bundle, st.session_state.thread_id)
    if not todos:
        st.caption("No plan yet. Ask for something multi-step and the agent will write todos.")
        return
    done = sum(1 for t in todos if t.get("status") == "completed")
    st.progress(done / len(todos), text=f"{done}/{len(todos)} complete")
    for todo in todos:
        status = str(todo.get("status", "pending"))
        st.markdown(f"{TODO_ICONS.get(status, '⬜')} {todo.get('content', '')}")


def render_files(bundle: AgentBundle) -> None:
    files = list_files(bundle, st.session_state.thread_id)
    st.caption(
        f"{len(files)} file(s) — {BACKEND_LABELS[bundle.config.backend].split('—')[1].strip()}"
    )
    if not files:
        st.caption("Nothing written yet. Ask the agent to save notes or a report.")
        return
    path = st.selectbox("File", list(files), key="file_browser")
    content = files.get(path, "")
    st.download_button(
        "Download", content, file_name=Path(path).name, use_container_width=True
    )
    language = "markdown" if path.endswith(".md") else None
    st.code(content[:20000], language=language)


def render_context(bundle: AgentBundle, skills) -> None:
    st.markdown("**Active configuration**")
    for feature in bundle.features:
        st.markdown(f"- {feature}")
    st.markdown(f"**Tools:** {', '.join(f'`{t}`' for t in bundle.tool_names)}")
    for note in bundle.notes:
        st.info(note, icon="ℹ️")

    if bundle.config.memory:
        with st.expander("Memory — AGENTS.md"):
            st.markdown(load_agents_md()[:8000] or "_empty_")

    if bundle.config.skills and bundle.config.skill_names:
        st.markdown("**Skills (progressive disclosure)**")
        for skill in skills:
            if skill.directory.name not in bundle.config.skill_names:
                continue
            with st.expander(skill.name):
                st.caption(skill.description or "_no description_")
                st.markdown(
                    "Files: " + ", ".join(f"`{f.name}`" for f in skill.files)
                )


# --- Main --------------------------------------------------------------------


def run_turn(bundle: AgentBundle, prompt: str) -> None:
    """Stream one turn into the chat and record it in the history."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    trace: list[dict] = []
    answer = ""
    structured = None
    error = None

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)
        answer_slot = st.empty()
        subagent_slot = None
        subagent_text = ""

        for event in stream_turn(bundle, prompt, st.session_state.thread_id):
            kind = event["type"]
            if kind == "token":
                if event["depth"] == 0:
                    answer += event["text"]
                    answer_slot.markdown(answer + "▌")
                else:
                    # Live view of what a subagent is producing in its own context.
                    if subagent_slot is None:
                        subagent_slot = status.empty()
                    subagent_text += event["text"]
                    label = event.get("agent") or "subagent"
                    subagent_slot.markdown(f"💬 **{label}** — {subagent_text[-1200:]}")
            elif kind in ("tool_call", "tool_result"):
                if kind == "tool_call":
                    subagent_slot, subagent_text = None, ""
                    status.update(label=f"{TOOL_ICONS.get(event['name'], '🔧')} {event['name']}")
                trace.append(event)
                render_trace([event], status)
            elif kind == "assistant_message":
                if not answer.strip():
                    answer = event["text"]
                    answer_slot.markdown(answer)
            elif kind == "structured":
                if event.get("depth", 0) == 0:
                    structured = event["value"]
                else:
                    trace.append(event)
                    render_trace([event], status)
            elif kind == "error":
                error = event["message"]

        status.update(label=f"Done — {len(trace)} step(s)", state="complete", expanded=False)
        if error:
            st.error(error)
        answer_slot.markdown(answer or "_(no text returned)_")
        if structured is not None:
            st.markdown("**Structured response**")
            st.json(
                structured.model_dump() if hasattr(structured, "model_dump") else structured
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "trace": trace,
            "structured": structured,
            "error": error,
        }
    )


def main() -> None:
    _init_state()
    skills = discover_skills()
    cfg = sidebar_config(skills)

    missing = cfg.missing_api_keys()
    bundle = None if missing else _get_bundle(cfg)

    st.title("Deep Agent Studio")
    st.caption(
        "Planning · virtual filesystem · pluggable backends · AGENTS.md memory · "
        "skills · subagents · Tavily search"
    )

    if missing:
        st.error(
            "Missing environment variable(s): "
            + ", ".join(f"`{k}`" for k in missing)
            + f". Add them to `{ENV_PATH}` or turn off the features that need them."
        )
    if st.session_state.build_error:
        st.error(f"Could not build the agent — {st.session_state.build_error}")

    chat_col, panel_col = st.columns([3, 2], gap="large")

    prompt = st.chat_input(
        "Ask the deep agent something…" if bundle else "Fix the configuration above to start",
        disabled=bundle is None,
    )
    if not prompt:
        prompt = st.session_state.pop("pending_prompt", None)

    with chat_col:
        if bundle:
            extras = " · ".join(bundle.features[1:])
            st.markdown(
                f"`{cfg.model}` · **{cfg.backend}** backend" + (f" · {extras}" if extras else "")
            )
        if not st.session_state.messages:
            st.markdown("**Try one of these:**")
            cols = st.columns(2)
            for index, (label, text) in enumerate(EXAMPLE_PROMPTS.items()):
                if cols[index % 2].button(label, use_container_width=True, disabled=bundle is None):
                    st.session_state.pending_prompt = text
                    st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message.get("trace"):
                    with st.expander(f"🔎 {len(message['trace'])} agent step(s)"):
                        render_trace(message["trace"], st)
                st.markdown(message["content"] or "_(no text returned)_")
                if message.get("structured") is not None:
                    value = message["structured"]
                    st.json(value.model_dump() if hasattr(value, "model_dump") else value)
                if message.get("error"):
                    st.error(message["error"])

        if prompt and bundle:
            run_turn(bundle, prompt)

    with panel_col:
        if bundle:
            plan_tab, files_tab, context_tab = st.tabs(["📋 Plan", "📁 Files", "🧠 Context"])
            with plan_tab:
                render_plan(bundle)
            with files_tab:
                render_files(bundle)
            with context_tab:
                render_context(bundle, skills)


main()
