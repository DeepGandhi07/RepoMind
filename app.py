import html

import streamlit as st

from repomind.ingest import RepositoryError, is_local_source
from repomind.llm import LLM_MODEL, LLMUnavailableError, ModelUnavailableError
from repomind.service import analyze, ask, get_repository, list_repositories, remove

SUGGESTIONS = [
    "What does this project do?",
    "Explain the overall architecture.",
    "How does error handling work?",
    "Where is configuration loaded from?",
    "How would I add a new feature safely?",
    "What are the main entry points?",
]

GROUNDING_STYLE = {
    "supported": ("Supported", "#16a34a"),
    "partially_supported": ("Partially supported", "#d97706"),
    "unsupported": ("Unsupported", "#dc2626"),
    "insufficient_evidence": ("Insufficient evidence", "#64748b"),
    "unavailable": ("Check unavailable", "#94a3b8"),
    "skipped": ("Check skipped", "#94a3b8"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatMessage, button, input, textarea {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
.block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 980px; }
#MainMenu, footer { visibility: hidden; }
.rm-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 45%, #0e7490 100%);
    color: #fff; border-radius: 22px; padding: 30px 34px; margin-bottom: 22px;
    box-shadow: 0 18px 40px -12px rgba(76, 29, 149, .45);
}
.rm-hero h1 { margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -.02em; color: #fff; }
.rm-hero p { margin: .4rem 0 0; font-size: 1rem; opacity: .88; max-width: 640px; }
.rm-chip {
    display: inline-block; margin-top: 14px; padding: 4px 12px; border-radius: 999px;
    background: rgba(255,255,255,.14); font-size: .8rem; font-weight: 500;
}
.rm-card {
    border: 1px solid rgba(128,128,128,.22); border-radius: 16px;
    padding: 14px 16px; margin-bottom: 10px; background: rgba(127,127,127,.04);
}
.rm-card h4 { margin: 0 0 4px; font-size: .95rem; font-weight: 600; word-break: break-all; }
.rm-muted { opacity: .7; font-size: .85rem; word-break: break-all; }
.rm-stats { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
.rm-stat {
    flex: 1; min-width: 70px; text-align: center; padding: 8px 6px; border-radius: 12px;
    background: rgba(124,58,237,.10);
}
.rm-stat b { display: block; font-size: 1.1rem; color: #7c3aed; }
.rm-stat span { font-size: .72rem; opacity: .75; text-transform: uppercase; letter-spacing: .05em; }
.rm-pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
    border-radius: 999px; font-size: .75rem; font-weight: 600; color: #fff; margin: 4px 6px 0 0;
}
.rm-dot { width: 7px; height: 7px; border-radius: 50%; background: #fff; opacity: .85; }
.rm-pill-soft { background: rgba(127,127,127,.15); color: inherit; }
.rm-source {
    display: flex; justify-content: space-between; align-items: center; gap: 10px;
    border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 9px 12px; margin-bottom: 8px;
}
.rm-file { font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace; font-size: .85rem; word-break: break-all; }
.rm-lines {
    font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace; font-size: .75rem;
    background: rgba(14,116,144,.14); color: #0e7490; padding: 3px 8px; border-radius: 8px; white-space: nowrap;
}
.rm-empty {
    text-align: center; padding: 44px 22px; border-radius: 20px;
    border: 1px dashed rgba(128,128,128,.4); margin-top: 6px;
}
.rm-empty h3 { margin: 6px 0 4px; font-weight: 650; }
.rm-empty-icon { font-size: 2.2rem; }
div[data-testid="stChatMessage"] { border-radius: 18px; padding: 6px 10px; margin-bottom: 6px; }
div[data-testid="stChatInput"] textarea { border-radius: 14px; }
.stButton > button { border-radius: 12px; font-weight: 500; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #7c3aed, #0e7490); border: none; color: #fff; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #fff; }
</style>
"""


def grounding_pill(status):
    label, color = GROUNDING_STYLE.get(status or "unavailable", GROUNDING_STYLE["unavailable"])
    return (
        f'<span class="rm-pill" style="background:{color}">'
        f'<span class="rm-dot"></span>{html.escape(label)}</span>'
    )


def render_sources(sources):
    if not sources:
        st.caption("No source files were retrieved.")
        return
    cards = []
    for source in sources:
        lines = ""
        if "start_line" in source and "end_line" in source:
            lines = f'<span class="rm-lines">lines {source["start_line"]}–{source["end_line"]}</span>'
        cards.append(f'<div class="rm-source"><span class="rm-file">{html.escape(source["file"])}</span>{lines}</div>')
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_assistant(message):
    if message.get("error"):
        st.warning(message["error"], icon="⏳")
        return
    st.markdown(message["content"])
    meta = grounding_pill(message.get("grounding")) + (
        f'<span class="rm-pill rm-pill-soft">{len(message.get("sources", []))} code sources</span>'
    )
    st.markdown(meta, unsafe_allow_html=True)
    with st.expander("View retrieved code sources"):
        render_sources(message.get("sources", []))


def render_empty_state(has_repo):
    if has_repo:
        st.markdown(
            '<div class="rm-empty"><div class="rm-empty-icon">💬</div>'
            "<h3>Ask anything about the code</h3>"
            '<div class="rm-muted">Answers cite file paths and line ranges.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rm-empty"><div class="rm-empty-icon">📦</div>'
            "<h3>Start by analyzing a repository</h3>"
            '<div class="rm-muted">Paste a public or token-accessible https repository URL in the sidebar.</div></div>',
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="RepoMind", page_icon="🔍", layout="centered", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("selected_repo", None)

records = list_repositories()
keys = [record.key for record in records]
if st.session_state.selected_repo not in keys:
    st.session_state.selected_repo = keys[0] if keys else None
selected = get_repository(st.session_state.selected_repo) if st.session_state.selected_repo else None

with st.sidebar:
    st.markdown("### 📦 Repositories")
    repo_url = st.text_input(
        "Repository URL",
        placeholder="https://github.com/psf/requests",
        label_visibility="collapsed",
    )
    if st.button("Analyze repository", type="primary", use_container_width=True):
        if not repo_url.strip():
            st.error("Enter a repository URL first.")
        elif is_local_source(repo_url):
            st.error("Local folders can only be analyzed from the command line, for security reasons.")
        else:
            with st.status("Indexing repository", expanded=True) as status:
                try:
                    report = analyze(repo_url.strip(), on_step=lambda message: st.write(message))
                    status.update(label=f"Indexed {report.record.name}", state="complete")
                    st.session_state.selected_repo = report.record.key
                    st.session_state.messages = []
                except RepositoryError as exc:
                    status.update(label="Indexing failed", state="error")
                    st.error(str(exc))
                except Exception as exc:
                    status.update(label="Indexing failed", state="error")
                    st.error(f"Unexpected error: {exc}")
            st.rerun()

    if records:
        labels = {record.key: record.name for record in records}
        chosen = st.selectbox(
            "Active repository",
            options=keys,
            format_func=lambda key: labels[key],
            index=keys.index(st.session_state.selected_repo),
        )
        if chosen != st.session_state.selected_repo:
            st.session_state.selected_repo = chosen
            st.session_state.messages = []
            st.rerun()

    if selected:
        top_languages = ", ".join(list(selected.languages)[:5]) or "—"
        st.markdown(
            f'<div class="rm-card"><h4>{html.escape(selected.name)}</h4>'
            f'<div class="rm-muted">{html.escape(selected.source)}</div>'
            '<div class="rm-stats">'
            f'<div class="rm-stat"><b>{selected.files}</b><span>files</span></div>'
            f'<div class="rm-stat"><b>{selected.chunks}</b><span>chunks</span></div>'
            "</div>"
            f'<div class="rm-muted" style="margin-top:10px">Languages: {html.escape(top_languages)}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Code chunks per answer", min_value=3, max_value=10, value=5)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if selected and st.button("Remove", use_container_width=True):
            remove(selected.key)
            st.session_state.selected_repo = None
            st.session_state.messages = []
            st.rerun()

    st.markdown(
        '<div class="rm-muted" style="margin-top:18px">Code is cloned and indexed on the server. '
        "Questions and retrieved snippets are sent to Gemini to write answers.</div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="rm-hero"><h1>RepoMind</h1>'
    "<p>Ask questions about any source repository and get answers grounded in its code, "
    "with file and line citations.</p>"
    '<span class="rm-chip">Any language · Hybrid search · Grounded answers</span></div>',
    unsafe_allow_html=True,
)

has_repo = selected is not None
clicked_suggestion = None
retry_question = None

empty_slot = st.empty()
with empty_slot.container():
    if not st.session_state.messages:
        render_empty_state(has_repo)
        if has_repo:
            columns = st.columns(2)
            for position, suggestion in enumerate(SUGGESTIONS):
                with columns[position % 2]:
                    if st.button(suggestion, key=f"suggest-{position}", use_container_width=True):
                        clicked_suggestion = suggestion

for position, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant(message)
            if message.get("error") and st.button("Try again", key=f"retry-{position}"):
                retry_question = message["question"]

typed_question = st.chat_input("Ask about the repository…", disabled=not has_repo)
question = typed_question or clicked_suggestion or retry_question

if question:
    if not has_repo:
        st.warning("Analyze a repository first.")
    else:
        if retry_question:
            cleaned = [m for m in st.session_state.messages if m.get("error") is None]
            if cleaned and cleaned[-1]["role"] == "user" and cleaned[-1]["content"] == question:
                cleaned.pop()
            st.session_state.messages = cleaned

        empty_slot.empty()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the code and writing an answer…"):
                    result = ask(question, repo=selected.key, k=top_k)
                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "grounding": (result.get("grounding") or {}).get("overall_status"),
                }
                render_assistant(assistant_message)
            except ModelUnavailableError as exc:
                assistant_message = {
                    "role": "assistant",
                    "error": f"{exc} The current setting is '{LLM_MODEL}'.",
                    "question": question,
                }
                render_assistant(assistant_message)
            except LLMUnavailableError:
                assistant_message = {
                    "role": "assistant",
                    "error": "The language model is busy or over its usage limit right now. RepoMind already retried automatically. Please wait a moment and try again.",
                    "question": question,
                }
                render_assistant(assistant_message)
            except Exception as exc:
                assistant_message = {
                    "role": "assistant",
                    "error": f"Something went wrong while answering: {exc}",
                    "question": question,
                }
                render_assistant(assistant_message)

        st.session_state.messages.append(assistant_message)
        if assistant_message.get("error"):
            st.rerun()
