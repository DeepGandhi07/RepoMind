"""Page composition and interaction logic for the RepoMind web app."""

import streamlit as st

from repomind.ingest import is_local_source
from repomind.llm import (
    DEFAULT_MODELS,
    PROVIDER_NAMES,
    LLMConfig,
    LLMUnavailableError,
    ModelUnavailableError,
    list_models,
    server_key_configured,
)
from repomind.service import analyze, ask, get_repository, list_repositories, remove
from repomind.ui.components import (
    render_answer,
    render_empty_state,
    render_error,
    render_hero,
    render_how_it_works,
    render_repo_summary,
    render_suggestions,
)
from repomind.ui.theme import apply_theme

MESSAGES = "messages"
SELECTED = "selected_repo"
CONNECTION = "connection"
PROVIDER_CHOICES = {"Google Gemini": "gemini", "Groq": "groq"}
BUSY_MESSAGE = "The model provider is busy or over its usage limit right now. Wait a moment and try again."


def run() -> None:
    st.set_page_config(
        page_title="RepoMind",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    apply_theme()
    st.session_state.setdefault(MESSAGES, [])
    st.session_state.setdefault(SELECTED, None)
    st.session_state.setdefault(CONNECTION, None)

    records = list_repositories()
    selected = _resolve_selection(records)
    llm_config = _active_llm_config()

    top_k = _render_sidebar(records, selected)

    provider_label = PROVIDER_NAMES[llm_config.provider] if llm_config else ""
    render_hero(provider_label, llm_config.model if llm_config else None)
    if not st.session_state[MESSAGES]:
        render_how_it_works()
    _render_conversation(selected, llm_config, top_k)


def _resolve_selection(records):
    keys = [record.key for record in records]
    if st.session_state[SELECTED] not in keys:
        st.session_state[SELECTED] = keys[0] if keys else None
    if st.session_state[SELECTED] is None:
        return None
    return get_repository(st.session_state[SELECTED])


def _active_llm_config() -> LLMConfig | None:
    connection = st.session_state[CONNECTION]
    if connection:
        return LLMConfig(
            provider=connection["provider"],
            model=connection["model"],
            api_key=connection["api_key"],
        )
    if server_key_configured():
        return LLMConfig.from_env()
    return None


def _render_sidebar(records, selected) -> int:
    with st.sidebar:
        st.markdown(
            '<div class="rm-brand"><div class="rm-logo">R</div>'
            '<div><div class="rm-brand-name">RepoMind</div>'
            '<div class="rm-brand-tag">Chat with any codebase</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("##### Model access")
        _render_connection()

        st.markdown("##### Repository")
        url = st.text_input(
            "Repository URL",
            placeholder="https://github.com/owner/repo",
            label_visibility="collapsed",
        )
        if st.button("Analyze repository", type="primary"):
            _analyze(url)

        if records:
            _render_repository_switcher(records, selected)
        if selected:
            render_repo_summary(selected)
            if st.button("Remove repository"):
                remove(selected.key)
                st.session_state[SELECTED] = None
                st.session_state[MESSAGES] = []
                st.rerun()

        st.markdown("##### Answers")
        top_k = st.slider("Code chunks per answer", min_value=3, max_value=10, value=5)
        if st.button("Clear conversation"):
            st.session_state[MESSAGES] = []
            st.rerun()

    return top_k


def _render_connection() -> None:
    connection = st.session_state[CONNECTION]

    if connection:
        st.success(f"Connected to {PROVIDER_NAMES[connection['provider']]}", icon="🔒")
        models = connection["models"]
        index = models.index(connection["model"]) if connection["model"] in models else 0
        connection["model"] = st.selectbox("Model", options=models, index=index)
        st.caption("Your key is held only in this browser session.")
        if st.button("Disconnect"):
            st.session_state[CONNECTION] = None
            st.rerun()
        return

    provider_label = st.selectbox("Provider", list(PROVIDER_CHOICES))
    api_key = st.text_input("API key", type="password", placeholder="Paste your API key")
    st.caption("Bring your own key. It is kept only in this browser session and never written to disk.")
    if st.button("Connect", type="primary", disabled=not api_key.strip()):
        _connect(PROVIDER_CHOICES[provider_label], api_key.strip())


def _connect(provider: str, api_key: str) -> None:
    try:
        with st.spinner("Checking your key"):
            models = list_models(LLMConfig(provider=provider, model=DEFAULT_MODELS[provider], api_key=api_key))
    except Exception as exc:
        st.error(_scrub(str(exc), api_key) or "This key could not be verified.")
        return

    if not models:
        st.error("This key has no chat models available.")
        return

    preferred = DEFAULT_MODELS[provider]
    st.session_state[CONNECTION] = {
        "provider": provider,
        "api_key": api_key,
        "models": models,
        "model": preferred if preferred in models else models[0],
    }
    st.rerun()


def _render_repository_switcher(records, selected) -> None:
    labels = {record.key: record.name for record in records}
    keys = list(labels)
    current = selected.key if selected else keys[0]
    choice = st.selectbox(
        "Active repository",
        options=keys,
        index=keys.index(current),
        format_func=labels.get,
    )
    if choice != current:
        st.session_state[SELECTED] = choice
        st.session_state[MESSAGES] = []
        st.rerun()


def _analyze(url: str) -> None:
    url = url.strip()
    if not url:
        st.error("Enter a repository URL.")
        return
    if is_local_source(url):
        st.error("Local folders can only be analyzed from the command line.")
        return

    with st.status("Indexing repository", expanded=True) as status:
        try:
            report = analyze(url, on_step=st.write)
        except Exception as exc:
            status.update(label="Could not index this repository", state="error")
            st.error(str(exc))
            return
        status.update(label=f"Indexed {report.record.name}", state="complete")

    st.session_state[SELECTED] = report.record.key
    st.session_state[MESSAGES] = []
    st.rerun()


def _render_conversation(selected, llm_config, top_k: int) -> None:
    messages = st.session_state[MESSAGES]
    placeholder = st.empty()
    suggested = None

    with placeholder.container():
        if not messages:
            render_empty_state(
                needs_repository=selected is None,
                needs_key=llm_config is None,
            )
            if selected is not None and llm_config is not None:
                suggested = render_suggestions()

    retry_question = None
    for position, message in enumerate(messages):
        avatar = "🧑‍💻" if message["role"] == "user" else "🔍"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "user":
                st.markdown(message["content"])
            elif message.get("error"):
                render_error(message["error"])
                if st.button("Try again", key=f"retry-{position}"):
                    retry_question = message["question"]
            else:
                render_answer(message)

    ready = selected is not None and llm_config is not None
    placeholder_text = (
        "Ask about the code" if ready else "Add an API key and analyze a repository to start"
    )
    typed = st.chat_input(placeholder_text, disabled=not ready)
    question = typed or suggested or retry_question
    if not question:
        return

    placeholder.empty()
    _answer(question, selected.key, llm_config, top_k, is_retry=retry_question is not None)


def _answer(question: str, repo_key: str, llm_config: LLMConfig, top_k: int, is_retry: bool) -> None:
    if is_retry:
        _drop_failed_attempt(question)

    st.session_state[MESSAGES].append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🔍"):
        try:
            with st.spinner("Searching the code and writing an answer"):
                result = ask(question, repo=repo_key, k=top_k, llm=llm_config)
            message = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "grounding": (result.get("grounding") or {}).get("overall_status"),
            }
            render_answer(message)
        except ModelUnavailableError as exc:
            message = _error_message(_scrub(str(exc), llm_config.api_key), question)
        except LLMUnavailableError:
            message = _error_message(BUSY_MESSAGE, question)
        except Exception as exc:
            message = _error_message(
                f"Something went wrong while answering: {_scrub(str(exc), llm_config.api_key)}",
                question,
            )
        if message.get("error"):
            render_error(message["error"])

    st.session_state[MESSAGES].append(message)
    if message.get("error"):
        st.rerun()


def _drop_failed_attempt(question: str) -> None:
    kept = [m for m in st.session_state[MESSAGES] if m.get("error") is None]
    if kept and kept[-1]["role"] == "user" and kept[-1]["content"] == question:
        kept.pop()
    st.session_state[MESSAGES] = kept


def _error_message(text: str, question: str) -> dict:
    return {"role": "assistant", "error": text, "question": question}


def _scrub(text: str, api_key: str | None) -> str:
    if not api_key:
        return text
    return text.replace(api_key, "••••••")
