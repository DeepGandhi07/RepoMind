"""Rendering helpers for the RepoMind interface. Every value from the user or the model is escaped."""

import html

import streamlit as st

STATUS_STYLES = {
    "supported": ("Supported", "#059669"),
    "partially_supported": ("Partially supported", "#d97706"),
    "unsupported": ("Unsupported", "#dc2626"),
    "insufficient_evidence": ("Insufficient evidence", "#64748b"),
    "unavailable": ("Check unavailable", "#64748b"),
    "skipped": ("Check off", "#64748b"),
}

SUGGESTED_QUESTIONS = [
    "What does this project do?",
    "Explain the overall architecture.",
    "How does error handling work?",
    "Where is configuration loaded from?",
    "What are the main entry points?",
    "How would I add a feature safely?",
]


def status_label(status: str | None) -> tuple[str, str]:
    return STATUS_STYLES.get(status or "unavailable", STATUS_STYLES["unavailable"])


def source_location(source: dict) -> str:
    location = source["file"]
    if "start_line" in source and "end_line" in source:
        location += f" · lines {source['start_line']}–{source['end_line']}"
    return location


def render_hero(provider_label: str, model: str | None) -> None:
    model_chip = f"{provider_label} · {model}" if model else "Bring your own API key"
    chips = [model_chip, "Any language", "Grounded answers"]
    chip_html = "".join(
        f'<span class="rm-chip">{html.escape(chip)}</span>' for chip in chips
    )
    st.markdown(
        '<section class="rm-hero">'
        '<span class="rm-hero-eyebrow">AI code assistant</span>'
        "<h1>Ask any codebase anything</h1>"
        "<p>RepoMind reads a repository, finds the code that matters, and answers in plain language. "
        "Every claim points to the file and lines it came from.</p>"
        f'<div class="rm-chips">{chip_html}</div>'
        "</section>",
        unsafe_allow_html=True,
    )


def render_how_it_works() -> None:
    steps = [
        (
            "1",
            "Index",
            "Clone the repository and split its files into searchable chunks.",
        ),
        (
            "2",
            "Retrieve",
            "Find the chunks that best match your question using meaning and keywords.",
        ),
        (
            "3",
            "Answer",
            "A model writes a reply from that code only, with a citation for each claim.",
        ),
    ]
    cards = "".join(
        '<div class="rm-step">'
        f'<span class="rm-step-index">{index}</span>'
        f'<p class="rm-step-title">{html.escape(title)}</p>'
        f'<p class="rm-step-text">{html.escape(text)}</p>'
        "</div>"
        for index, title, text in steps
    )
    st.markdown(f'<div class="rm-steps">{cards}</div>', unsafe_allow_html=True)


def render_repo_summary(record) -> None:
    languages = "".join(
        f'<span class="rm-tag">{html.escape(name)}</span>'
        for name in list(record.languages)[:6]
    )
    st.markdown(
        '<div class="rm-card">'
        f'<p class="rm-card-title">{html.escape(record.name)}</p>'
        f'<p class="rm-card-subtitle">{html.escape(record.source)}</p>'
        '<div class="rm-stats">'
        f'<div class="rm-stat"><span class="rm-stat-value">{record.files}</span>'
        '<span class="rm-stat-label">files</span></div>'
        f'<div class="rm-stat"><span class="rm-stat-value">{record.chunks}</span>'
        '<span class="rm-stat-label">chunks</span></div>'
        f'<div class="rm-stat"><span class="rm-stat-value">{len(record.languages)}</span>'
        '<span class="rm-stat-label">languages</span></div>'
        "</div>"
        f'<div class="rm-languages">{languages}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_empty_state(needs_repository: bool, needs_key: bool) -> None:
    if needs_repository:
        icon, title, text = (
            "📦",
            "Analyze a repository",
            "Paste an https URL in the sidebar. Public and private repositories both work.",
        )
    elif needs_key:
        icon, title, text = (
            "🔑",
            "Add your API key",
            "Enter a key for Gemini or Groq in the sidebar. Indexing works without one; answers need a model.",
        )
    else:
        icon, title, text = (
            "💬",
            "Ask anything about the code",
            "Answers cite file paths and line ranges, and a separate check shows whether the code supports them.",
        )

    st.markdown(
        f'<div class="rm-empty"><div class="rm-empty-icon">{icon}</div>'
        f"<h3>{html.escape(title)}</h3><p>{html.escape(text)}</p></div>",
        unsafe_allow_html=True,
    )


def render_suggestions(key_prefix: str = "suggest") -> str | None:
    """Render a responsive grid of suggested questions."""
    icons = ["📦", "🏗️", "🛡️", "⚙️", "🚪", "🧩"]

    st.markdown(
        """
        <style>
        /* Suggested-question buttons */
        div[class*="st-key-suggest-"] {
            width: 100%;
        }

        div[class*="st-key-suggest-"] button {
            width: 100%;
            min-height: 64px;
            height: auto;
            padding: 15px 17px;
            border: 1px solid #29364c;
            border-radius: 12px;
            background: #111a2b;
            color: #e5edf9;
            box-shadow: none;
            transition: background 150ms ease,
                        border-color 150ms ease,
                        transform 150ms ease;
        }

        div[class*="st-key-suggest-"] button:hover {
            background: #19263b;
            border-color: #6387bd;
            color: #ffffff;
            transform: translateY(-1px);
        }

        div[class*="st-key-suggest-"] button:focus-visible {
            outline: 2px solid #80aaff;
            outline-offset: 3px;
        }

        div[class*="st-key-suggest-"] button p {
            font-size: 14px;
            font-weight: 500;
            line-height: 1.5;
            text-align: left;
        }

        /* Consistent spacing between rows */
        div[data-testid="stHorizontalBlock"] {
            gap: 16px;
        }

        @media (max-width: 640px) {
            div[class*="st-key-suggest-"] button {
                min-height: 58px;
                padding: 12px;
            }

            div[class*="st-key-suggest-"] button p {
                font-size: 13px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    clicked = None

    # Row-major ordering: 1 and 2 on row one, 3 and 4 on row two.
    for start in range(0, len(SUGGESTED_QUESTIONS), 2):
        columns = st.columns(2, gap="medium")

        for offset, column in enumerate(columns):
            position = start + offset

            if position >= len(SUGGESTED_QUESTIONS):
                continue

            question = SUGGESTED_QUESTIONS[position]
            label = f"{icons[position]}  {question}"

            with column:
                if st.button(
                    label,
                    key=f"{key_prefix}-{position}",
                    use_container_width=True,
                    type="secondary",
                ):
                    clicked = question

    return clicked


def render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No source files were retrieved.")
        return
    cards = []
    for source in sources:
        lines = ""
        if "start_line" in source and "end_line" in source:
            lines = f'<span class="rm-source-lines">lines {source["start_line"]}–{source["end_line"]}</span>'
        cards.append(
            f'<div class="rm-source"><span class="rm-source-path">{html.escape(source["file"])}</span>{lines}</div>'
        )
    st.markdown("".join(cards), unsafe_allow_html=True)


def render_answer(message: dict) -> None:
    st.markdown(message["content"])
    label, color = status_label(message.get("grounding"))
    count = len(message.get("sources", []))
    st.markdown(
        f'<span class="rm-badge" style="background:{color}"><span class="rm-dot"></span>'
        f"{html.escape(label)}</span>"
        f'<span class="rm-badge rm-badge-soft">{count} code sources</span>',
        unsafe_allow_html=True,
    )
    with st.expander("View the code behind this answer"):
        render_sources(message.get("sources", []))


def render_error(text: str) -> None:
    st.warning(text, icon="⚠️")
