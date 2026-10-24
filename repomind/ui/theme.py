"""Design tokens and global styles for the RepoMind interface."""

import streamlit as st

_STYLES = """
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap");

:root {
  --rm-bg: #f5f6fb;
  --rm-surface: #ffffff;
  --rm-surface-raised: #ffffff;
  --rm-surface-muted: #eef1f8;
  --rm-border: #e2e6f0;
  --rm-text: #0b1220;
  --rm-text-muted: #5b6678;
  --rm-accent: #6366f1;
  --rm-accent-strong: #4338ca;
  --rm-accent-2: #0891b2;
  --rm-success: #059669;
  --rm-warning: #d97706;
  --rm-danger: #dc2626;
  --rm-neutral: #64748b;
  --rm-gradient: linear-gradient(135deg, #4338ca 0%, #6366f1 45%, #0891b2 100%);
  --rm-radius: 16px;
  --rm-radius-sm: 10px;
  --rm-shadow: 0 1px 2px rgba(11, 18, 32, 0.04), 0 12px 32px -18px rgba(11, 18, 32, 0.22);
  --rm-font: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --rm-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Consolas, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --rm-bg: #0a0f1d;
    --rm-surface: #111a2d;
    --rm-surface-raised: #15213a;
    --rm-surface-muted: #0f1728;
    --rm-border: #1f2c46;
    --rm-text: #e8edf7;
    --rm-text-muted: #9aa7bd;
    --rm-shadow: 0 1px 2px rgba(0, 0, 0, 0.4), 0 16px 40px -20px rgba(0, 0, 0, 0.75);
  }
}

html, body, .stApp, .stMarkdown, button, input, textarea, select {
  font-family: var(--rm-font) !important;
}
.stApp { background: var(--rm-bg); color: var(--rm-text); }
.block-container { max-width: 980px; padding-top: 2rem; padding-bottom: 8rem; }
#MainMenu, footer, .stDeployButton { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
  background: var(--rm-surface-muted);
  border-right: 1px solid var(--rm-border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.4rem 1.1rem 2rem; }
section[data-testid="stSidebar"] h5 {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--rm-text-muted);
  margin: 0.4rem 0 0.6rem;
}

.rm-brand { display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem; }
.rm-logo {
  width: 36px; height: 36px; border-radius: 11px;
  background: var(--rm-gradient);
  display: grid; place-items: center;
  color: #ffffff; font-weight: 800; font-size: 1rem;
  box-shadow: 0 8px 20px -8px rgba(99, 102, 241, 0.8);
}
.rm-brand-name { font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em; color: var(--rm-text); }
.rm-brand-tag { font-size: 0.76rem; color: var(--rm-text-muted); }

.rm-hero {
  position: relative;
  overflow: hidden;
  border-radius: 24px;
  padding: 34px 38px;
  margin-bottom: 26px;
  color: #ffffff;
  background: var(--rm-gradient);
  box-shadow: 0 24px 60px -24px rgba(67, 56, 202, 0.7);
}
.rm-hero::before, .rm-hero::after {
  content: "";
  position: absolute;
  border-radius: 50%;
  pointer-events: none;
}
.rm-hero::before {
  right: -90px; top: -110px; width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(255, 255, 255, 0.28), transparent 70%);
}
.rm-hero::after {
  left: -60px; bottom: -120px; width: 260px; height: 260px;
  background: radial-gradient(circle, rgba(8, 145, 178, 0.55), transparent 70%);
}
.rm-hero-eyebrow {
  display: inline-block;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
  opacity: 0.85; margin-bottom: 10px;
}
.rm-hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.025em; color: #ffffff; }
.rm-hero p { margin: 0.55rem 0 0; font-size: 1.02rem; line-height: 1.6; opacity: 0.92; max-width: 640px; }
.rm-chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; position: relative; z-index: 1; }
.rm-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 12px; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600;
  background: rgba(255, 255, 255, 0.16);
  border: 1px solid rgba(255, 255, 255, 0.22);
  color: #ffffff;
}

.rm-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 6px 0 18px; }
.rm-step {
  background: var(--rm-surface);
  border: 1px solid var(--rm-border);
  border-radius: var(--rm-radius);
  padding: 16px;
  box-shadow: var(--rm-shadow);
}
.rm-step-index {
  width: 26px; height: 26px; border-radius: 8px;
  display: grid; place-items: center;
  font-size: 0.78rem; font-weight: 800; color: var(--rm-accent-strong);
  background: rgba(99, 102, 241, 0.14);
}
.rm-step-title { margin: 10px 0 4px; font-size: 0.95rem; font-weight: 650; color: var(--rm-text); }
.rm-step-text { margin: 0; font-size: 0.84rem; line-height: 1.5; color: var(--rm-text-muted); }

.rm-empty {
  text-align: center;
  padding: 42px 28px;
  border-radius: 22px;
  border: 1px dashed var(--rm-border);
  background: var(--rm-surface);
  margin-bottom: 14px;
}
.rm-empty-icon {
  width: 56px; height: 56px; margin: 0 auto;
  border-radius: 18px; display: grid; place-items: center;
  font-size: 1.6rem; background: var(--rm-surface-muted);
}
[data-testid="stHeaderActionElements"], .stMarkdown h1 > a, .stMarkdown h2 > a, .stMarkdown h3 > a, .stMarkdown h4 > a {
  display: none !important;
}
.rm-empty h3 { margin: 14px 0 6px; font-size: 1.2rem; font-weight: 700; color: var(--rm-text); }
.rm-empty p { margin: 0 auto; max-width: 460px; color: var(--rm-text-muted); font-size: 0.95rem; line-height: 1.55; }

.rm-card {
  background: var(--rm-surface);
  border: 1px solid var(--rm-border);
  border-radius: var(--rm-radius);
  padding: 16px 18px;
  box-shadow: var(--rm-shadow);
  margin-bottom: 14px;
}
.rm-card-title { margin: 0; font-size: 0.98rem; font-weight: 700; color: var(--rm-text); word-break: break-all; }
.rm-card-subtitle { margin: 3px 0 0; font-size: 0.8rem; color: var(--rm-text-muted); word-break: break-all; }
.rm-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 14px; }
.rm-stat { border-radius: var(--rm-radius-sm); padding: 10px 6px; text-align: center; background: var(--rm-surface-muted); }
.rm-stat-value { display: block; font-size: 1.1rem; font-weight: 800; color: var(--rm-accent-strong); }
.rm-stat-label { display: block; font-size: 0.66rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: var(--rm-text-muted); }
.rm-languages { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.rm-tag {
  font-size: 0.72rem; font-weight: 600; padding: 3px 9px; border-radius: 999px;
  background: var(--rm-surface-muted); color: var(--rm-text-muted); border: 1px solid var(--rm-border);
}

.rm-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 11px; border-radius: 999px;
  font-size: 0.74rem; font-weight: 650; color: #ffffff; margin: 8px 6px 0 0;
}
.rm-badge-soft { background: var(--rm-surface-muted); color: var(--rm-text-muted); border: 1px solid var(--rm-border); }
.rm-dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; opacity: 0.95; }

.rm-source {
  display: flex; justify-content: space-between; align-items: center; gap: 12px;
  padding: 10px 12px; margin-bottom: 8px;
  border: 1px solid var(--rm-border); border-radius: 12px;
  background: var(--rm-surface-muted);
}
.rm-source-path { font-family: var(--rm-mono); font-size: 0.82rem; color: var(--rm-text); word-break: break-all; }
.rm-source-lines {
  font-family: var(--rm-mono); font-size: 0.72rem; font-weight: 500;
  padding: 3px 9px; border-radius: 8px; white-space: nowrap;
  color: var(--rm-accent-2); background: rgba(8, 145, 178, 0.14);
}

.stButton > button {
  width: 100%;
  border-radius: 12px;
  border: 1px solid var(--rm-border);
  background: var(--rm-surface);
  color: var(--rm-text);
  font-weight: 600;
  padding: 0.6rem 0.9rem;
  box-shadow: var(--rm-shadow);
  transition: transform 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}
.stButton > button:hover { border-color: var(--rm-accent); color: var(--rm-accent-strong); transform: translateY(-1px); }
.stButton > button:focus:not(:active) { border-color: var(--rm-accent); box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25); }
.stButton > button[kind="primary"] { background: var(--rm-gradient); border: none; color: #ffffff; }
.stButton > button[kind="primary"]:hover { color: #ffffff; filter: brightness(1.07); }
.stButton > button:disabled { opacity: 0.5; }

div[data-testid="stChatMessage"] {
  background: var(--rm-surface);
  border: 1px solid var(--rm-border);
  border-radius: var(--rm-radius);
  padding: 16px 18px;
  box-shadow: var(--rm-shadow);
  margin-bottom: 14px;
}
div[data-testid="stChatInput"] { border-radius: 18px; box-shadow: var(--rm-shadow); }
div[data-testid="stChatInput"] textarea { font-size: 0.97rem; }
div[data-testid="stExpander"] { border-radius: 14px; border: 1px solid var(--rm-border); background: var(--rm-surface); }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-radius: var(--rm-radius-sm) !important; }
div[data-testid="stAlert"] { border-radius: var(--rm-radius-sm); }

@media (max-width: 760px) {
  .rm-steps { grid-template-columns: 1fr; }
  .rm-hero { padding: 26px 24px; }
  .rm-hero h1 { font-size: 1.9rem; }
}
"""


def apply_theme() -> None:
    st.markdown(f"<style>{_STYLES}</style>", unsafe_allow_html=True)
