"""
app.py — Multi-Agent Research Assistant  (entry point)
=======================================================
This file is intentionally thin.  All logic lives in src/:

  src/ui/styles/shared.css   ← fonts, pills, bento-grid, nav-button shape
  src/ui/styles/dark.css     ← dark-mode colour tokens  (edit to change dark theme)
  src/ui/styles/light.css    ← light-mode colour tokens (edit to change light theme)
  src/ui/theme.py            ← CSS loader
  src/ui/sidebar.py          ← sidebar navigation
  src/ui/pipeline.py         ← Full Pipeline UI (9 agents)
  src/ui/playground.py       ← Agent Playground UI (single agents)
  src/ui/components.py       ← shared widgets (file upload, entity table)
  src/agents/                ← agent implementations
  src/tools/                 ← MCP tool registry
  src/graph.py               ← LangGraph StateGraph

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config  (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme injection ───────────────────────────────────────────────────────────
# Default to Dark so crashes / reloads don't flash white at users.
from src.ui.theme import inject_theme_css  # noqa: E402

if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "Dark"
inject_theme_css(st.session_state["app_theme"])

# ── UI modules ────────────────────────────────────────────────────────────────
from src.ui.sidebar   import render_sidebar          # noqa: E402
from src.ui.pipeline  import render_full_pipeline_ui # noqa: E402
from src.ui.playground import render_playground_ui   # noqa: E402


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Render header → sidebar → selected mode UI."""
    _is_dark  = st.session_state.get("app_theme", "Dark") == "Dark"
    _primary  = "#92ccff" if _is_dark else "#006397"
    _subtitle = "#bfc7d2" if _is_dark else "#64748b"

    st.markdown(
        f"""<div style="margin-bottom:8px;">
          <h1 style="font-size:1.6rem;font-weight:700;letter-spacing:-0.02em;
                     color:{_primary};margin:0 0 4px 0;">
            🔬 Multi-Agent Research Assistant
          </h1>
          <p style="font-size:0.875rem;color:{_subtitle};margin:0;line-height:1.6;">
            <b>Full Pipeline</b> — runs all 9 LangGraph agents and produces a
            complete cited report.&nbsp;
            <b>Agent Playground</b> — run any single agent in isolation.&nbsp;
            Upload a document to analyse your own files without web search.
          </p>
        </div>""",
        unsafe_allow_html=True,
    )

    selected_mode = render_sidebar()
    st.divider()

    if selected_mode == "Full Pipeline":
        render_full_pipeline_ui()
    else:
        render_playground_ui(selected_mode)


main()
