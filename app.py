"""
app.py — Multi-Agent Research Assistant: Streamlit UI.
Powered by L2 team.

TWO OPERATING MODES
===================
Full Pipeline  (default)
    Runs all 9 agents in sequence via LangGraph.  Each agent's output is
    streamed in real-time to expandable panels.  Produces a complete Markdown
    + PDF report.  This is the mandatory mode required by the course project.

Agent Playground
    Runs a single selected agent in isolation.  Dependency resolution is
    handled automatically by src/router.py — the user only sees the output of
    the agent they chose.

FILE UPLOAD
===========
Users can upload PDF, DOCX, TXT, or CSV files.  The file content is extracted
and injected into any agent as if it were a scraped web source.  This allows
running NER, classification, analysis, or writing directly on uploaded documents
without any web search.

RUN:
    streamlit run app.py
"""

from __future__ import annotations

import dataclasses
import logging
from collections import Counter
from pathlib import Path

import streamlit as st

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme (Dark / Light) ──────────────────────────────────────────────────────
# Component CSS that is shared by both themes (pills, cards, etc.).  Colours that
# differ between themes are filled in by _inject_theme_css() below.
_SHARED_CSS = """
/* ── Inter font (Stitch design system) ──────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

.stApp, .stApp * { font-family: 'Inter', system-ui, -apple-system, sans-serif !important; }

/* Thin scrollbar — matches Stitch dark theme */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3f4850; border-radius: 10px; }

/* Progress bar — Stitch primary (#3498db) */
.stProgress .st-bo { background-color: #3498db; }

/* ── Mode badge pills ────────────────────────────────────────────────────── */
.mode-pill {
    display:inline-block; padding:4px 12px; border-radius:12px;
    font-size:0.75rem; font-weight:600; letter-spacing:0.04em;
    margin-bottom:8px; text-transform:uppercase;
}
.stApp .mode-full       { background:#0f3320 !important; color:#61de8a !important;
                           border:1px solid rgba(97,222,138,0.35) !important; }
.stApp .mode-playground { background:#0d2240 !important; color:#92ccff !important;
                           border:1px solid rgba(146,204,255,0.35) !important; }
.stApp .file-pill {
    display:inline-block; padding:4px 12px; border-radius:12px;
    font-size:0.75rem; font-weight:600; margin-bottom:8px;
    background:#3d2a00 !important; color:#ffba4b !important;
    border:1px solid rgba(255,186,75,0.35) !important;
}

/* ── Score colours ──────────────────────────────────────────────────────── */
.stApp .score-high { color:#61de8a !important; font-weight:700; font-size:1.4rem; }
.stApp .score-low  { color:#ffb4ab !important; font-weight:700; font-size:1.4rem; }

/* ── Bento-grid (metric cards) ──────────────────────────────────────────── */
.bento-grid {
    display:grid;
    grid-template-columns:repeat(auto-fill, minmax(120px, 1fr));
    gap:8px; margin:12px 0;
}
.bento-card {
    border-radius:12px; padding:14px 12px; text-align:center;
    border:1px solid rgba(63,72,80,0.8);
    backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
}
.bento-card .bc-value {
    font-size:1.6rem; font-weight:700; line-height:1.2;
    font-family:'Inter',sans-serif;
}
.bento-card .bc-label {
    font-size:0.70rem; font-weight:500; letter-spacing:0.05em;
    text-transform:uppercase; margin-top:4px; opacity:0.75;
}

/* ── Glass card (general purpose) ──────────────────────────────────────── */
.glass-card {
    border-radius:12px; padding:16px;
    border:1px solid rgba(255,255,255,0.06);
    backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px);
}

/* ── Nav item (sidebar) ─────────────────────────────────────────────────── */
.nav-item {
    display:flex; align-items:center; gap:10px; padding:8px 12px;
    border-radius:8px; font-size:0.88rem; font-weight:500;
    margin-bottom:2px; cursor:pointer; transition:background 0.15s;
}
.nav-item.active {
    border-right:2px solid #92ccff; /* Stitch active indicator */
}
"""

# Per-theme colour tokens.  We override Streamlit's main containers so the whole
# page (not just our custom elements) switches between light and dark.
_LIGHT_CSS = """
/* ── Stitch light tokens ─────────────────────────────────────────────────── */
.stApp { background-color: #f8fafc; color: #1a1a1a; }
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background-color: #f8fafc !important; }
[data-testid="stHeader"] { background-color: #f8fafc; border-bottom:1px solid #e2e8f0; }
section[data-testid="stSidebar"] {
    background-color: #f0f4f8 !important;
    border-right:1px solid #e2e8f0 !important;
}

/* Active nav in light mode */
.nav-item.active { color:#006397; background:#e8f4ff; border-right:2px solid #006397; }
.nav-item { color:#475569; }
.nav-item:hover { background:#e2eaf4; }

.metric-card {
    background:#ffffff; border-radius:12px; padding:14px;
    border:1px solid #e2e8f0; border-left:4px solid #006397;
    margin:4px 0; color:#1a1a1a;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
.bento-card {
    background:#ffffff; color:#1a1a1a;
    box-shadow:0 1px 4px rgba(0,0,0,0.08);
}
.bento-card .bc-value { color:#006397; }
.bento-card .bc-label { color:#64748b; }
.glass-card { background:#ffffff; border-color:#e2e8f0; }
.entity-tag {
    display:inline-block; padding:2px 8px; border-radius:10px;
    background:#e8f4fd; color:#1a5276; font-size:0.82rem; margin:2px;
}
.file-info-card {
    background:#fffdf0; border:1px solid #ffc107; border-radius:8px;
    padding:10px 14px; margin:8px 0; font-size:0.88rem; color:#1a1a1a;
}
.l2-footer {
    text-align:center; padding:12px; color:#64748b;
    font-size:0.80rem; border-top:1px solid #e2e8f0; margin-top:24px;
}

/* Light mode: restore pill backgrounds */
.stApp .mode-full       { background:#dcfce7 !important; color:#166534 !important;
                           border:1px solid rgba(22,101,52,0.3) !important; }
.stApp .mode-playground { background:#dbeafe !important; color:#1e40af !important;
                           border:1px solid rgba(30,64,175,0.3) !important; }
.stApp .file-pill { background:#fef3c7 !important; color:#92400e !important;
                    border:1px solid rgba(146,64,14,0.3) !important; }
.stApp .score-high { color:#15803d !important; }
.stApp .score-low  { color:#dc2626 !important; }
"""

_DARK_CSS = """
/* ── Stitch dark tokens ──────────────────────────────────────────────────── */
/* surface-dim=#101418  container=#1c2024  container-high=#262a2f
   container-highest=#31353a  container-lowest=#0a0f13
   outline-variant=#3f4850  outline=#89929b
   on-surface=#e0e3e8  on-surface-variant=#bfc7d2
   primary=#92ccff  primary-container=#3498db
   secondary=#61de8a  tertiary=#ffba4b                                        */

/* Main containers */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background-color: #101418 !important; color: #e0e3e8 !important; }
[data-testid="stHeader"] { background-color: #101418 !important;
                            border-bottom:1px solid #3f4850 !important; }
[data-testid="stToolbar"] { color: #e0e3e8 !important; }

/* Sidebar — surface-container-lowest */
section[data-testid="stSidebar"] {
    background-color: #0a0f13 !important;
    border-right:1px solid #3f4850 !important;
}

/* Nav item colours for dark mode */
.nav-item { color: #bfc7d2; }
.nav-item.active { color:#92ccff; background:#262a2f; border-right:2px solid #92ccff; }
.nav-item:hover { background:#1c2024; }

/* Body text */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #e0e3e8 !important;
}
.stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricLabel"] { color: #e0e3e8 !important; }

/* Caption / muted text */
.stApp [data-testid="stCaptionContainer"] { color: #bfc7d2 !important; }

/* Inputs, text areas, selectboxes */
.stApp textarea, .stApp input,
.stApp .stTextInput input, .stApp .stTextArea textarea,
.stApp .stSelectbox div[data-baseweb="select"] > div {
    background-color: #1c2024 !important; color: #e0e3e8 !important;
    border-color: #3f4850 !important;
}

/* Secondary buttons */
.stApp .stButton button[kind="secondary"],
.stApp .stDownloadButton button {
    background-color: #1c2024 !important; color: #e0e3e8 !important;
    border: 1px solid #3f4850 !important;
}

/* ── Expanders (Stitch surface-container) ─────────────────────────────── */
.stApp [data-testid="stExpander"] {
    background-color: #1c2024 !important;
    border: 1px solid #3f4850 !important;
    border-radius: 12px !important;
}
.stApp [data-testid="stExpander"] summary { color: #e0e3e8 !important; }
.stApp [data-testid="stExpander"] summary svg { fill: #e0e3e8 !important; }

/* Nuclear fix — force dark background into all open expander content */
.stApp details[open] > div,
.stApp details[open] > div *:not(button):not(.mode-full):not(.mode-playground):not(.file-pill):not(.score-high):not(.score-low) {
    background-color: #1c2024 !important;
    color: #e0e3e8 !important;
    border-color: #3f4850 !important;
}

/* Restore pill / badge colours */
.stApp details[open] .mode-full       { background:#0f3320 !important; color:#61de8a !important;
                                        border:1px solid rgba(97,222,138,0.35) !important; }
.stApp details[open] .mode-playground { background:#0d2240 !important; color:#92ccff !important;
                                        border:1px solid rgba(146,204,255,0.35) !important; }
.stApp details[open] .file-pill       { background:#3d2a00 !important; color:#ffba4b !important;
                                        border:1px solid rgba(255,186,75,0.35) !important; }
.stApp details[open] .score-high      { color:#61de8a !important; }
.stApp details[open] .score-low       { color:#ffb4ab !important; }

/* Restore file-uploader dropzone */
.stApp details[open] [data-testid="stFileUploadDropzone"] {
    background-color: #1c2024 !important;
    border: 2px dashed #3f4850 !important;
}

/* Restore primary buttons inside expanders */
.stApp details[open] button[kind="primary"],
.stApp details[open] .stButton > button[data-testid="stBaseButton-primary"] {
    background-color: #3498db !important; color: #ffffff !important;
    border-color: #3498db !important;
}
/* Restore secondary buttons inside expanders */
.stApp details[open] button[kind="secondary"],
.stApp details[open] .stButton > button[data-testid="stBaseButton-secondary"] {
    background-color: #1c2024 !important; color: #e0e3e8 !important;
    border-color: #3f4850 !important;
}

/* Dataframes / tables */
.stApp [data-testid="stTable"], .stApp [data-testid="stDataFrame"] {
    background-color: #1c2024 !important; color: #e0e3e8 !important;
}

/* ── Custom component colours (Stitch tokens) ────────────────────────── */
.metric-card {
    background: #1c2024; border-radius: 12px; padding: 14px;
    border: 1px solid #3f4850; border-left: 4px solid #3498db;
    margin: 4px 0; color: #e0e3e8;
}
.bento-card {
    background: rgba(28,32,36,0.9); color: #e0e3e8;
    border-color: #3f4850;
}
.bento-card .bc-value { color: #92ccff; }
.bento-card .bc-label { color: #bfc7d2; }
.glass-card {
    background: rgba(28,32,36,0.85); border-color: rgba(255,255,255,0.06);
}
.entity-tag {
    display:inline-block; padding:2px 8px; border-radius:10px;
    background:#1a2d45; color:#92ccff; font-size:0.82rem; margin:2px;
}
.file-info-card {
    background: #2a2615; border: 1px solid #ffba4b; border-radius: 12px;
    padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; color: #f0e6c0;
}
.l2-footer {
    text-align:center; padding:12px; color:#bfc7d2;
    font-size:0.80rem; border-top:1px solid #3f4850; margin-top:24px;
}

/* ── Form containers ─────────────────────────────────────────────────── */
.stApp [data-testid="stForm"] {
    background-color: #1c2024 !important;
    border: 1px solid #3f4850 !important;
    border-radius: 12px !important;
}

/* ── File upload widget ──────────────────────────────────────────────── */
.stApp [data-testid="stFileUploader"] {
    background-color: #1c2024 !important; border-radius: 12px !important;
}
.stApp [data-testid="stFileUploader"] > div,
.stApp [data-testid="stFileUploader"] > div > div {
    background-color: #1c2024 !important; border-radius: 12px !important;
}
.stApp [data-testid="stFileUploadDropzone"],
.stApp [data-testid="stFileUploaderDropzone"],
.stApp section[data-testid="stFileUploadDropzone"] {
    background-color: #1c2024 !important;
    border: 2px dashed #3f4850 !important;
    border-radius: 12px !important;
}
.stApp [data-testid="stFileUploadDropzone"] *,
.stApp [data-testid="stFileUploaderDropzone"] * { color: #bfc7d2 !important; }
.stApp [data-testid="stFileUploaderFileName"] { color: #e0e3e8 !important; }
.stApp button[data-testid="stBaseButton-minimal"] { color: #bfc7d2 !important; }

/* ── Alert boxes ────────────────────────────────────────────────────── */
.stApp [data-testid="stAlertContainer"] { background-color: #1c2024 !important; }
.stApp [data-testid="stAlertContainer"] p,
.stApp [data-testid="stAlertContainer"] div { color: #e0e3e8 !important; }

/* ── Divider ─────────────────────────────────────────────────────────── */
.stApp hr { border-color: #3f4850 !important; }

/* ── Dropdown (selectbox) ────────────────────────────────────────────── */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #1c2024 !important;
    border: 1px solid #3f4850 !important;
}
[data-baseweb="popover"] [data-baseweb="menu"] li { color: #e0e3e8 !important; }
[data-baseweb="popover"] [data-baseweb="menu"] li:hover {
    background-color: #262a2f !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────── */
.stApp [data-testid="stSpinner"] p { color: #e0e3e8 !important; }

/* ── Small / caption text ────────────────────────────────────────────── */
.stApp small { color: #bfc7d2 !important; }

/* ── Selectbox value text ────────────────────────────────────────────── */
.stApp .stSelectbox [data-baseweb="select"] [data-baseweb="value"] {
    color: #e0e3e8 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────── */
.stApp [data-baseweb="tab"] { color: #bfc7d2 !important; }
.stApp [data-baseweb="tab"][aria-selected="true"] { color: #92ccff !important; }
.stApp [data-baseweb="tab-list"] { background-color: #101418 !important; }

/* ── Progress bar track ──────────────────────────────────────────────── */
.stApp [data-testid="stProgressBar"] > div { background-color: #3f4850 !important; }

/* ── Primary button glow on dark ─────────────────────────────────────── */
.stApp .stButton > button[data-testid="stBaseButton-primary"],
.stApp .stButton button[kind="primary"] {
    background-color: #3498db !important;
    border-color: #3498db !important;
    color: #ffffff !important;
    box-shadow: 0 0 12px rgba(52,152,219,0.25) !important;
}
"""


def _inject_theme_css(theme: str) -> None:
    """
    Inject the CSS for the chosen theme ("Light" or "Dark").

    Called once near the top of every script run.  The chosen theme is read
    from st.session_state["app_theme"], which is set by the sidebar toggle and
    persists across reruns — so changing the toggle re-runs the script and this
    function re-injects the matching colours.
    """
    palette = _DARK_CSS if theme == "Dark" else _LIGHT_CSS
    st.markdown(f"<style>{_SHARED_CSS}\n{palette}</style>", unsafe_allow_html=True)


# Read the persisted theme.  We default to "Dark" so that if the session is
# reset by a crash or page reload, the app comes back in dark mode rather than
# suddenly flashing white — which confused users who keep dark mode selected.
if "app_theme" not in st.session_state:
    st.session_state["app_theme"] = "Dark"
_active_theme = st.session_state["app_theme"]
_inject_theme_css(_active_theme)


# ─────────────────────────────────────────────────────────────────────────────
#  Imports (after page config)
# ─────────────────────────────────────────────────────────────────────────────
from src.config import config
from src.router import (
    ALL_MODES, PLAYGROUND_MODES, MODE_DESCRIPTIONS, MODE_INPUTS, route,
)
from src.utils.error_handler import check_api_keys
from src.utils.cost_tracker import format_cost_table
from src.utils.file_processor import (
    extract_text, get_file_info, SUPPORTED_EXTENSIONS, SUPPORTED_EXTENSIONS_DISPLAY,
)
from src.graph import (
    build_graph, _dict_to_state, _state_to_full_dict, get_graph_config,
)
from src.state import ResearchState


@st.cache_resource(show_spinner=False)
def _get_pipeline_graph():
    """
    Build the LangGraph StateGraph ONCE and cache it across all reruns.

    Why this matters:
    - build_graph() constructs + compiles a StateGraph which is CPU-intensive.
    - Without caching, it runs on EVERY button click.
    - The compiled graph is fully stateless (state is passed in at run time),
      so sharing it across runs is safe.
    - Caching also prevents the second run from hitting a memory spike that
      can crash the Railway container (OOM kill → page reload → apparent crash).
    """
    logger.info("Building LangGraph pipeline (first run or cache miss)…")
    return build_graph()


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """
    Render the sidebar (Stitch design) and return the selected mode string.

    Uses Material Symbols icons and the Stitch nav-item style with active-state
    left border indicator.  Falls back gracefully if the CDN is unavailable.
    """
    # Icon map for each mode
    _MODE_ICONS: dict[str, str] = {
        "Full Pipeline":       "hub",
        "Research Only":       "search",
        "Classification Only": "category",
        "NER Only":            "label",
        "Browser Only":        "language",
        "Analysis Only":       "analytics",
        "Writer Only":         "edit_note",
        "Critic Only":         "rate_review",
        "Illustration Only":   "image",
    }

    with st.sidebar:
        # ── Brand header ─────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="padding:20px 4px 16px 4px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <span class="material-symbols-outlined"
                          style="font-size:28px;color:#92ccff;">
                        neurology
                    </span>
                    <div>
                        <div style="font-size:1rem;font-weight:700;
                                    letter-spacing:-0.01em;color:#92ccff;">
                            Research Assistant
                        </div>
                        <div style="font-size:0.70rem;color:#bfc7d2;
                                    letter-spacing:0.04em;text-transform:uppercase;">
                            Powered by L2 Team
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Theme toggle ──────────────────────────────────────────────────────
        st.radio(
            "🌓 Theme",
            options=["Light", "Dark"],
            key="app_theme",
            horizontal=True,
            help="Switch between light and dark appearance.",
        )
        st.divider()

        # ── API key warnings ──────────────────────────────────────────────────
        for w in check_api_keys():
            st.warning(w, icon="⚠️")

        # ── Mode selector ─────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#89929b;padding:0 4px 8px 4px;'>"
            "Navigation</div>",
            unsafe_allow_html=True,
        )

        selected_mode = st.selectbox(
            "Choose an agent",
            ALL_MODES,
            index=0,
            key="mode_selector",
            label_visibility="collapsed",
        )

        # Render nav items as Stitch-styled rows
        for mode in ALL_MODES:
            is_active = mode == selected_mode
            icon = _MODE_ICONS.get(mode, "chevron_right")
            active_cls  = "active" if is_active else ""
            section_lbl = "Full Pipeline" if mode == "Full Pipeline" else mode
            st.markdown(
                f"""<div class="nav-item {active_cls}">
                    <span class="material-symbols-outlined"
                          style="font-size:18px;flex-shrink:0;">{icon}</span>
                    <span style="font-size:0.875rem;">{section_lbl}</span>
                </div>""",
                unsafe_allow_html=True,
            )

        # ── Mode badge + description ──────────────────────────────────────────
        st.divider()
        if selected_mode == "Full Pipeline":
            st.markdown(
                '<span class="mode-pill mode-full">📋 Competition Requirement</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="mode-pill mode-playground">🧪 Agent Playground</span>',
                unsafe_allow_html=True,
            )
        st.info(MODE_DESCRIPTIONS.get(selected_mode, ""), icon="ℹ️")

        # ── System config ─────────────────────────────────────────────────────
        st.divider()
        st.markdown(
            "<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.08em;"
            "text-transform:uppercase;color:#89929b;padding:0 4px 8px 4px;'>"
            "Configuration</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:0.82rem;line-height:1.9;'>"
            f"<b>Model:</b> <code>{config.anthropic_model}</code><br>"
            f"<b>Min sources:</b> {config.min_sources}<br>"
            f"<b>Max revisions:</b> {config.max_revisions}<br>"
            f"<b>Critic pass:</b> {config.critic_pass_score}/10</div>",
            unsafe_allow_html=True,
        )

        # ── MCP tools list ────────────────────────────────────────────────────
        with st.expander("🔧 MCP Tools"):
            from src.tools.mcp_tools import list_tools  # noqa: PLC0415
            for tool in list_tools():
                st.markdown(f"- **{tool['name']}**")

        # ── Footer ────────────────────────────────────────────────────────────
        st.markdown(
            "<div class='l2-footer'>Multi-Agent Research Assistant<br>"
            "<span style='opacity:0.6;font-size:0.72rem;'>L2 Team · 2025</span></div>",
            unsafe_allow_html=True,
        )

    return selected_mode


# ─────────────────────────────────────────────────────────────────────────────
#  Shared inline file upload widget
# ─────────────────────────────────────────────────────────────────────────────

def _render_file_upload(prefix: str) -> None:
    """
    Render an inline file uploader in the main content area.

    Stores extracted text in st.session_state["uploaded_file_text"] so every
    agent — Full Pipeline and all Playground modes — can read it from the same
    place.

    Args:
        prefix: Short namespace string to avoid widget key collisions between
                different render contexts ("fp" for Full Pipeline, "pg_ner_only"
                for NER Playground, etc.).
    """
    with st.expander("📁 Include a document (optional)", expanded=False):
        st.caption(f"Supported: {SUPPORTED_EXTENSIONS_DISPLAY} · Max 200 MB")

        uploaded = st.file_uploader(
            "Upload document",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            key=f"file_upload_{prefix}",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            # Re-extract only when the filename changes — avoids re-parsing on
            # every Streamlit rerun (which happens on any widget interaction).
            if uploaded.name != st.session_state.get("uploaded_file_name"):
                with st.spinner("Reading file…"):
                    extracted = extract_text(uploaded.read(), uploaded.name)
                st.session_state["uploaded_file_text"] = extracted
                st.session_state["uploaded_file_name"] = uploaded.name

            chars = len(st.session_state.get("uploaded_file_text", ""))
            fn    = st.session_state.get("uploaded_file_name", uploaded.name)
            st.success(f"✅ **{fn}** — {chars:,} characters extracted")

        else:
            # No file selected in this widget.  If a file was loaded in a
            # previous mode, keep showing it with a Clear button so the user
            # can intentionally remove it.
            if st.session_state.get("uploaded_file_text"):
                fn = st.session_state.get("uploaded_file_name", "file")
                st.markdown(
                    f'<span class="file-pill">📄 Using: {fn}</span>',
                    unsafe_allow_html=True,
                )
                if st.button("🗑️ Remove file", key=f"clear_file_{prefix}"):
                    st.session_state.pop("uploaded_file_text", None)
                    st.session_state.pop("uploaded_file_name", None)
                    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Full Pipeline UI  (Mode 1)
# ─────────────────────────────────────────────────────────────────────────────

def _set_demo_topic() -> None:
    """
    Callback for the "Demo Topic" button.

    Runs at the START of the next script run — before the fp_topic text_input
    is instantiated — so it is allowed to set st.session_state.fp_topic.
    """
    st.session_state.fp_topic = "Mixture of Experts in Large Language Models"


def render_full_pipeline_ui() -> None:
    """
    Render the Full Pipeline interface.

    Shows a topic input, a Run button, and a real-time streaming view of each
    agent's output as the LangGraph pipeline executes.

    This is the mandatory project mode — all 9 agents run sequentially.
    """
    st.markdown(
        '<span class="mode-pill mode-full">📋 Full Pipeline — Competition Mode</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "All **9 agents** execute in sequence via LangGraph.  "
        "A complete cited Markdown report is produced at the end."
    )

    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Transformer Architecture and Mixture of Experts",
        key="fp_topic",
    )

    # ── Inline file upload (below topic) ──────────────────────────────────────
    # Users can give a topic, upload a document, or both.
    # The uploaded file is injected as a high-relevance source into the pipeline.
    _render_file_upload(prefix="fp")

    col1, col2 = st.columns([3, 1])
    with col1:
        run_btn = st.button(
            "🚀 Generate Full Report", type="primary",
            use_container_width=True, key="fp_run",
        )
    with col2:
        # Use an on_click callback so the demo topic is set BEFORE the
        # text_input widget is re-instantiated on the next run.  Modifying
        # st.session_state.fp_topic directly after the widget exists raises
        # a StreamlitAPIException.
        st.button(
            "💡 Demo Topic", key="fp_demo", use_container_width=True,
            on_click=_set_demo_topic,
        )

    if run_btn:
        if not topic.strip():
            st.error("Please enter a research topic.")
        else:
            _stream_full_pipeline(topic.strip())


def _stream_full_pipeline(topic: str) -> None:
    """
    Execute the LangGraph pipeline and stream each agent's result into the UI.

    Uses LangGraph's stream(mode="updates") so every node's output arrives
    incrementally — the user sees results as each agent finishes.
    """
    # ── Progress elements ─────────────────────────────────────────────────────
    progress_bar = st.progress(0, text="Initialising…")
    status_text  = st.empty()

    # Ordered list of (node_name, display_label) for the progress bar
    STEPS = [
        ("orchestrator",         "🧠 Orchestrator"),
        ("research_agent",       "🔍 Research Agent"),
        ("classification_agent", "📂 Classification Agent"),
        ("ner_agent",            "🏷️ NER Agent"),
        ("browser_agent",        "🌐 Browser Agent"),
        ("analyzer_agent",       "📊 Analyzer Agent"),
        ("illustration_agent",   "🎨 Illustration Agent"),
        ("writer_agent",         "✍️ Writer Agent"),
        ("critic_agent",         "🔎 Critic Agent"),
        ("finalize_node",        "✅ Finalising"),
    ]
    TOTAL = len(STEPS)

    # ── Agent progress section (Stitch-styled) ────────────────────────────────
    _is_dark  = st.session_state.get("app_theme", "Dark") == "Dark"
    _head_col = "#92ccff" if _is_dark else "#006397"
    st.markdown(
        f"<h3 style='font-size:1rem;font-weight:600;color:{_head_col};"
        f"letter-spacing:-0.01em;margin:16px 0 8px 0;'>🔄 Agent Progress</h3>",
        unsafe_allow_html=True,
    )
    expanders: dict[str, Any] = {}
    for _, label in STEPS:
        expanders[label] = st.expander(label, expanded=False)

    st.markdown("### 📝 Live Preview")
    preview = st.empty()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    initial    = ResearchState(topic=topic)

    # If a file was uploaded, inject it as an extra source so the pipeline
    # can use its content alongside web-searched sources.
    file_text = st.session_state.get("uploaded_file_text", "")
    file_name = st.session_state.get("uploaded_file_name", "uploaded_file")
    if file_text:
        from src.state import SourceRecord  # noqa: PLC0415
        file_source = SourceRecord(
            url=f"file://{file_name}",
            title=f"Uploaded: {file_name}",
            snippet=file_text[:500],
            full_content=file_text,
            relevance_score=9.0,   # treat uploaded file as highly relevant
            source_type="documentation",
            domain="general",
            relevance_tier="high",
        )
        initial.raw_sources.append(file_source)
        initial.classified_sources.append(file_source)
        st.info(f"📁 File '{file_name}' added as a source to the pipeline.")

    state_dict = _state_to_full_dict(initial)
    app        = _get_pipeline_graph()   # cached — does NOT rebuild on every run
    step_idx   = 0
    final_state: ResearchState | None = None

    try:
        for event in app.stream(
            state_dict,
            stream_mode="updates",
            config=get_graph_config(),
        ):
            for node_name, node_output in event.items():
                step_idx += 1
                label = next(
                    (lbl for nm, lbl in STEPS if nm == node_name),
                    f"⚙️ {node_name}",
                )
                progress_bar.progress(
                    min(step_idx / TOTAL, 1.0),
                    text=f"Running {label}…",
                )
                status_text.info(f"▶ {label}")

                # node_output IS the full state (because _wrap returns full dict)
                state_dict  = dict(node_output)
                state       = _dict_to_state(state_dict)
                final_state = state

                # ── Agent-specific expander content ───────────────────────────
                _fill_expander(node_name, label, state, expanders)

                # Live preview of the draft as it's written
                if state.draft:
                    preview.markdown(
                        state.draft[:2500]
                        + "\n\n*(live preview — full report below when done)*"
                    )

        progress_bar.progress(1.0, text="Complete ✓")
        status_text.success("✅ Pipeline complete!")

    except Exception as exc:
        progress_bar.progress(1.0, text="Error")
        st.error(f"Pipeline error: {exc}")
        logger.exception("Full pipeline error")
        return

    if final_state:
        _render_final_report(final_state, topic)


def _fill_expander(
    node_name: str,
    label: str,
    state: ResearchState,
    expanders: dict,
) -> None:
    """Populate one expander with this agent's specific output."""
    if label not in expanders:
        return

    with expanders[label]:
        if node_name == "orchestrator":
            st.success(f"{len(state.sub_questions)} sub-questions generated")
            for q in state.sub_questions:
                st.markdown(f"- {q}")

        elif node_name == "research_agent":
            st.success(f"Collected **{len(state.raw_sources)}** sources")
            for src in state.raw_sources[:6]:
                st.markdown(
                    f"**{src.title}** — score {src.relevance_score:.1f}/10  \n"
                    f"<{src.url}>"
                )

        elif node_name == "classification_agent":
            kept = len(state.classified_sources)
            disc = len(state.classification_log)
            st.success(f"{kept} sources kept · {disc} discarded")
            counts = Counter(s.source_type for s in state.classified_sources)
            if counts:
                st.table({
                    "Source Type": list(counts.keys()),
                    "Count": list(counts.values()),
                })
            if state.classification_log:
                with st.expander("Discarded sources log", expanded=False):
                    for entry in state.classification_log:
                        st.caption(entry)

        elif node_name == "ner_agent":
            st.success(f"{len(state.entities)} unique entities extracted")
            _render_entity_table(state, max_rows=15)
            if state.entity_relationships:
                st.caption(
                    f"Co-occurrence pairs: {len(state.entity_relationships)}"
                )

        elif node_name == "browser_agent":
            st.success(f"Browser visited {len(state.browser_results)} pages")
            for br in state.browser_results:
                st.markdown(f"**{br.get('title', 'N/A')}** — `{br.get('url', '')}`")
                if br.get("screenshot_path"):
                    try:
                        st.image(br["screenshot_path"], width=420)
                    except Exception:
                        st.caption(f"Screenshot: {br['screenshot_path']}")
                if "error" in br:
                    st.warning(f"Browser error: {br['error']}")

        elif node_name == "analyzer_agent":
            st.success(f"{len(state.themes)} themes · {len(state.contradictions)} contradictions")
            st.markdown("**Key themes:**")
            for t in state.themes:
                st.markdown(f"- {t}")
            if state.contradictions:
                st.markdown("**Contradictions:**")
                for c in state.contradictions:
                    st.markdown(f"- ⚠️ {c}")

        elif node_name == "illustration_agent":
            st.success(f"{len(state.illustrations)} figures generated")
            for path in state.illustrations:
                try:
                    st.image(path, caption=Path(path).name, width=480)
                except Exception:
                    st.caption(f"Figure: {path}")

        elif node_name == "writer_agent":
            st.success(f"Draft: {len(state.draft):,} characters")
            st.markdown(state.draft[:1500] + "\n\n*(preview truncated)*")

        elif node_name == "critic_agent":
            score = state.critic_score
            cls   = "score-high" if score >= config.critic_pass_score else "score-low"
            st.markdown(
                f"<span class='{cls}'>{score}/10</span> → **{state.critic_decision}**",
                unsafe_allow_html=True,
            )
            st.markdown(f"*{state.critic_feedback}*")

        elif node_name == "finalize_node":
            st.success("Report saved to `reports/`")


def _render_final_report(state: ResearchState, topic: str) -> None:
    """Show the complete report, cost metrics, and download buttons."""
    st.markdown("---")
    st.markdown("## 📄 Final Report")
    st.markdown(state.final_report)

    st.markdown("---")
    st.markdown("### 💰 Cost & Token Summary")
    cost_data = format_cost_table(state.cost_metrics)
    st.table(cost_data)

    st.markdown("### 📥 Download")
    col_md, col_pdf = st.columns(2)

    with col_md:
        st.download_button(
            "⬇️ Download Markdown",
            data=state.final_report.encode("utf-8"),
            file_name=f"{topic[:30].replace(' ', '_')}_report.md",
            mime="text/markdown",
        )

    with col_pdf:
        from src.utils.report_exporter import save_pdf  # noqa: PLC0415
        pdf_path = save_pdf(state.final_report, topic)
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF",
                    data=f.read(),
                    file_name=f"{topic[:30].replace(' ', '_')}_report.pdf",
                    mime="application/pdf",
                )

    if state.errors:
        with st.expander("⚠️ Non-fatal pipeline errors"):
            for e in state.errors:
                st.warning(e)


# ─────────────────────────────────────────────────────────────────────────────
#  Agent Playground UI  (Mode 2)
# ─────────────────────────────────────────────────────────────────────────────

def render_playground_ui(mode: str) -> None:
    """
    Render the Agent Playground interface for a specific single-agent mode.

    Each mode shows ONLY the inputs it needs and ONLY the outputs it produces.
    The router (src/router.py) handles dependency resolution transparently.
    """
    st.markdown(
        '<span class="mode-pill mode-playground">🧪 Agent Playground</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {mode}")
    st.info(MODE_DESCRIPTIONS.get(mode, ""), icon="ℹ️")

    # ── Inline file upload (above form) ───────────────────────────────────────
    # Each playground agent gets its own upload widget so the user can clearly
    # see and control what document is being processed by that specific agent.
    _render_file_upload(prefix=f"pg_{mode.replace(' ', '_').lower()}")

    # ── Build mode-specific input form ────────────────────────────────────────
    inputs: dict[str, Any] = {}
    accepted_inputs = MODE_INPUTS.get(mode, [])

    with st.form(key=f"form_{mode.replace(' ', '_')}"):

        if "topic" in accepted_inputs:
            inputs["topic"] = st.text_input(
                "Research Topic",
                placeholder="e.g. Large Language Model training techniques",
            )

        if "text" in accepted_inputs:
            inputs["text"] = st.text_area(
                "Paste text for NER analysis  (leave blank to auto-fetch from topic)",
                height=180,
                placeholder="Paste any text here, or leave blank to auto-search the topic above.",
            )

        if "urls" in accepted_inputs:
            raw_urls = st.text_area(
                "URLs  (one per line)",
                placeholder="https://example.com\nhttps://another.com",
                height=120,
            )
            inputs["urls"] = [u.strip() for u in raw_urls.splitlines() if u.strip()]

        if "draft" in accepted_inputs:
            inputs["draft"] = st.text_area(
                "Paste your draft report for the critic to review",
                height=350,
                placeholder="Paste any Markdown draft here…",
            )
            inputs["topic"] = st.text_input(
                "Topic label  (optional — used to give the critic context)",
                value="",
            )

        if "image_prompts" in accepted_inputs and mode == "Illustration Only":
            raw_prompts = st.text_area(
                "Image prompts  (one per line, or leave blank to auto-derive from topic)",
                placeholder="Architecture diagram of Transformer model\nComparison chart of MoE vs Dense models",
                height=120,
            )
            inputs["image_prompts"] = [
                p.strip() for p in raw_prompts.splitlines() if p.strip()
            ]

        # ── Dependency notice ─────────────────────────────────────────────────
        _dep_notice = _dependency_notice(mode, inputs)
        if _dep_notice:
            st.caption(_dep_notice)

        run_btn = st.form_submit_button(
            f"▶ Run {mode}", type="primary", use_container_width=True
        )

    # ── Execute and render results ────────────────────────────────────────────
    if run_btn:
        # If a file was uploaded and the mode accepts text/draft input,
        # inject the file content automatically so the user doesn't need
        # to paste it manually.
        file_text = st.session_state.get("uploaded_file_text", "")
        file_name = st.session_state.get("uploaded_file_name", "uploaded_file")

        if file_text:
            # For NER: inject as text if user didn't type anything
            if mode == "NER Only" and not inputs.get("text", "").strip():
                inputs["text"] = file_text
            # For Critic: inject as draft if user didn't type anything
            elif mode == "Critic Only" and not inputs.get("draft", "").strip():
                inputs["draft"] = file_text
                if not inputs.get("topic", "").strip():
                    inputs["topic"] = file_name
            # For all other modes: attach as extra_file_text for router
            else:
                inputs["extra_file_text"] = file_text
                inputs["extra_file_name"] = file_name

        with st.spinner(f"Running {mode}…"):
            result = route(mode, inputs)

        if result.get("status") == "error":
            for err in result.get("errors", ["Unknown error"]):
                st.error(err)
        else:
            _render_playground_result(mode, result)

            # Show cost summary
            cost = result.get("cost_metrics")
            if cost and hasattr(cost, "input_tokens") and cost.input_tokens > 0:
                with st.expander("💰 Cost & Token Usage"):
                    st.table(format_cost_table(cost))

            # Show any non-fatal errors
            errs = result.get("errors", [])
            if errs:
                with st.expander("⚠️ Non-fatal errors"):
                    for e in errs:
                        st.warning(e)


def _dependency_notice(mode: str, inputs: dict) -> str:
    """Return a string explaining which prerequisites will be auto-run."""
    notices = {
        "Research Only":
            "ℹ️ Auto-runs: Orchestrator (sub-questions) → Research Agent",
        "Classification Only":
            "ℹ️ Auto-runs: Orchestrator → Research Agent → Classification Agent"
            if not inputs.get("urls")
            else "ℹ️ Classifying the URLs you provided directly.",
        "NER Only":
            "ℹ️ Auto-runs: Orchestrator → Research Agent → NER Agent"
            if not inputs.get("text")
            else "ℹ️ Running NER on the text you pasted.",
        "Browser Only":
            "ℹ️ Visits the URLs you provide — no search needed.",
        "Analysis Only":
            "ℹ️ Auto-runs: Orchestrator → Research → Classification → NER → Analyzer",
        "Writer Only":
            "ℹ️ Auto-runs: full chain up to Illustrations, then Writer.",
        "Critic Only":
            "ℹ️ Reviews the draft you paste — no web search needed.",
        "Illustration Only":
            "ℹ️ Uses your prompts directly."
            if inputs.get("image_prompts")
            else "ℹ️ Auto-runs: Research → Analysis → Illustration",
    }
    return notices.get(mode, "")


def _render_playground_result(mode: str, result: dict) -> None:
    """Dispatch to the correct result renderer for each playground mode."""
    renderers = {
        "Research Only":        _render_research_result,
        "Classification Only":  _render_classification_result,
        "NER Only":             _render_ner_result,
        "Browser Only":         _render_browser_result,
        "Analysis Only":        _render_analysis_result,
        "Writer Only":          _render_writer_result,
        "Critic Only":          _render_critic_result,
        "Illustration Only":    _render_illustration_result,
    }
    renderer = renderers.get(mode)
    if renderer:
        renderer(result)
    else:
        st.json(result)


# ── Individual result renderers ────────────────────────────────────────────────

def _render_research_result(result: dict) -> None:
    sources = result.get("sources", [])
    st.success(f"✅ Collected **{len(sources)}** sources")

    if result.get("sub_questions"):
        with st.expander("Sub-questions generated"):
            for q in result["sub_questions"]:
                st.markdown(f"- {q}")

    if sources:
        st.markdown("#### Sources")
        for i, src in enumerate(sources, 1):
            with st.expander(f"[{i}] {src.title[:80]} — score {src.relevance_score:.1f}/10"):
                st.markdown(f"**URL:** {src.url}")
                st.markdown(f"**Type:** {src.source_type} · **Domain:** {src.domain}")
                st.markdown(f"**Snippet:** {src.snippet[:400]}")

        # Summary table
        st.markdown("#### Summary Table")
        st.dataframe({
            "#":       list(range(1, len(sources) + 1)),
            "Title":   [s.title[:60] for s in sources],
            "Score":   [s.relevance_score for s in sources],
            "URL":     [s.url for s in sources],
        }, use_container_width=True)

    # Download sources as CSV
    import io, csv  # noqa: PLC0415
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["#", "Title", "URL", "Score", "Snippet"])
    for i, src in enumerate(sources, 1):
        w.writerow([i, src.title, src.url, src.relevance_score, src.snippet[:200]])
    st.download_button(
        "⬇️ Download Sources CSV",
        buf.getvalue().encode(),
        file_name="research_sources.csv",
        mime="text/csv",
    )


def _render_classification_result(result: dict) -> None:
    classified = result.get("classified_sources", [])
    st.success(f"✅ Classified **{len(classified)}** sources")

    if classified:
        type_counts   = Counter(s.source_type for s in classified)
        domain_counts = Counter(s.domain for s in classified)
        rel_counts    = Counter(s.relevance_tier for s in classified)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Source Types**")
            for t, c in type_counts.most_common():
                st.markdown(f"- {t}: **{c}**")
        with col2:
            st.markdown("**Domains**")
            for d, c in domain_counts.most_common():
                st.markdown(f"- {d}: **{c}**")
        with col3:
            st.markdown("**Relevance Tiers**")
            for r, c in rel_counts.most_common():
                st.markdown(f"- {r}: **{c}**")

        st.markdown("#### Classified Sources")
        st.dataframe({
            "Title":     [s.title[:60] for s in classified],
            "Type":      [s.source_type for s in classified],
            "Domain":    [s.domain for s in classified],
            "Relevance": [s.relevance_tier for s in classified],
            "Score":     [s.relevance_score for s in classified],
            "URL":       [s.url for s in classified],
        }, use_container_width=True)

    log = result.get("classification_log", [])
    if log:
        with st.expander(f"🗑️ Discarded sources ({len(log)})"):
            for entry in log:
                st.caption(entry)


def _render_ner_result(result: dict) -> None:
    entities = result.get("entities", [])
    rels     = result.get("entity_relationships", [])

    st.success(f"✅ Extracted **{len(entities)}** unique entities")

    if not entities:
        st.info("No named entities found in the provided text.")
        return

    # ── Theme-aware category styles ────────────────────────────────────────────
    # Dark mode uses dark-surface chips with bright accent text (Stitch tokens).
    # Light mode uses pastel chips.
    _is_dark = st.session_state.get("app_theme", "Dark") == "Dark"

    _CAT_STYLE_DARK: dict[str, tuple[str, str, str]] = {
        # (bg, fg, border)
        "person":       ("#3d1212", "#ffb3b3", "rgba(255,179,179,0.30)"),
        "organization": ("#0d1e3a", "#92ccff", "rgba(146,204,255,0.30)"),
        "location":     ("#0d2d1a", "#61de8a", "rgba(97,222,138,0.30)"),
        "technology":   ("#3d2a00", "#ffba4b", "rgba(255,186,75,0.30)"),
        "concept":      ("#2a0d3a", "#c9a4ff", "rgba(201,164,255,0.30)"),
        "date":         ("#0d3330", "#61de8a", "rgba(97,222,138,0.30)"),
    }
    _CAT_STYLE_LIGHT: dict[str, tuple[str, str, str]] = {
        "person":       ("#fde8e8", "#7b1c1c", "rgba(123,28,28,0.20)"),
        "organization": ("#e8eeff", "#1c357a", "rgba(28,53,122,0.20)"),
        "location":     ("#e8f7e8", "#1a5c1a", "rgba(26,92,26,0.20)"),
        "technology":   ("#fff5e0", "#7a4d00", "rgba(122,77,0,0.20)"),
        "concept":      ("#f5e8ff", "#4c1a7a", "rgba(76,26,122,0.20)"),
        "date":         ("#e0faf4", "#0e5c44", "rgba(14,92,68,0.20)"),
    }
    _CAT_STYLE = _CAT_STYLE_DARK if _is_dark else _CAT_STYLE_LIGHT

    # Icon per category (Material Symbols)
    _CAT_ICON: dict[str, str] = {
        "person": "person", "organization": "corporate_fare",
        "location": "location_on", "technology": "memory",
        "concept": "lightbulb", "date": "calendar_today",
    }

    # ── Bento-grid metric summary ─────────────────────────────────────────────
    cat_counts = Counter(e.category for e in entities)
    total      = len(entities)

    cards_html = ""
    for cat, cnt in cat_counts.most_common():
        bg, fg, border = _CAT_STYLE.get(cat, ("#1c2024", "#e0e3e8", "#3f4850"))
        icon = _CAT_ICON.get(cat, "label")
        pct  = int(cnt / total * 100) if total else 0
        cards_html += (
            f'<div class="bento-card" style="background:{bg};border-color:{border};">'
            f'  <div class="bc-value" style="color:{fg};">{cnt}</div>'
            f'  <div class="bc-label" style="color:{fg};opacity:0.75;">'
            f'    {cat.title()}'
            f'  </div>'
            f'  <div style="font-size:0.65rem;opacity:0.5;margin-top:2px;">{pct}%</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="bento-grid">{cards_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Grouped display by category ────────────────────────────────────────────
    st.markdown("#### 🏷️ Entities by Category")
    st.caption(
        "Each chip shows the extracted word and its occurrence count (×N). "
        "Grouped by entity type — colour-coded per the Stitch design system."
    )

    cat_groups: dict = {}
    for e in entities:
        cat_groups.setdefault(e.category, []).append(e)

    for cat in sorted(cat_groups.keys()):
        ents = sorted(cat_groups[cat], key=lambda x: x.count, reverse=True)
        bg, fg, border = _CAT_STYLE.get(cat, ("#1c2024", "#e0e3e8", "#3f4850"))
        icon = _CAT_ICON.get(cat, "label")

        # Category header
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;"
            f"margin:10px 0 4px 0;'>"
            f"<span class='material-symbols-outlined' "
            f"style='font-size:16px;color:{fg};'>{icon}</span>"
            f"<b style='font-size:0.92rem;color:{fg};'>{cat.title()}</b>"
            f"<span style='font-size:0.75rem;opacity:0.55;margin-left:4px;'>"
            f"({len(ents)} entities)</span></div>",
            unsafe_allow_html=True,
        )

        # Entity chips
        tags_html = "".join(
            f'<span style="background:{bg}; color:{fg}; border:1px solid {border}; '
            f'padding:4px 12px; border-radius:14px; font-size:0.84rem; '
            f'font-weight:500; margin:3px 2px; display:inline-block; '
            f'white-space:nowrap; font-family:Inter,sans-serif;">'
            f'{e.text}'
            f'<span style="opacity:0.50; font-size:0.70rem; margin-left:5px;">'
            f'×{e.count}</span>'
            f'</span>'
            for e in ents[:25]
        )
        st.markdown(
            f'<div style="margin-bottom:14px; line-height:2.4;">{tags_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Full table (collapsed) ────────────────────────────────────────────────
    with st.expander("📋 Full entity table (sortable)"):
        st.dataframe({
            "Entity":      [e.text     for e in entities[:80]],
            "Category":    [e.category for e in entities[:80]],
            "SpaCy Label": [e.label    for e in entities[:80]],
            "Count":       [e.count    for e in entities[:80]],
        }, use_container_width=True)

    # ── Co-occurrence relationships ────────────────────────────────────────────
    if rels:
        with st.expander(f"🔗 Co-occurrence pairs ({len(rels)})"):
            st.caption(
                "Entities that appeared in the same sentence — "
                "a proxy for semantic relationships in the text."
            )
            # Build entity→category lookup
            _cat_map = {e.text: e.category for e in entities}
            for a, b in rels[:30]:
                bg_a, fg_a, _ = _CAT_STYLE.get(_cat_map.get(a, "concept"),
                                                ("#1c2024", "#e0e3e8", "#3f4850"))
                bg_b, fg_b, _ = _CAT_STYLE.get(_cat_map.get(b, "concept"),
                                                ("#1c2024", "#e0e3e8", "#3f4850"))
                st.markdown(
                    f'<span style="background:{bg_a};color:{fg_a};padding:3px 10px;'
                    f'border-radius:10px;font-size:0.83rem;font-weight:500;">{a}</span>'
                    f'<span style="opacity:0.5;margin:0 6px;font-size:0.9rem;">↔</span>'
                    f'<span style="background:{bg_b};color:{fg_b};padding:3px 10px;'
                    f'border-radius:10px;font-size:0.83rem;font-weight:500;">{b}</span>',
                    unsafe_allow_html=True,
                )

    # ── Download ───────────────────────────────────────────────────────────────
    import io, csv  # noqa: PLC0415
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Entity", "Category", "SpaCy Label", "Count"])
    for e in entities:
        w.writerow([e.text, e.category, e.label, e.count])
    st.download_button(
        "⬇️ Download Entities CSV",
        buf.getvalue().encode(),
        file_name="ner_entities.csv",
        mime="text/csv",
    )


def _render_browser_result(result: dict) -> None:
    browser_results = result.get("browser_results", [])
    st.success(f"✅ Visited **{len(browser_results)}** pages")

    for i, br in enumerate(browser_results, 1):
        with st.expander(f"Page {i}: {br.get('title', br.get('url', 'Unknown'))}"):
            st.markdown(f"**URL:** {br.get('url', 'N/A')}")
            if br.get("title"):
                st.markdown(f"**Title:** {br['title']}")
            if br.get("description"):
                st.markdown(f"**Description:** {br['description']}")
            if br.get("headings"):
                st.markdown("**Headings:**")
                for h in br["headings"]:
                    st.markdown(f"  - {h}")
            if br.get("body_text"):
                st.markdown("**Body preview:**")
                st.text(br["body_text"][:600])
            if br.get("screenshot_path"):
                try:
                    st.image(
                        br["screenshot_path"],
                        caption=f"Screenshot — {br.get('url', '')}",
                        use_column_width=True,
                    )
                except Exception:
                    st.caption(f"Screenshot saved: {br['screenshot_path']}")
            if "error" in br:
                st.warning(f"Error: {br['error']}")


def _render_analysis_result(result: dict) -> None:
    themes        = result.get("themes", [])
    contradictions = result.get("contradictions", [])
    outline       = result.get("outline", [])
    evidence      = result.get("evidence_summary", {})
    prompts       = result.get("image_prompts", [])

    st.success(
        f"✅ {len(themes)} themes · {len(contradictions)} contradictions · "
        f"{result.get('sources_used', '?')} sources analysed"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Key Themes")
        for i, t in enumerate(themes, 1):
            st.markdown(f"**{i}.** {t}")
            if t in evidence:
                st.caption(f"> {evidence[t]}")

    with col2:
        st.markdown("#### 📋 Report Outline")
        for s in outline:
            st.markdown(f"- {s}")

    if contradictions:
        st.markdown("#### ⚠️ Source Contradictions")
        for c in contradictions:
            st.warning(c)

    if prompts:
        st.markdown("#### 🎨 Suggested Image Prompts")
        for p in prompts:
            st.markdown(f"- *{p}*")


def _render_writer_result(result: dict) -> None:
    draft = result.get("draft", "")
    st.success(
        f"✅ Draft generated — **{len(draft):,}** characters · "
        f"{result.get('sources_used', '?')} sources cited"
    )

    st.markdown("#### 📝 Draft Report")
    st.markdown(draft)

    # Download draft
    st.download_button(
        "⬇️ Download Draft (Markdown)",
        data=draft.encode("utf-8"),
        file_name="draft_report.md",
        mime="text/markdown",
    )


def _render_critic_result(result: dict) -> None:
    score    = result.get("critic_score", 0)
    decision = result.get("critic_decision", "N/A")
    feedback = result.get("critic_feedback", "")

    score_class = "score-high" if score >= config.critic_pass_score else "score-low"
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            f'<div class="metric-card"><span class="{score_class}">'
            f'{score}/10</span><br><b>{decision}</b></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("#### Feedback")
        st.markdown(feedback)

    st.markdown("#### Rubric Explanation")
    st.caption(
        "Scores reflect: accuracy · completeness · clarity · citations · "
        "NER usage · source classification · illustration quality.  "
        f"Pass threshold: {config.critic_pass_score}/10."
    )


def _render_illustration_result(result: dict) -> None:
    illustrations = result.get("illustrations", [])
    prompts       = result.get("image_prompts", [])

    st.success(f"✅ Generated **{len(illustrations)}** figures")

    for i, path in enumerate(illustrations, 1):
        prompt = prompts[i - 1] if i <= len(prompts) else f"Figure {i}"
        st.markdown(f"**Figure {i}:** *{prompt}*")
        try:
            st.image(path, use_column_width=True)
        except Exception:
            st.caption(f"Saved: {path}")

    if not illustrations:
        st.warning("No figures were generated.  Check that matplotlib is installed.")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_entity_table(state: ResearchState, max_rows: int = 20) -> None:
    """Render a compact entity table from a ResearchState."""
    entities = state.entities[:max_rows]
    if not entities:
        st.caption("No entities extracted.")
        return
    st.dataframe({
        "Entity":   [e.text for e in entities],
        "Category": [e.category for e in entities],
        "Count":    [e.count for e in entities],
    }, use_container_width=True)


# Type stub for Streamlit containers
from typing import Any  # noqa: E402 (after functions that use it)


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Render the app: sidebar → mode → appropriate UI."""

    # ── Inject Material Symbols CDN (needed for sidebar + NER icons) ──────────
    st.markdown(
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200">'
        '<style>.material-symbols-outlined{font-variation-settings:'
        "'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 24;"
        "display:inline-block;vertical-align:middle;line-height:1;}</style>",
        unsafe_allow_html=True,
    )

    # ── Stitch-styled header ──────────────────────────────────────────────────
    _is_dark = st.session_state.get("app_theme", "Dark") == "Dark"
    _primary  = "#92ccff" if _is_dark else "#006397"
    _subtitle = "#bfc7d2" if _is_dark else "#64748b"
    st.markdown(
        f"""
        <div style="margin-bottom:8px;">
          <h1 style="font-size:1.6rem;font-weight:700;letter-spacing:-0.02em;
                     color:{_primary};margin:0 0 4px 0;font-family:Inter,sans-serif;">
            🔬 Multi-Agent Research Assistant
          </h1>
          <p style="font-size:0.875rem;color:{_subtitle};margin:0;line-height:1.6;">
            <b>Full Pipeline</b> — runs all 9 LangGraph agents and produces a
            complete cited report.&nbsp; <b>Agent Playground</b> — run any
            single agent in isolation.&nbsp; Upload a document to analyse your
            own files without web search.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar (returns selected mode) ───────────────────────────────────────
    selected_mode = render_sidebar()

    st.divider()

    # ── Route to the correct UI ───────────────────────────────────────────────
    if selected_mode == "Full Pipeline":
        render_full_pipeline_ui()
    else:
        render_playground_ui(selected_mode)


if __name__ == "__main__":
    main()
else:
    # Streamlit runs the module directly; call main() at module level
    main()
