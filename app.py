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
/* Progress bar colour (same in both themes) */
.stProgress .st-bo { background-color: #3498db; }

/* Mode badge pills — high specificity + !important so theme text rules
   (e.g. the dark-mode "make all text light") can never wash them out. */
.mode-pill {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:0.78rem; font-weight:600; margin-bottom:6px;
}
.stApp .mode-full       { background:#d4edda !important; color:#155724 !important; }
.stApp .mode-playground { background:#cce5ff !important; color:#004085 !important; }
.stApp .file-pill {
    display:inline-block; padding:3px 10px; border-radius:12px;
    font-size:0.78rem; font-weight:600; margin-bottom:6px;
    background:#fff3cd !important; color:#856404 !important;
}

/* Score colours (also protected with !important) */
.stApp .score-high { color:#27ae60 !important; font-weight:bold; font-size:1.4rem; }
.stApp .score-low  { color:#e74c3c !important; font-weight:bold; font-size:1.4rem; }
"""

# Per-theme colour tokens.  We override Streamlit's main containers so the whole
# page (not just our custom elements) switches between light and dark.
_LIGHT_CSS = """
.stApp { background-color: #ffffff; color: #1a1a1a; }
[data-testid="stHeader"] { background-color: #ffffff; }
section[data-testid="stSidebar"] { background-color: #f4f6f8; }
.metric-card {
    background:#f8f9fa; border-radius:8px; padding:14px;
    border-left:4px solid #3498db; margin:4px 0; color:#1a1a1a;
}
.entity-tag {
    display:inline-block; padding:2px 8px; border-radius:10px;
    background:#e8f4fd; color:#1a5276; font-size:0.82rem; margin:2px;
}
.file-info-card {
    background:#fffdf0; border:1px solid #ffc107; border-radius:8px;
    padding:10px 14px; margin:8px 0; font-size:0.88rem; color:#1a1a1a;
}
.l2-footer {
    text-align:center; padding:12px; color:#6c757d;
    font-size:0.80rem; border-top:1px solid #e9ecef; margin-top:24px;
}
"""

_DARK_CSS = """
/* Main containers + the top header toolbar */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] { background-color: #0e1117 !important; color: #e6e6e6 !important; }
[data-testid="stHeader"] { background-color: #0e1117 !important; }
[data-testid="stToolbar"] { color: #e6e6e6 !important; }
section[data-testid="stSidebar"] { background-color: #161a23 !important; }

/* Body text — target Streamlit's markdown containers and common text elements.
   NOTE: we deliberately do NOT use a bare `span` selector here, because that
   would override the coloured badge pills above. */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricLabel"] { color: #e6e6e6 !important; }

/* Caption / help text slightly dimmer */
.stApp [data-testid="stCaptionContainer"] { color: #9aa4b2 !important; }

/* Inputs, text areas, selectboxes */
.stApp textarea, .stApp input,
.stApp .stTextInput input, .stApp .stTextArea textarea,
.stApp .stSelectbox div[data-baseweb="select"] > div {
    background-color:#1c2230 !important; color:#e6e6e6 !important;
    border-color:#2a2f3a !important;
}

/* Secondary buttons (e.g. "Demo Topic") — primary blue button keeps its colour */
.stApp .stButton button[kind="secondary"],
.stApp .stDownloadButton button {
    background-color:#1c2230 !important; color:#e6e6e6 !important;
    border:1px solid #2a2f3a !important;
}

/* Expanders */
.stApp [data-testid="stExpander"] {
    background-color:#161a23 !important; border:1px solid #2a2f3a !important;
}
.stApp [data-testid="stExpander"] summary { color:#e6e6e6 !important; }

/* Dataframes / tables */
.stApp [data-testid="stTable"], .stApp [data-testid="stDataFrame"] {
    background-color:#161a23 !important; color:#e6e6e6 !important;
}

.metric-card {
    background:#1c2230; border-radius:8px; padding:14px;
    border-left:4px solid #3498db; margin:4px 0; color:#e6e6e6;
}
.entity-tag {
    display:inline-block; padding:2px 8px; border-radius:10px;
    background:#22384d; color:#9ecbff; font-size:0.82rem; margin:2px;
}
.file-info-card {
    background:#2a2615; border:1px solid #ffc107; border-radius:8px;
    padding:10px 14px; margin:8px 0; font-size:0.88rem; color:#f0e6c0;
}
.l2-footer {
    text-align:center; padding:12px; color:#9aa4b2;
    font-size:0.80rem; border-top:1px solid #2a2f3a; margin-top:24px;
}

/* ── Form containers ─────────────────────────────────────────────────────── */
.stApp [data-testid="stForm"] {
    background-color: #161a23 !important;
    border: 1px solid #2a2f3a !important;
    border-radius: 8px !important;
}

/* ── File upload dropzone ────────────────────────────────────────────────── */
.stApp [data-testid="stFileUploadDropzone"] {
    background-color: #1c2230 !important;
    border: 2px dashed #3a4055 !important;
    border-radius: 8px !important;
}
.stApp [data-testid="stFileUploadDropzone"] p,
.stApp [data-testid="stFileUploadDropzone"] span {
    color: #9aa4b2 !important;
}
.stApp [data-testid="stFileUploaderFileName"] { color: #e6e6e6 !important; }
/* The small "X" / browse button inside the uploader */
.stApp button[data-testid="stBaseButton-minimal"] { color: #9aa4b2 !important; }

/* ── Alert / info / success / warning / error boxes ─────────────────────── */
.stApp [data-testid="stAlertContainer"] {
    background-color: #1c2230 !important;
}
.stApp [data-testid="stAlertContainer"] p,
.stApp [data-testid="stAlertContainer"] div { color: #e6e6e6 !important; }

/* ── Horizontal rule ─────────────────────────────────────────────────────── */
.stApp hr { border-color: #2a2f3a !important; }

/* ── Dropdown menu (selectbox options) ───────────────────────────────────── */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background-color: #1c2230 !important;
}
[data-baseweb="popover"] [data-baseweb="menu"] li { color: #e6e6e6 !important; }
[data-baseweb="popover"] [data-baseweb="menu"] li:hover {
    background-color: #2a2f3a !important;
}

/* ── Spinner message ─────────────────────────────────────────────────────── */
.stApp [data-testid="stSpinner"] p { color: #e6e6e6 !important; }

/* ── Caption / small helper text ─────────────────────────────────────────── */
.stApp small { color: #9aa4b2 !important; }

/* ── Selectbox selected value text ───────────────────────────────────────── */
.stApp .stSelectbox [data-baseweb="select"] [data-baseweb="value"] {
    color: #e6e6e6 !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stApp [data-baseweb="tab"] { color: #9aa4b2 !important; }
.stApp [data-baseweb="tab"][aria-selected="true"] { color: #e6e6e6 !important; }
.stApp [data-baseweb="tab-list"] { background-color: #0e1117 !important; }

/* ── Progress bar background track ───────────────────────────────────────── */
.stApp [data-testid="stProgressBar"] > div {
    background-color: #2a2f3a !important;
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


# Read the persisted theme (defaults to Light on first load) and apply it.
_active_theme = st.session_state.get("app_theme", "Light")
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


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    """
    Render the sidebar and return the selected mode string.

    Layout:
      ┌────────────────────────────┐
      │  [logo]  Research Asst.    │
      │  ─────────────────         │
      │  ■ Full Pipeline           │  ← default, competition requirement
      │                            │
      │  Agent Playground          │  ← innovation feature
      │  ○ Research Only           │
      │  ○ Classification Only     │
      │  ○ NER Only                │
      │  ○ Browser Only            │
      │  ○ Analysis Only           │
      │  ○ Writer Only             │
      │  ○ Critic Only             │
      │  ○ Illustration Only       │
      └────────────────────────────┘
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/artificial-intelligence.png",
            width=72,
        )
        st.title("Research Assistant")
        st.caption("Powered by L2 Team")
        st.divider()

        # ── Theme toggle (Dark / Light) ───────────────────────────────────────
        # The widget writes to st.session_state["app_theme"], which is read at
        # the top of the script to inject the matching CSS palette.  Changing it
        # triggers a rerun, so the new theme applies immediately.
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
        st.markdown("### 🎛️ Operating Mode")

        # Single unified dropdown.  ALL_MODES already starts with "Full Pipeline"
        # (the default / competition mode), followed by every single-agent mode.
        # Selecting "Full Pipeline" runs the complete workflow; any other choice
        # runs that one agent in isolation via the Agent Playground.
        selected_mode = st.selectbox(
            "Choose an agent",
            ALL_MODES,                 # ["Full Pipeline", "Research Only", …]
            index=0,                   # default to Full Pipeline
            key="mode_selector",
        )

        # Show a coloured badge indicating which kind of mode is active
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

        # ── Mode description ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Mode description**")
        st.info(MODE_DESCRIPTIONS.get(selected_mode, ""), icon="ℹ️")

        # ── System config ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        st.markdown(f"**Model:** `{config.anthropic_model}`")
        st.markdown(f"**Min sources:** {config.min_sources}")
        st.markdown(f"**Max revisions:** {config.max_revisions}")
        st.markdown(f"**Critic pass score:** {config.critic_pass_score}/10")

        # ── MCP tools list ────────────────────────────────────────────────────
        with st.expander("🔧 Available MCP Tools"):
            from src.tools.mcp_tools import list_tools  # noqa: PLC0415
            for tool in list_tools():
                st.markdown(f"- **{tool['name']}**")

        st.markdown("---")
        # ── Branding ─────────────────────────────────────────────────────────
        st.markdown(
            "<div style='text-align:center; color:#6c757d; font-size:0.78rem;'>"
            "Powered by <b>L2 team</b></div>",
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

    # Pre-create expander containers so they appear in order
    st.markdown("### 🔄 Agent Progress")
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
    app        = build_graph()
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

    if entities:
        # Category breakdown
        cat_counts = Counter(e.category for e in entities)
        cols = st.columns(len(cat_counts) or 1)
        for col, (cat, cnt) in zip(cols, cat_counts.items()):
            col.metric(label=cat.title(), value=cnt)

        # Full entity table
        st.markdown("#### Entity Frequency Table")
        st.dataframe({
            "Entity":   [e.text for e in entities[:50]],
            "Category": [e.category for e in entities[:50]],
            "SpaCy Label": [e.label for e in entities[:50]],
            "Count":    [e.count for e in entities[:50]],
        }, use_container_width=True)

        # Tag cloud style display
        st.markdown("#### Entity Tags")
        tag_html = " ".join(
            f'<span class="entity-tag">{e.text} ({e.count})</span>'
            for e in entities[:40]
        )
        st.markdown(tag_html, unsafe_allow_html=True)

    if rels:
        st.markdown(f"#### Co-occurrence Relationships ({len(rels)} pairs)")
        with st.expander("View relationship pairs"):
            for a, b in rels[:30]:
                st.markdown(f"- **{a}** ↔ **{b}**")

    # Download
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

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🔬 Multi-Agent Research Assistant")
    st.markdown(
        "**Full Pipeline** runs all 9 agents and produces a complete report.  "
        "**Agent Playground** lets you run any single agent in isolation.  "
        "**Upload a file** from the sidebar to run any operation on your own documents."
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
