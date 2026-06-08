"""
src/ui/sidebar.py — Stitch-styled sidebar with clickable nav buttons.

Each navigation mode is a real st.button().  Clicking it sets
st.session_state["nav_mode"] and triggers st.rerun() so the main area
updates immediately.  The active button uses type="primary", which the
CSS in dark.css / light.css transforms into the Stitch active-indicator
style (tinted background + right-side border in the primary colour).
"""

from __future__ import annotations

import streamlit as st

from src.config import config
from src.router import ALL_MODES, MODE_DESCRIPTIONS
from src.utils.error_handler import check_api_keys

# Emoji icon per mode — always renders correctly, no font loading required.
_MODE_EMOJI: dict[str, str] = {
    "Full Pipeline":       "⬡",
    "Research Only":       "🔍",
    "Classification Only": "📂",
    "NER Only":            "🏷️",
    "Browser Only":        "🌐",
    "Analysis Only":       "📊",
    "Writer Only":         "✍️",
    "Critic Only":         "⭐",
    "Illustration Only":   "🎨",
}


def render_sidebar() -> str:
    """
    Render the sidebar and return the currently selected mode name.

    Returns:
        One of the strings in ALL_MODES (e.g. "Full Pipeline", "NER Only").
    """
    # Persist selected mode across reruns
    if "nav_mode" not in st.session_state:
        st.session_state["nav_mode"] = "Full Pipeline"

    with st.sidebar:
        _is_dark = st.session_state.get("app_theme", "Dark") == "Dark"
        _p_col   = "#92ccff" if _is_dark else "#006397"
        _m_col   = "#bfc7d2" if _is_dark else "#64748b"
        _lbl_col = "#89929b" if _is_dark else "#94a3b8"

        # ── Brand header ──────────────────────────────────────────────────
        st.markdown(
            f"""<div style="padding:20px 4px 12px 4px;">
                <div style="font-size:1.05rem;font-weight:700;
                            letter-spacing:-0.01em;color:{_p_col};">
                    🔬 Research Assistant
                </div>
                <div style="font-size:0.70rem;color:{_m_col};
                            letter-spacing:0.05em;text-transform:uppercase;
                            margin-top:3px;">
                    Powered by L2 Team
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

        # ── Theme toggle ──────────────────────────────────────────────────
        st.radio("🌓 Theme", options=["Light", "Dark"], key="app_theme", horizontal=True)
        st.divider()

        # ── API key warnings ──────────────────────────────────────────────
        for w in check_api_keys():
            st.warning(w, icon="⚠️")

        # ── Nav buttons ───────────────────────────────────────────────────
        st.markdown(
            f"<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{_lbl_col};padding:0 4px 6px 4px;'>"
            f"Navigation</div>",
            unsafe_allow_html=True,
        )
        for mode in ALL_MODES:
            is_active = st.session_state["nav_mode"] == mode
            label     = f"{_MODE_EMOJI.get(mode, '▸')}  {mode}"
            if st.button(
                label,
                key=f"nav_btn_{mode.replace(' ', '_')}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["nav_mode"] = mode
                st.rerun()

        selected_mode = st.session_state["nav_mode"]

        # ── Mode badge + description ──────────────────────────────────────
        st.divider()
        badge = (
            '<span class="mode-pill mode-full">📋 Competition Mode</span>'
            if selected_mode == "Full Pipeline"
            else '<span class="mode-pill mode-playground">🧪 Agent Playground</span>'
        )
        st.markdown(badge, unsafe_allow_html=True)
        st.info(MODE_DESCRIPTIONS.get(selected_mode, ""), icon="ℹ️")

        # ── Configuration ─────────────────────────────────────────────────
        st.divider()
        st.markdown(
            f"<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.08em;"
            f"text-transform:uppercase;color:{_lbl_col};padding:0 4px 6px 4px;'>"
            f"Configuration</div>",
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

        # ── MCP tools list ────────────────────────────────────────────────
        with st.expander("🔧 MCP Tools"):
            from src.tools.mcp_tools import list_tools  # noqa: PLC0415
            for tool in list_tools():
                st.markdown(f"- **{tool['name']}**")

        # ── Recent runs summary ───────────────────────────────────────────
        history = st.session_state.get("pipeline_history", [])
        if history:
            st.divider()
            st.markdown(
                f"<div style='font-size:0.68rem;font-weight:600;letter-spacing:0.08em;"
                f"text-transform:uppercase;color:{_lbl_col};padding:0 4px 4px 4px;'>"
                f"📚 Recent Runs</div>",
                unsafe_allow_html=True,
            )
            for i, run in enumerate(history):
                icon = ["🕐", "🕑", "🕒"][i]
                score = run.get("critic_score", 0)
                score_col = "#61de8a" if score >= 7 else "#ffba4b" if score >= 5 else "#ffb4ab"
                st.markdown(
                    f"<div style='font-size:0.78rem;padding:4px 4px;line-height:1.5;"
                    f"border-left:3px solid {score_col};padding-left:8px;margin-bottom:4px;'>"
                    f"{icon} <b>{run['topic'][:28]}</b><br>"
                    f"<span style='color:{_lbl_col};font-size:0.70rem;'>"
                    f"{run['timestamp']} · {score}/10</span></div>",
                    unsafe_allow_html=True,
                )

        # ── Footer ────────────────────────────────────────────────────────
        st.markdown(
            "<div class='l2-footer'>Multi-Agent Research Assistant<br>"
            "<span style='opacity:0.6;font-size:0.72rem;'>L2 Team · 2025</span></div>",
            unsafe_allow_html=True,
        )

    return selected_mode
