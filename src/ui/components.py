"""
src/ui/components.py — Shared UI components used across pipeline and playground.

Functions:
  render_file_upload(prefix)   Inline file-upload expander (reads/writes session_state)
  render_entity_table(state)   Compact entity dataframe inside Full Pipeline expanders
"""

from __future__ import annotations

import html

import streamlit as st

from src.state import ResearchState
from src.utils.file_processor import (
    extract_text,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_EXTENSIONS_DISPLAY,
)


def render_file_upload(prefix: str) -> None:
    """
    Render an inline file uploader expander.

    Extracted text is stored in st.session_state["uploaded_file_text"] so
    every agent mode (Full Pipeline and all Playground modes) can read it.

    Args:
        prefix: Short string to namespace widget keys and avoid collisions
                ("fp" for Full Pipeline, "pg_ner_only" for NER Playground, …).
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
            # Re-extract only when the filename changes (avoid re-parsing on every rerun)
            if uploaded.name != st.session_state.get("uploaded_file_name"):
                with st.spinner("Reading file…"):
                    extracted = extract_text(uploaded.read(), uploaded.name)
                st.session_state["uploaded_file_text"] = extracted
                st.session_state["uploaded_file_name"] = uploaded.name

            chars = len(st.session_state.get("uploaded_file_text", ""))
            fn    = st.session_state.get("uploaded_file_name", uploaded.name)
            st.success(f"✅ **{fn}** — {chars:,} characters extracted")

        else:
            # Keep showing the previously-loaded file with a Clear button
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


def render_table(columns: dict, max_height: int = 360) -> None:
    """
    Render a {column_name: [values]} dict as a theme-aware HTML table.

    Used instead of st.dataframe, whose canvas-based grid renders BLANK under
    the app's custom dark CSS (cell text becomes invisible — an empty box).
    An HTML table with inline colours is guaranteed to render in both themes.
    """
    col_names = list(columns.keys())
    if not col_names:
        return
    n_rows = len(next(iter(columns.values())))
    if n_rows == 0:
        st.caption("No data.")
        return

    is_dark = st.session_state.get("app_theme", "Dark") == "Dark"
    if is_dark:
        head_bg, head_fg = "#1c2024", "#92ccff"
        row_bg, alt_bg, fg, border = "#101418", "#161b20", "#e0e3e8", "#3f4850"
    else:
        head_bg, head_fg = "#e8eef4", "#006397"
        row_bg, alt_bg, fg, border = "#ffffff", "#f3f6f9", "#1a1c1e", "#cdd5dd"

    ths = "".join(
        f"<th style='padding:8px 12px;color:{head_fg};text-align:left;white-space:nowrap;"
        f"position:sticky;top:0;background:{head_bg};'>{html.escape(str(c))}</th>"
        for c in col_names
    )
    body = []
    for i in range(n_rows):
        bg = row_bg if i % 2 == 0 else alt_bg
        tds = "".join(
            f"<td style='padding:6px 12px;color:{fg};border-bottom:1px solid {border};"
            f"max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
            f"{html.escape(str(columns[c][i]))}</td>"
            for c in col_names
        )
        body.append(f"<tr style='background:{bg};'>{tds}</tr>")

    st.markdown(
        f"<div style='max-height:{max_height}px;overflow:auto;border:1px solid {border};"
        f"border-radius:8px;margin-top:4px;'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>"
        f"<thead><tr>{ths}</tr></thead><tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_entity_table(state: ResearchState, max_rows: int = 20) -> None:
    """Render a compact entity table (Entity / Category / Count) for the NER expander."""
    entities = state.entities[:max_rows]
    if not entities:
        st.caption("No entities extracted.")
        return
    render_table({
        "Entity":   [e.text     for e in entities],
        "Category": [e.category for e in entities],
        "Count":    [e.count    for e in entities],
    })
