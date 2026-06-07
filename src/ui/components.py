"""
src/ui/components.py — Shared UI components used across pipeline and playground.

Functions:
  render_file_upload(prefix)   Inline file-upload expander (reads/writes session_state)
  render_entity_table(state)   Compact entity dataframe inside Full Pipeline expanders
"""

from __future__ import annotations

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


def render_entity_table(state: ResearchState, max_rows: int = 20) -> None:
    """
    Render a compact entity dataframe from a ResearchState object.
    Used inside the NER expander in the Full Pipeline progress view.
    """
    entities = state.entities[:max_rows]
    if not entities:
        st.caption("No entities extracted.")
        return
    st.dataframe(
        {
            "Entity":   [e.text     for e in entities],
            "Category": [e.category for e in entities],
            "Count":    [e.count    for e in entities],
        },
        use_container_width=True,
    )
