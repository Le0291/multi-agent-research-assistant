"""
src/ui/history.py — Disk-persisted run history (last 3 pipeline runs).

History used to live only in st.session_state, which dies on every page
refresh (each refresh starts a brand-new Streamlit session) — so the user's
recent runs vanished on F5.  Entries are now mirrored to reports/history.json,
and the session cache is seeded from that file on first access.

Entries are small JSON-safe dicts (report text, figure paths, scores).
Figure/PDF paths can go stale after a redeploy wipes the filesystem, so
renderers treat them as best-effort (missing files degrade gracefully).
"""

from __future__ import annotations

import json
import logging

import streamlit as st

from src.config import config

logger = logging.getLogger(__name__)

_HISTORY_FILE = config.reports_dir / "history.json"
_MAX_ENTRIES = 3


def _write_to_disk(history: list[dict]) -> None:
    try:
        _HISTORY_FILE.write_text(
            json.dumps(history, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not persist history file: %s", exc)


def _load_from_disk() -> list[dict]:
    try:
        if _HISTORY_FILE.exists():
            data = json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data[:_MAX_ENTRIES]
    except Exception as exc:
        logger.warning("Could not read history file: %s", exc)
    return []


def get_history() -> list[dict]:
    """Session-cached history, seeded from disk on a session's first access."""
    if "pipeline_history" not in st.session_state:
        st.session_state["pipeline_history"] = _load_from_disk()
    return st.session_state["pipeline_history"]


def add_history_entry(entry: dict) -> None:
    """Insert a run at the front, keep the newest 3, persist to disk."""
    history = ([entry] + get_history())[:_MAX_ENTRIES]
    st.session_state["pipeline_history"] = history
    _write_to_disk(history)


def update_entry(index: int, **fields) -> None:
    """Patch an existing entry (e.g. cache a lazily generated pdf_path)."""
    history = get_history()
    if 0 <= index < len(history):
        history[index].update(fields)
        st.session_state["pipeline_history"] = history
        _write_to_disk(history)
