"""
src/ui/theme.py — CSS loader and theme injector.

Reads the three CSS files from src/ui/styles/ and injects them into the
Streamlit page with st.markdown.  Call inject_theme_css() once per script
run (before any other UI is rendered).

To change the look of the app edit the CSS files directly — no Python needed:
  src/ui/styles/shared.css   fonts, pills, bento-grid, nav-button shape
  src/ui/styles/dark.css     dark-mode colour tokens
  src/ui/styles/light.css    light-mode colour tokens
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_STYLES_DIR = Path(__file__).parent / "styles"


def _load(filename: str) -> str:
    """Read a CSS file from the styles/ directory."""
    return (_STYLES_DIR / filename).read_text(encoding="utf-8")


def inject_theme_css(theme: str) -> None:
    """
    Inject shared + theme-specific CSS into the page.

    Args:
        theme: "Dark" or "Light"
    """
    shared  = _load("shared.css")
    palette = _load("dark.css") if theme == "Dark" else _load("light.css")
    st.markdown(f"<style>\n{shared}\n{palette}\n</style>", unsafe_allow_html=True)
