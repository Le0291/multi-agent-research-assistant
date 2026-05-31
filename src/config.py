"""
config.py — Centralised configuration for the Multi-Agent Research Assistant.

All settings are read from environment variables so no secrets are ever
hard-coded.  Call load_config() once at startup; every other module imports
the resulting Config dataclass.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists (harmless if it does not)
load_dotenv()

logger = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "generated_images"
CHROMA_DIR = BASE_DIR / ".chroma_db"

# Ensure output directories exist at import time
for _d in (REPORTS_DIR, IMAGES_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """All runtime settings, read once from the environment."""

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"))
    anthropic_max_tokens: int = field(default_factory=lambda: int(os.environ.get("ANTHROPIC_MAX_TOKENS", "8192")))
    anthropic_temperature: float = field(default_factory=lambda: float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.3")))

    # ── Search ────────────────────────────────────────────────────────────────
    tavily_api_key: str = field(default_factory=lambda: os.environ.get("TAVILY_API_KEY", ""))
    brave_api_key: str = field(default_factory=lambda: os.environ.get("BRAVE_API_KEY", ""))

    # ── Optional Image Gen ────────────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))

    # ── Pipeline Tuning ───────────────────────────────────────────────────────
    max_sources: int = field(default_factory=lambda: int(os.environ.get("MAX_SOURCES", "15")))
    min_sources: int = field(default_factory=lambda: int(os.environ.get("MIN_SOURCES", "10")))
    max_revisions: int = field(default_factory=lambda: int(os.environ.get("MAX_REVISIONS", "3")))
    critic_pass_score: int = field(default_factory=lambda: int(os.environ.get("CRITIC_PASS_SCORE", "7")))

    # ── Paths (expose to callers) ─────────────────────────────────────────────
    reports_dir: Path = REPORTS_DIR
    images_dir: Path = IMAGES_DIR
    chroma_dir: Path = CHROMA_DIR

    def validate(self) -> None:
        """Raise ValueError if required keys are missing."""
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set.  "
                "Copy .env.example → .env and fill in your key."
            )
        if not self.tavily_api_key and not self.brave_api_key:
            logger.warning(
                "Neither TAVILY_API_KEY nor BRAVE_API_KEY is set. "
                "Search will fall back to mock results."
            )


# Singleton instance — import this everywhere else
config = Config()
