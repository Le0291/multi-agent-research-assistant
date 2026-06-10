"""
illustration_agent.py — Academic figure generator.

Generates 3 research infographic figures using OpenAI gpt-image-1.
Requires OPENAI_API_KEY to be set.  If the API call fails, a clear
error is logged and the figure slot is skipped (no matplotlib fallback).

All generated files are saved to generated_images/.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.config import config
from src.state import ResearchState

logger = logging.getLogger(__name__)

# ── OpenAI image model settings ────────────────────────────────────────────── #
# Model preference: gpt-image-1 → dall-e-3 (automatic fallback)
# Override via OPENAI_IMAGE_MODEL env var to pin a specific model.
_OPENAI_IMAGE_MODEL   = os.environ.get("OPENAI_IMAGE_MODEL", "")   # empty = auto-detect
_OPENAI_IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "")  # empty = per-model default

# Models tried in order until one succeeds
_MODEL_CASCADE = ["gpt-image-1", "dall-e-3", "dall-e-2"]

# Quality values per model (APIs differ)
_MODEL_QUALITY = {
    "gpt-image-1": "medium",   # low / medium / high / auto
    "dall-e-3":    "standard", # standard / hd
    "dall-e-2":    None,       # no quality param
}


# ═══════════════════════════════════════════════════════════════════════════ #
#  OpenAI image generator                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def _call_model(client: "openai.OpenAI", model: str, prompt: str) -> bytes:
    """
    Call a specific OpenAI image model and return raw image bytes.
    Raises openai.APIError on failure so the caller can try the next model.
    """
    import base64   # noqa: PLC0415

    # Quality param differs per model
    if _OPENAI_IMAGE_QUALITY:
        quality = _OPENAI_IMAGE_QUALITY          # user override
    else:
        quality = _MODEL_QUALITY.get(model)      # per-model default

    kwargs: dict = dict(model=model, prompt=prompt[:1000], size="1024x1024", n=1)
    if quality:
        kwargs["quality"] = quality

    response = client.images.generate(**kwargs)
    data0    = response.data[0]
    b64      = getattr(data0, "b64_json", None)
    url      = getattr(data0, "url",      None)

    if b64:
        return base64.b64decode(b64)
    if url:
        import httpx   # noqa: PLC0415
        return httpx.get(url, timeout=30).content
    raise RuntimeError("Response contained neither b64_json nor url.")


def _make_openai_image(prompt: str, index: int, topic: str) -> tuple[str, str]:
    """
    Generate one research infographic using OpenAI, trying models in cascade:
      gpt-image-1 → dall-e-3 → dall-e-2

    Returns (file_path, model_used).  Raises RuntimeError if all models fail.
    """
    import openai   # noqa: PLC0415

    client = openai.OpenAI(api_key=config.openai_api_key, timeout=120.0)

    enhanced = (
        f"Professional academic research infographic about '{topic}': {prompt}. "
        "CRITICAL TYPOGRAPHY RULES: "
        "All text LARGE (min 24pt), BOLD, white or bright-colored on dark background, "
        "clean sans-serif font (Arial/Helvetica). "
        "Keep every label to 1-3 words — NO long sentences inside the image. "
        "Use icons, arrows, color-coded shapes. "
        "Dark navy (#0d1b2a) background. Accents: #92ccff #61de8a #ffba4b. "
        "Clean, modern, scientific poster. NOT abstract art."
    )

    # If user pinned a model, use only that one
    cascade = [_OPENAI_IMAGE_MODEL] if _OPENAI_IMAGE_MODEL else _MODEL_CASCADE
    last_exc: Exception | None = None

    for model in cascade:
        try:
            img_bytes = _call_model(client, model, enhanced)
            filename  = f"dalle_{index + 1}_{topic[:20].replace(' ', '_')}.png"
            filepath  = config.images_dir / filename
            filepath.write_bytes(img_bytes)
            logger.info("Figure %d: saved via %s → %s", index + 1, model, filepath)
            return str(filepath), model
        except Exception as exc:
            logger.warning("Model %s failed for figure %d: %s", model, index + 1, exc)
            last_exc = exc

    raise RuntimeError(
        f"All image models failed. Last error: {last_exc}"
    )


# ═══════════════════════════════════════════════════════════════════════════ #
#  Prompt builder                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def _build_prompts(topic: str, state: ResearchState) -> list[str]:
    """Build 3 data-driven prompts from real pipeline state."""
    top_entities = (
        ", ".join(e.text for e in state.entities[:8])
        if state.entities else topic
    )
    top_themes = (
        "; ".join(state.themes[:5])
        if state.themes else f"key aspects of {topic}"
    )

    return [
        # Figure 1 — Knowledge graph
        (
            f"Knowledge graph for '{topic}'. "
            f"Show 6-8 concept nodes as colored circles. "
            f"Node labels (1-2 words each): {top_entities[:120]}. "
            "Connect nodes with short arrow labels: 'uses', 'detects', 'improves'. "
            "Each node: solid fill, BLACK bold text inside for maximum contrast. "
            "Background: dark navy. Arrow lines: white. "
            "Colors: blue, green, amber, red — one per category. "
            "Title at top in large white bold text. Spacious, no overlapping text."
        ),

        # Figure 2 — Analysis dashboard
        (
            f"Clean research analysis infographic for '{topic}'. "
            "Layout: large title at top, then TWO wide rows below. "
            "TOP ROW — horizontal bar chart, 4 bars with INSIDE labels: "
            "'Papers' (blue, longest), 'Web Sources' (green), 'News' (amber), 'Preprints' (teal). "
            "Each bar has its percentage value written at the end in large white bold text. "
            "BOTTOM ROW — pie chart on left: exactly 3 equal slices labeled "
            "'High 50%' (bright green), 'Medium 30%' (amber), 'Low 20%' (red). "
            "Labels OUTSIDE each slice with leader lines. "
            "On the right: numbered list of 4 key findings, each one line, white text. "
            "Background: dark navy #0d1b2a. All fonts: white, bold, minimum 22pt. "
            "High contrast, plenty of whitespace, no overlapping text."
        ),

        # Figure 3 — Theme mind map
        (
            f"Radial mind map for '{topic}'. "
            f"Center circle: bold white text '{topic[:25]}'. "
            f"5 branch themes radiating outward: {top_themes[:150]}. "
            "Each theme: rounded rectangle, unique bright color, bold white 2-3 word label. "
            "Connecting lines: smooth curves, white/light gray. "
            "Sub-nodes (2 per theme): smaller circles, same color family, 1-2 word labels. "
            "Dark navy background. All labels: bold, white, clearly legible. No crowding."
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════ #
#  LangGraph node                                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

def illustration_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: generate 3 research infographic figures via OpenAI gpt-image-1.

    Requires OPENAI_API_KEY.  Each failed figure is logged as an error and
    skipped — no matplotlib fallback.
    """
    if not config.openai_api_key:
        msg = (
            "Illustration Agent: OPENAI_API_KEY is not set. "
            "Add it to your .env file to enable figure generation."
        )
        logger.error(msg)
        state.errors.append(msg)
        return {"illustrations": [], "status": "writing"}

    logger.info(
        "Illustration Agent: generating 3 figures with %s (%s quality).",
        _OPENAI_IMAGE_MODEL, _OPENAI_IMAGE_QUALITY,
    )

    prompts = _build_prompts(state.topic, state)
    illustration_paths: list[str] = []
    models_used: list[str] = []

    for i, prompt in enumerate(prompts):
        try:
            path, model_used = _make_openai_image(prompt, i, state.topic)
            illustration_paths.append(path)
            models_used.append(model_used)
        except Exception as exc:
            err = f"Figure {i + 1} generation failed: {type(exc).__name__}: {exc}"
            logger.error(err)
            state.errors.append(err)

    if illustration_paths:
        used = ", ".join(sorted(set(models_used)))
        logger.info(
            "Illustration Agent: %d/%d figures generated (model: %s).",
            len(illustration_paths), len(prompts), used,
        )
        # Store model info so UI can display it
        state.errors = [
            e for e in state.errors
            if "Figure" not in e or "failed" in e  # keep only real errors
        ]
    else:
        logger.error("Illustration Agent: all figure generation attempts failed.")

    return {
        "illustrations": illustration_paths,
        "status": "writing",
    }
