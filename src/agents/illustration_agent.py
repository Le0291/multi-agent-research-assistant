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
_OPENAI_IMAGE_MODEL   = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
_OPENAI_IMAGE_QUALITY = os.environ.get("OPENAI_IMAGE_QUALITY", "medium")


# ═══════════════════════════════════════════════════════════════════════════ #
#  OpenAI image generator                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

def _make_openai_image(prompt: str, index: int, topic: str) -> str:
    """
    Generate one research infographic via OpenAI gpt-image-1.
    Returns the saved file path.  Raises on any API failure.
    """
    import openai  # noqa: PLC0415
    import base64  # noqa: PLC0415

    client = openai.OpenAI(api_key=config.openai_api_key, timeout=120.0)

    # Wrap the prompt with strong typography rules for readable output
    enhanced = (
        f"Professional academic research infographic about '{topic}': {prompt}. "
        "CRITICAL TYPOGRAPHY RULES: "
        "All text must be LARGE (minimum 24pt), BOLD, white or bright-colored "
        "on a dark background, clean sans-serif font (Arial or Helvetica). "
        "Keep every label to 1-3 words — NO long sentences inside the image. "
        "Use icons, arrows, and color-coded shapes to convey information visually. "
        "Dark navy (#0d1b2a) background. Accent colors: #92ccff, #61de8a, #ffba4b. "
        "Style: clean, modern, scientific poster. NOT abstract art. NOT decorative."
    )

    response = client.images.generate(
        model=_OPENAI_IMAGE_MODEL,
        prompt=enhanced[:1000],
        size="1024x1024",
        quality=_OPENAI_IMAGE_QUALITY,
        n=1,
    )

    # Handle both response shapes: b64_json (gpt-image-1) and url (dall-e-3)
    data0  = response.data[0]
    b64    = getattr(data0, "b64_json", None)
    url    = getattr(data0, "url",      None)

    if b64:
        img_bytes = base64.b64decode(b64)
    elif url:
        import httpx  # noqa: PLC0415
        img_bytes = httpx.get(url, timeout=30).content
    else:
        raise RuntimeError("OpenAI response contained neither b64_json nor url.")

    filename = f"dalle_{index + 1}_{topic[:20].replace(' ', '_')}.png"
    filepath = config.images_dir / filename
    filepath.write_bytes(img_bytes)
    logger.info("Saved OpenAI image (%s): %s", _OPENAI_IMAGE_MODEL, filepath)
    return str(filepath)


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
            f"Research analysis dashboard for '{topic}'. "
            "Three panels side by side: "
            "(1) Horizontal bar chart — bars labeled 'Papers', 'Web', 'News', 'Preprints'. "
            "Bold white axis labels. "
            "(2) Pie chart — slices labeled 'High', 'Medium', 'Low' relevance. "
            "Bright slice colors with white bold percentages. "
            "(3) Simple timeline — milestone dots with 1-2 word labels. "
            "Dark charcoal background. All text: white, bold, minimum 20pt. "
            "Subtle panel borders. Title: large white bold at top."
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

    for i, prompt in enumerate(prompts):
        try:
            path = _make_openai_image(prompt, i, state.topic)
            illustration_paths.append(path)
            logger.info("Figure %d: ✓ saved to %s", i + 1, path)
        except Exception as exc:
            err = f"Figure {i + 1} generation failed: {type(exc).__name__}: {exc}"
            logger.error(err)
            state.errors.append(err)

    logger.info(
        "Illustration Agent: %d/%d figures generated.",
        len(illustration_paths), len(prompts),
    )
    return {
        "illustrations": illustration_paths,
        "status": "writing",
    }
