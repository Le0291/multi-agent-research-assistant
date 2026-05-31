"""
illustration_agent.py — Academic figure generator.

Attempts image generation in this order:
  1. DALL-E 3 via OpenAI API (if OPENAI_API_KEY is set).
  2. Matplotlib placeholder diagram (always works, no API required).
  3. Mermaid code blocks embedded directly into the Markdown report.

All generated files are saved to generated_images/.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any

from src.config import config
from src.state import ResearchState

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Matplotlib placeholder generator                                            #
# --------------------------------------------------------------------------- #
def _make_matplotlib_figure(prompt: str, index: int, topic: str) -> str:
    """
    Generate a placeholder academic figure using Matplotlib.

    Returns the file path (relative to project root) of the saved PNG.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")  # Non-interactive backend — safe in server environments
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    # Wrap long prompt text for display
    wrapped = textwrap.fill(prompt, width=60)

    ax.text(
        0.5, 0.55, wrapped,
        ha="center", va="center",
        fontsize=12, color="#333333",
        transform=ax.transAxes,
        wrap=True,
    )
    ax.text(
        0.5, 0.2,
        f"Figure {index + 1}: {topic}",
        ha="center", va="center",
        fontsize=10, color="#666666", style="italic",
        transform=ax.transAxes,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Add a border rectangle
    rect = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.01",
        linewidth=2, edgecolor="#3498db", facecolor="none",
        transform=ax.transAxes,
    )
    ax.add_patch(rect)

    plt.title(f"[Placeholder] {prompt[:50]}", fontsize=9, color="#999999")
    plt.tight_layout()

    filename = f"figure_{index + 1}_{topic[:20].replace(' ', '_')}.png"
    filepath = config.images_dir / filename
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved matplotlib figure: %s", filepath)
    return str(filepath)


# --------------------------------------------------------------------------- #
#  DALL-E 3 generator (optional)                                               #
# --------------------------------------------------------------------------- #
def _make_dalle_image(prompt: str, index: int, topic: str) -> str:
    """
    Generate an image with DALL-E 3.  Raises ImportError / API errors if
    OpenAI is unavailable — callers should fall back to matplotlib.
    """
    import openai  # noqa: PLC0415

    client = openai.OpenAI(api_key=config.openai_api_key)
    enhanced_prompt = (
        f"Academic research illustration: {prompt}.  "
        "Clean, professional, white background, minimalist style suitable for a research paper."
    )
    response = client.images.generate(
        model="dall-e-3",
        prompt=enhanced_prompt[:1000],
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url

    # Download and save
    import httpx  # noqa: PLC0415
    img_bytes = httpx.get(image_url, timeout=30).content
    filename = f"dalle_{index + 1}_{topic[:20].replace(' ', '_')}.png"
    filepath = config.images_dir / filename
    filepath.write_bytes(img_bytes)
    logger.info("Saved DALL-E image: %s", filepath)
    return str(filepath)


# --------------------------------------------------------------------------- #
#  Node                                                                        #
# --------------------------------------------------------------------------- #
def illustration_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: generate figures for the report.

    Input state fields used : image_prompts, topic
    Output state fields set  : illustrations, status
    """
    prompts = state.image_prompts or [
        f"System architecture diagram of {state.topic}",
        f"Performance comparison chart related to {state.topic}",
    ]

    illustration_paths: list[str] = []

    for i, prompt in enumerate(prompts[:3]):  # Max 3 figures per report
        path = None

        # Try DALL-E first if key is available
        if config.openai_api_key:
            try:
                path = _make_dalle_image(prompt, i, state.topic)
            except Exception as exc:
                logger.warning("DALL-E failed: %s — using matplotlib.", exc)

        # Always fall back to matplotlib
        if not path:
            try:
                path = _make_matplotlib_figure(prompt, i, state.topic)
            except Exception as exc:
                logger.error("Matplotlib figure failed: %s", exc)
                state.errors.append(f"Figure {i + 1} generation failed: {exc}")
                continue

        illustration_paths.append(path)

    logger.info("Illustration Agent: generated %d figures.", len(illustration_paths))
    return {
        "illustrations": illustration_paths,
        "status": "writing",
    }
