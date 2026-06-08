"""
illustration_agent.py — Academic figure generator.

Attempts image generation in this order:
  1. DALL-E 3 via OpenAI API (if OPENAI_API_KEY is set).
  2. Data-driven Matplotlib charts built from actual pipeline state
     (entity distribution, source breakdown, theme map) — no API needed.

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

# ── Shared dark-theme palette (matches the app's Stitch dark tokens) ──────── #
_BG      = "#101418"
_SURFACE = "#1c2024"
_BORDER  = "#3f4850"
_TEXT    = "#e0e3e8"
_MUTED   = "#bfc7d2"
_PRIMARY = "#92ccff"
_GREEN   = "#61de8a"
_AMBER   = "#ffba4b"
_RED     = "#ffb4ab"
_PURPLE  = "#c9a4ff"
_COLORS  = [_PRIMARY, _GREEN, _AMBER, _RED, _PURPLE, "#ff9f7f", "#7fd4ff", "#a8ffb8"]


# ═══════════════════════════════════════════════════════════════════════════ #
#  Chart builders                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

def _apply_dark_style(fig, ax_list: list) -> None:
    """Apply the shared dark theme to a figure and its axes."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig.patch.set_facecolor(_BG)
    for ax in ax_list:
        ax.set_facecolor(_SURFACE)
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        ax.title.set_color(_PRIMARY)
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)


def _save(fig, filename: str) -> str:
    """Save figure and return the path string."""
    import matplotlib.pyplot as plt  # noqa: PLC0415
    filepath = config.images_dir / filename
    plt.tight_layout(pad=1.5)
    plt.savefig(str(filepath), dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    logger.info("Saved figure: %s", filepath)
    return str(filepath)


# ── Chart 1: Entity category distribution ──────────────────────────────────
def _chart_entity_distribution(state: ResearchState, topic: str) -> str:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    entities = state.entities or []
    if not entities:
        return _chart_topic_breakdown(topic, 0)

    counts = Counter(e.category.lower() for e in entities)
    labels = [k.title() for k in counts.keys()]
    values = list(counts.values())

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5))
    _apply_dark_style(fig, [ax_bar, ax_pie])

    # Bar chart
    bar_colors = _COLORS[:len(labels)]
    bars = ax_bar.barh(labels, values, color=bar_colors, edgecolor=_BORDER, height=0.6)
    ax_bar.set_xlabel("Entity Count", color=_TEXT)
    ax_bar.set_title("Entities by Category", color=_PRIMARY, fontsize=12, pad=10)
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", ha="left", color=_TEXT, fontsize=9)
    ax_bar.set_xlim(0, max(values) * 1.2)

    # Pie chart
    wedges, texts, autotexts = ax_pie.pie(
        values, labels=labels, colors=bar_colors,
        autopct="%1.0f%%", startangle=140,
        wedgeprops={"edgecolor": _BORDER, "linewidth": 1.2},
        pctdistance=0.75,
    )
    for t in texts:
        t.set_color(_TEXT); t.set_fontsize(9)
    for at in autotexts:
        at.set_color(_BG); at.set_fontsize(8); at.set_fontweight("bold")
    ax_pie.set_title("Category Distribution", color=_PRIMARY, fontsize=12, pad=10)

    fig.suptitle(f"Named Entity Analysis — {topic[:50]}", color=_TEXT,
                 fontsize=13, fontweight="bold", y=1.01)

    slug = topic[:20].replace(" ", "_")
    return _save(fig, f"figure_1_entity_distribution_{slug}.png")


# ── Chart 2: Source breakdown ───────────────────────────────────────────────
def _chart_source_breakdown(state: ResearchState, topic: str) -> str:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    sources = state.classified_sources or state.raw_sources or []
    if len(sources) < 2:
        return _chart_topic_breakdown(topic, 1)

    # Source type distribution
    type_counts = Counter(getattr(s, "source_type", "other") for s in sources)
    # Relevance score histogram
    scores = [getattr(s, "relevance_score", 5.0) for s in sources if hasattr(s, "relevance_score")]

    fig, (ax_type, ax_score) = plt.subplots(1, 2, figsize=(12, 5))
    _apply_dark_style(fig, [ax_type, ax_score])

    # Source type bar
    labels = [k.replace("_", " ").title() for k in type_counts.keys()]
    values = list(type_counts.values())
    ax_type.bar(labels, values, color=_COLORS[:len(labels)], edgecolor=_BORDER, width=0.6)
    ax_type.set_title("Sources by Type", color=_PRIMARY, fontsize=12, pad=10)
    ax_type.set_ylabel("Count", color=_TEXT)
    plt.setp(ax_type.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(values):
        ax_type.text(i, v + 0.05, str(v), ha="center", color=_TEXT, fontsize=9)

    # Relevance score histogram
    if scores:
        n, bins, patches = ax_score.hist(scores, bins=8, color=_PRIMARY,
                                          edgecolor=_BORDER, alpha=0.85)
        ax_score.set_title("Relevance Score Distribution", color=_PRIMARY, fontsize=12, pad=10)
        ax_score.set_xlabel("Score (0-10)", color=_TEXT)
        ax_score.set_ylabel("Source Count", color=_TEXT)
        avg = sum(scores) / len(scores)
        ax_score.axvline(avg, color=_AMBER, linestyle="--", linewidth=1.5,
                         label=f"Avg: {avg:.1f}")
        ax_score.legend(facecolor=_SURFACE, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=9)

    fig.suptitle(f"Source Quality Analysis — {topic[:50]}", color=_TEXT,
                 fontsize=13, fontweight="bold", y=1.01)

    slug = topic[:20].replace(" ", "_")
    return _save(fig, f"figure_2_source_analysis_{slug}.png")


# ── Chart 3: Theme map ──────────────────────────────────────────────────────
def _chart_theme_map(state: ResearchState, topic: str) -> str:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    themes       = state.themes[:8]       if state.themes       else []
    contradictions = state.contradictions[:4] if state.contradictions else []

    if not themes:
        return _chart_topic_breakdown(topic, 2)

    fig, ax = plt.subplots(figsize=(12, 7))
    _apply_dark_style(fig, [ax])
    ax.axis("off")

    # Draw central topic node
    cx, cy = 0.5, 0.5
    topic_short = topic[:30]
    ax.add_patch(plt.Circle((cx, cy), 0.09, color=_PRIMARY, zorder=3, transform=ax.transAxes))
    ax.text(cx, cy, topic_short, ha="center", va="center", fontsize=8,
            fontweight="bold", color=_BG, transform=ax.transAxes, zorder=4,
            wrap=True)

    # Place theme nodes in a circle around the center
    n = len(themes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 0.35
    for i, (theme, angle) in enumerate(zip(themes, angles)):
        tx = cx + r * np.cos(angle)
        ty = cy + r * np.sin(angle)
        color = _COLORS[i % len(_COLORS)]

        # Line from center to node
        ax.annotate("", xy=(tx, ty), xytext=(cx, cy),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", color=_BORDER, lw=1.2))

        # Node
        ax.add_patch(plt.Circle((tx, ty), 0.055, color=color, alpha=0.85,
                                 zorder=3, transform=ax.transAxes))
        wrapped = textwrap.fill(theme[:40], 18)
        ax.text(tx, ty, wrapped, ha="center", va="center",
                fontsize=7, color=_BG, transform=ax.transAxes, zorder=4,
                fontweight="bold")

    # Contradictions as a legend
    if contradictions:
        legend_text = "⚠ Contradictions:\n" + "\n".join(
            f"• {c[:55]}" for c in contradictions
        )
        ax.text(0.01, 0.01, legend_text, transform=ax.transAxes,
                fontsize=7.5, color=_AMBER, va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=_SURFACE,
                          edgecolor=_AMBER, alpha=0.9))

    ax.set_title(f"Research Theme Map — {topic[:50]}",
                 color=_PRIMARY, fontsize=13, fontweight="bold", pad=12)

    slug = topic[:20].replace(" ", "_")
    return _save(fig, f"figure_3_theme_map_{slug}.png")


# ── Fallback: styled topic breakdown (no state data available) ──────────────
def _chart_topic_breakdown(topic: str, index: int) -> str:
    """Generic styled chart when no pipeline state data is available."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.patches as mpatches  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_style(fig, [ax])
    ax.axis("off")

    words = topic.split()
    chunks = [" ".join(words[i:i+3]) for i in range(0, min(len(words), 9), 3)] or [topic]
    colors = _COLORS[:len(chunks)]

    for i, (chunk, color) in enumerate(zip(chunks, colors)):
        x = 0.15 + (i % 3) * 0.32
        y = 0.65 - (i // 3) * 0.28
        rect = mpatches.FancyBboxPatch((x - 0.13, y - 0.09), 0.26, 0.18,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor=_BORDER,
                                        alpha=0.85, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, y, chunk, ha="center", va="center", fontsize=9,
                color=_BG, fontweight="bold", transform=ax.transAxes)

    ax.set_title(f"Figure {index + 1}: {topic[:60]}", color=_PRIMARY,
                 fontsize=12, fontweight="bold", pad=10)

    slug = topic[:20].replace(" ", "_")
    return _save(fig, f"figure_{index + 1}_{slug}.png")


# ═══════════════════════════════════════════════════════════════════════════ #
#  DALL-E 3 generator (optional, requires OPENAI_API_KEY)                    #
# ═══════════════════════════════════════════════════════════════════════════ #

def _make_dalle_image(prompt: str, index: int, topic: str) -> str:
    """Generate an image with DALL-E 3. Raises on any failure."""
    import openai  # noqa: PLC0415
    import httpx   # noqa: PLC0415

    client = openai.OpenAI(api_key=config.openai_api_key)
    # Keep prompt detailed and data-driven — avoid 'minimal' which causes blank output
    enhanced = (
        f"Detailed academic research infographic about '{topic}': {prompt}. "
        "Style: professional data visualization, rich in information, "
        "dark navy/charcoal background, vibrant accent colors for labels and nodes, "
        "clear typography, suitable for a scientific research paper. "
        "Include relevant diagrams, charts, or concept maps — NOT abstract art, "
        "NOT a simple logo, NOT minimalist."
    )
    response = client.images.generate(
        model="dall-e-3",
        prompt=enhanced[:1000],
        size="1024x1024",
        quality="hd",        # hd for better detail
        n=1,
    )
    img_bytes = httpx.get(response.data[0].url, timeout=30).content
    filename  = f"dalle_{index + 1}_{topic[:20].replace(' ', '_')}.png"
    filepath  = config.images_dir / filename
    filepath.write_bytes(img_bytes)
    logger.info("Saved DALL-E image: %s", filepath)
    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════ #
#  LangGraph node                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

# Map of (index → chart builder) used when DALL-E is unavailable
_CHART_BUILDERS = [
    lambda s, t: _chart_entity_distribution(s, t),
    lambda s, t: _chart_source_breakdown(s, t),
    lambda s, t: _chart_theme_map(s, t),
]

def _build_dalle_prompts(topic: str, state: "ResearchState") -> list[str]:
    """
    Build 3 topic-aware DALL-E prompts from actual pipeline data.
    Always uses real state data — never the writer-generated image_prompts
    which are usually vague or off-topic.
    """
    # Gather real entity names for Figure 1
    top_entities = ", ".join(e.text for e in state.entities[:8]) if state.entities else topic
    # Gather real themes for Figure 3
    top_themes = "; ".join(state.themes[:5]) if state.themes else f"key aspects of {topic}"

    return [
        # Figure 1 — Knowledge map
        (f"Detailed knowledge graph infographic for the research topic '{topic}'. "
         f"Show key entities and concepts as interconnected labeled nodes: {top_entities}. "
         "Use circles and arrows to show relationships, color-coded by category "
         "(blue for technology, green for organizations, amber for concepts). "
         "Dark background, white labels, professional academic style."),

        # Figure 2 — Data analysis dashboard
        (f"Academic research analysis dashboard for '{topic}'. "
         "Show: (1) a bar chart of source types (academic papers, websites, news), "
         "(2) a relevance score distribution histogram, "
         "(3) a timeline of information sources. "
         "Multi-panel layout, dark background, vibrant chart colors, clear axis labels."),

        # Figure 3 — Theme mind map
        (f"Radial mind map for research on '{topic}'. "
         f"Central node labeled '{topic}' connected to key themes: {top_themes}. "
         "Each theme has 2-3 sub-nodes showing related concepts. "
         "Use a dark navy background, color-coded branches, bold readable text, "
         "professional academic poster style."),
    ]


def illustration_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: generate 3 data-driven figures for the report.

    Priority:
      1. DALL-E 3 with curated topic-aware prompts (if OPENAI_API_KEY set).
         NOTE: state.image_prompts (from writer) are intentionally ignored —
         they are usually too vague.  We build richer prompts from actual
         pipeline data (entities, themes, sources).
      2. Data-driven Matplotlib charts built from pipeline state (always works).
    """
    illustration_paths: list[str] = []

    # Build topic-aware prompts once (uses real entities + themes from state)
    dalle_prompts = _build_dalle_prompts(state.topic, state)

    for i in range(3):  # Always produce 3 figures
        path = None

        # ── Try DALL-E first ──────────────────────────────────────────────
        if config.openai_api_key:
            prompt = dalle_prompts[i]   # Always use our curated prompts
            try:
                path = _make_dalle_image(prompt, i, state.topic)
                logger.info("Figure %d: DALL-E ✓", i + 1)
            except Exception as exc:
                logger.warning("DALL-E failed for figure %d: %s — using matplotlib.", i + 1, exc)

        # ── Fall back to data-driven matplotlib chart ─────────────────────
        if not path:
            try:
                builder = _CHART_BUILDERS[i]
                path = builder(state, state.topic)
                logger.info("Figure %d: matplotlib chart ✓", i + 1)
            except Exception as exc:
                logger.error("Figure %d generation failed: %s", i + 1, exc)
                state.errors.append(f"Figure {i + 1} generation failed: {exc}")
                continue

        illustration_paths.append(path)

    logger.info("Illustration Agent: generated %d figures.", len(illustration_paths))
    return {
        "illustrations": illustration_paths,
        "status": "writing",
    }
