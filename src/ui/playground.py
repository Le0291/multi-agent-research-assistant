"""
src/ui/playground.py — Agent Playground UI (Mode 2: single-agent isolation).

Each playground mode shows only the inputs it needs and only the outputs it
produces.  Dependency resolution (e.g. running Orchestrator before Research)
is handled transparently by src/router.py.
"""

from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from typing import Any

import streamlit as st

from src.config import config
from src.router import MODE_DESCRIPTIONS, MODE_INPUTS, route
from src.utils.cost_tracker import format_cost_table
from src.ui.components import render_file_upload, render_table

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Theme-aware entity-chip colour palettes (Stitch tokens)
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (background, foreground, border)
_CAT_DARK: dict[str, tuple[str, str, str]] = {
    "person":       ("#3d1212", "#ffb3b3", "rgba(255,179,179,0.30)"),
    "organization": ("#0d1e3a", "#92ccff", "rgba(146,204,255,0.30)"),
    "location":     ("#0d2d1a", "#61de8a", "rgba(97,222,138,0.30)"),
    "technology":   ("#3d2a00", "#ffba4b", "rgba(255,186,75,0.30)"),
    "concept":      ("#2a0d3a", "#c9a4ff", "rgba(201,164,255,0.30)"),
    "date":         ("#0d3330", "#61de8a", "rgba(97,222,138,0.30)"),
}
_CAT_LIGHT: dict[str, tuple[str, str, str]] = {
    "person":       ("#fde8e8", "#7b1c1c", "rgba(123,28,28,0.20)"),
    "organization": ("#e8eeff", "#1c357a", "rgba(28,53,122,0.20)"),
    "location":     ("#e8f7e8", "#1a5c1a", "rgba(26,92,26,0.20)"),
    "technology":   ("#fff5e0", "#7a4d00", "rgba(122,77,0,0.20)"),
    "concept":      ("#f5e8ff", "#4c1a7a", "rgba(76,26,122,0.20)"),
    "date":         ("#e0faf4", "#0e5c44", "rgba(14,92,68,0.20)"),
}
_CAT_ICON: dict[str, str] = {
    "person": "👤", "organization": "🏢",
    "location": "📍", "technology": "💡",
    "concept": "🧠", "date": "📅",
}
_DEFAULT_CHIP = ("#1c2024", "#e0e3e8", "#3f4850")


def _cat_style(theme_dark: bool) -> dict[str, tuple[str, str, str]]:
    return _CAT_DARK if theme_dark else _CAT_LIGHT


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_playground_ui(mode: str) -> None:
    """Render the Agent Playground interface for a single-agent mode."""
    st.markdown(
        '<span class="mode-pill mode-playground">🧪 Agent Playground</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {mode}")
    st.info(MODE_DESCRIPTIONS.get(mode, ""), icon="ℹ️")

    render_file_upload(prefix=f"pg_{mode.replace(' ', '_').lower()}")

    inputs: dict[str, Any] = {}
    accepted = MODE_INPUTS.get(mode, [])

    with st.form(key=f"form_{mode.replace(' ', '_')}"):
        if "topic" in accepted:
            inputs["topic"] = st.text_input(
                "Research Topic",
                placeholder="e.g. Large Language Model training techniques",
            )
        if "text" in accepted:
            inputs["text"] = st.text_area(
                "Paste text for NER analysis  (leave blank to auto-fetch from topic)",
                height=180,
                placeholder="Paste any text here, or leave blank to auto-search the topic above.",
            )
        if "urls" in accepted:
            raw_urls = st.text_area(
                "URLs  (one per line)",
                placeholder="https://example.com\nhttps://another.com",
                height=120,
            )
            inputs["urls"] = [u.strip() for u in raw_urls.splitlines() if u.strip()]
        if "draft" in accepted:
            inputs["draft"] = st.text_area(
                "Paste your draft report for the critic to review",
                height=350,
                placeholder="Paste any Markdown draft here…",
            )
            inputs["topic"] = st.text_input(
                "Topic label  (optional — used to give the critic context)", value=""
            )
        if "image_prompts" in accepted and mode == "Illustration Only":
            raw_prompts = st.text_area(
                "Image prompts  (one per line, or leave blank to auto-derive from topic)",
                placeholder="Architecture diagram of Transformer model",
                height=120,
            )
            inputs["image_prompts"] = [p.strip() for p in raw_prompts.splitlines() if p.strip()]

        notice = _dependency_notice(mode, inputs)
        if notice:
            st.caption(notice)

        run_btn = st.form_submit_button(f"▶ Run {mode}", type="primary", width="stretch")

    if run_btn:
        file_text = st.session_state.get("uploaded_file_text", "")
        file_name = st.session_state.get("uploaded_file_name", "uploaded_file")
        if file_text:
            if mode == "NER Only" and not inputs.get("text", "").strip():
                inputs["text"] = file_text
            elif mode == "Critic Only" and not inputs.get("draft", "").strip():
                inputs["draft"] = file_text
                if not inputs.get("topic", "").strip():
                    inputs["topic"] = file_name
            else:
                inputs["extra_file_text"] = file_text
                inputs["extra_file_name"] = file_name

        with st.spinner(f"Running {mode}…"):
            result = route(mode, inputs)

        if result.get("status") == "error":
            for err in result.get("errors", ["Unknown error"]):
                st.error(err)
        else:
            _render_result(mode, result)

            cost = result.get("cost_metrics")
            if cost and hasattr(cost, "input_tokens") and cost.input_tokens > 0:
                with st.expander("💰 Cost & Token Usage"):
                    st.table(format_cost_table(cost))

            errs = result.get("errors", [])
            if errs:
                with st.expander("⚠️ Non-fatal errors"):
                    for e in errs:
                        st.warning(e)


# ─────────────────────────────────────────────────────────────────────────────
#  Dependency notice helper
# ─────────────────────────────────────────────────────────────────────────────

def _dependency_notice(mode: str, inputs: dict) -> str:
    return {
        "Research Only":
            "ℹ️ Auto-runs: Orchestrator → Research Agent",
        "Classification Only":
            "ℹ️ Classifying the URLs you provided directly." if inputs.get("urls")
            else "ℹ️ Auto-runs: Orchestrator → Research Agent → Classification Agent",
        "NER Only":
            "ℹ️ Running NER on the text you pasted." if inputs.get("text")
            else "ℹ️ Auto-runs: Orchestrator → Research Agent → NER Agent",
        "Browser Only":
            "ℹ️ Visits the URLs you provide — no search needed.",
        "Analysis Only":
            "ℹ️ Auto-runs: Orchestrator → Research → Classification → NER → Analyzer",
        "Writer Only":
            "ℹ️ Auto-runs: full chain up to Illustrations, then Writer.",
        "Critic Only":
            "ℹ️ Reviews the draft you paste — no web search needed.",
        "Illustration Only":
            "ℹ️ Uses your prompts directly." if inputs.get("image_prompts")
            else "ℹ️ Auto-runs: Research → Analysis → Illustration",
    }.get(mode, "")


# ─────────────────────────────────────────────────────────────────────────────
#  Result dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _render_result(mode: str, result: dict) -> None:
    renderers = {
        "Research Only":        _render_research,
        "Classification Only":  _render_classification,
        "NER Only":             _render_ner,
        "Browser Only":         _render_browser,
        "Analysis Only":        _render_analysis,
        "Writer Only":          _render_writer,
        "Critic Only":          _render_critic,
        "Illustration Only":    _render_illustration,
    }
    fn = renderers.get(mode)
    if fn:
        fn(result)
    else:
        st.json(result)


# ─────────────────────────────────────────────────────────────────────────────
#  Individual result renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_research(result: dict) -> None:
    sources = result.get("sources", [])
    st.success(f"✅ Collected **{len(sources)}** sources")

    if result.get("sub_questions"):
        with st.expander("Sub-questions generated"):
            for q in result["sub_questions"]:
                st.markdown(f"- {q}")

    if sources:
        st.markdown("#### Sources")
        for i, src in enumerate(sources, 1):
            with st.expander(f"[{i}] {src.title[:80]} — score {src.relevance_score:.1f}/10"):
                st.markdown(f"**URL:** {src.url}")
                st.markdown(f"**Type:** {src.source_type} · **Domain:** {src.domain}")
                st.markdown(f"**Snippet:** {src.snippet[:400]}")
        st.markdown("#### Summary Table")
        render_table({
            "#":     list(range(1, len(sources) + 1)),
            "Title": [s.title[:60] for s in sources],
            "Score": [s.relevance_score for s in sources],
            "URL":   [s.url for s in sources],
        })

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["#", "Title", "URL", "Score", "Snippet"])
    for i, src in enumerate(sources, 1):
        w.writerow([i, src.title, src.url, src.relevance_score, src.snippet[:200]])
    st.download_button("⬇️ Download Sources CSV", buf.getvalue().encode(),
                       file_name="research_sources.csv", mime="text/csv")


def _render_classification(result: dict) -> None:
    classified = result.get("classified_sources", [])
    st.success(f"✅ Classified **{len(classified)}** sources")
    if classified:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Source Types**")
            for t, c in Counter(s.source_type for s in classified).most_common():
                st.markdown(f"- {t}: **{c}**")
        with col2:
            st.markdown("**Domains**")
            for d, c in Counter(s.domain for s in classified).most_common():
                st.markdown(f"- {d}: **{c}**")
        with col3:
            st.markdown("**Relevance Tiers**")
            for r, c in Counter(s.relevance_tier for s in classified).most_common():
                st.markdown(f"- {r}: **{c}**")
        render_table({
            "Title":     [s.title[:60]       for s in classified],
            "Type":      [s.source_type      for s in classified],
            "Domain":    [s.domain           for s in classified],
            "Relevance": [s.relevance_tier   for s in classified],
            "Score":     [s.relevance_score  for s in classified],
            "URL":       [s.url              for s in classified],
        })
    log = result.get("classification_log", [])
    if log:
        with st.expander(f"🗑️ Discarded sources ({len(log)})"):
            for entry in log:
                st.caption(entry)


def _render_ner(result: dict) -> None:
    entities = result.get("entities", [])
    rels     = result.get("entity_relationships", [])
    st.success(f"✅ Extracted **{len(entities)}** unique entities")

    if not entities:
        st.info("No named entities found in the provided text.")
        return

    is_dark  = st.session_state.get("app_theme", "Dark") == "Dark"
    styles   = _cat_style(is_dark)

    # Bento-grid summary
    cat_counts = Counter(e.category for e in entities)
    total      = len(entities)
    cards_html = ""
    for cat, cnt in cat_counts.most_common():
        bg, fg, border = styles.get(cat, _DEFAULT_CHIP)
        icon = _CAT_ICON.get(cat, "🔹")
        pct  = int(cnt / total * 100) if total else 0
        cards_html += (
            f'<div class="bento-card" style="background:{bg};border-color:{border};">'
            f'<div style="font-size:1.1rem;margin-bottom:4px;">{icon}</div>'
            f'<div class="bc-value" style="color:{fg};">{cnt}</div>'
            f'<div class="bc-label" style="color:{fg};opacity:0.75;">{cat.title()}</div>'
            f'<div style="font-size:0.65rem;opacity:0.5;margin-top:2px;">{pct}%</div>'
            f'</div>'
        )
    st.markdown(f'<div class="bento-grid">{cards_html}</div>', unsafe_allow_html=True)

    # Grouped entity chips
    st.markdown("#### 🏷️ Entities by Category")
    st.caption("Each chip shows the extracted word and its occurrence count (×N).")
    cat_groups: dict = {}
    for e in entities:
        cat_groups.setdefault(e.category, []).append(e)

    for cat in sorted(cat_groups.keys()):
        ents = sorted(cat_groups[cat], key=lambda x: x.count, reverse=True)
        bg, fg, border = styles.get(cat, _DEFAULT_CHIP)
        icon = _CAT_ICON.get(cat, "🔹")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;margin:10px 0 4px 0;'>"
            f"<span style='font-size:15px;'>{icon}</span>"
            f"<b style='font-size:0.92rem;color:{fg};'>{cat.title()}</b>"
            f"<span style='font-size:0.75rem;opacity:0.55;margin-left:4px;'>({len(ents)} entities)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        chips = "".join(
            f'<span style="background:{bg};color:{fg};border:1px solid {border};'
            f'padding:4px 12px;border-radius:14px;font-size:0.84rem;font-weight:500;'
            f'margin:3px 2px;display:inline-block;white-space:nowrap;">'
            f'{e.text}<span style="opacity:0.50;font-size:0.70rem;margin-left:5px;">×{e.count}</span>'
            f'</span>'
            for e in ents[:25]
        )
        st.markdown(f'<div style="margin-bottom:14px;line-height:2.4;">{chips}</div>',
                    unsafe_allow_html=True)

    # Full table
    with st.expander("📋 Full entity table (sortable)"):
        render_table({
            "Entity":      [e.text     for e in entities[:80]],
            "Category":    [e.category for e in entities[:80]],
            "SpaCy Label": [e.label    for e in entities[:80]],
            "Count":       [e.count    for e in entities[:80]],
        })

    # Co-occurrence
    if rels:
        with st.expander(f"🔗 Co-occurrence pairs ({len(rels)})"):
            st.caption("Entities appearing in the same sentence — proxy for semantic relationships.")
            cat_map = {e.text: e.category for e in entities}
            for a, b in rels[:30]:
                bg_a, fg_a, _ = styles.get(cat_map.get(a, "concept"), _DEFAULT_CHIP)
                bg_b, fg_b, _ = styles.get(cat_map.get(b, "concept"), _DEFAULT_CHIP)
                st.markdown(
                    f'<span style="background:{bg_a};color:{fg_a};padding:3px 10px;'
                    f'border-radius:10px;font-size:0.83rem;font-weight:500;">{a}</span>'
                    f'<span style="opacity:0.5;margin:0 6px;">↔</span>'
                    f'<span style="background:{bg_b};color:{fg_b};padding:3px 10px;'
                    f'border-radius:10px;font-size:0.83rem;font-weight:500;">{b}</span>',
                    unsafe_allow_html=True,
                )

    # Download
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Entity", "Category", "SpaCy Label", "Count"])
    for e in entities:
        w.writerow([e.text, e.category, e.label, e.count])
    st.download_button("⬇️ Download Entities CSV", buf.getvalue().encode(),
                       file_name="ner_entities.csv", mime="text/csv")


def _render_browser(result: dict) -> None:
    browser_results = result.get("browser_results", [])
    st.success(f"✅ Visited **{len(browser_results)}** pages")
    for i, br in enumerate(browser_results, 1):
        with st.expander(f"Page {i}: {br.get('title', br.get('url', 'Unknown'))}"):
            st.markdown(f"**URL:** {br.get('url', 'N/A')}")
            if br.get("title"):        st.markdown(f"**Title:** {br['title']}")
            if br.get("description"):  st.markdown(f"**Description:** {br['description']}")
            if br.get("headings"):
                st.markdown("**Headings:**")
                for h in br["headings"]:
                    st.markdown(f"  - {h}")
            if br.get("body_text"):
                st.markdown("**Body preview:**")
                st.text(br["body_text"][:600])
            if br.get("screenshot_path"):
                try:
                    st.image(br["screenshot_path"],
                             caption=f"Screenshot — {br.get('url', '')}",
                             use_column_width=True)
                except Exception:
                    st.caption(f"Screenshot saved: {br['screenshot_path']}")
            if "error" in br:
                st.warning(f"Error: {br['error']}")


def _render_analysis(result: dict) -> None:
    themes         = result.get("themes", [])
    contradictions = result.get("contradictions", [])
    outline        = result.get("outline", [])
    evidence       = result.get("evidence_summary", {})
    prompts        = result.get("image_prompts", [])
    st.success(
        f"✅ {len(themes)} themes · {len(contradictions)} contradictions · "
        f"{result.get('sources_used', '?')} sources analysed"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Key Themes")
        for i, t in enumerate(themes, 1):
            st.markdown(f"**{i}.** {t}")
            if t in evidence:
                st.caption(f"> {evidence[t]}")
    with col2:
        st.markdown("#### 📋 Report Outline")
        for s in outline:
            st.markdown(f"- {s}")
    if contradictions:
        st.markdown("#### ⚠️ Source Contradictions")
        for c in contradictions:
            st.warning(c)
    if prompts:
        st.markdown("#### 🎨 Suggested Image Prompts")
        for p in prompts:
            st.markdown(f"- *{p}*")


def _render_writer(result: dict) -> None:
    draft = result.get("draft", "")
    st.success(
        f"✅ Draft generated — **{len(draft):,}** characters · "
        f"{result.get('sources_used', '?')} sources cited"
    )
    st.markdown("#### 📝 Draft Report")
    st.markdown(draft)
    st.download_button("⬇️ Download Draft (Markdown)", data=draft.encode("utf-8"),
                       file_name="draft_report.md", mime="text/markdown")


def _render_critic(result: dict) -> None:
    score    = result.get("critic_score", 0)
    decision = result.get("critic_decision", "N/A")
    feedback = result.get("critic_feedback", "")
    cls      = "score-high" if score >= config.critic_pass_score else "score-low"
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(
            f'<div class="metric-card"><span class="{cls}">{score}/10</span>'
            f'<br><b>{decision}</b></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("#### Feedback")
        st.markdown(feedback)
    st.caption(
        f"Scores reflect accuracy · completeness · clarity · citations · "
        f"NER · classification · illustrations.  Pass threshold: {config.critic_pass_score}/10."
    )


def _render_illustration(result: dict) -> None:
    illustrations = result.get("illustrations", [])
    prompts       = result.get("image_prompts", [])
    st.success(f"✅ Generated **{len(illustrations)}** figures")
    for i, path in enumerate(illustrations, 1):
        prompt = prompts[i - 1] if i <= len(prompts) else f"Figure {i}"
        st.markdown(f"**Figure {i}:** *{prompt}*")
        try:
            st.image(path, use_column_width=True)
        except Exception:
            st.caption(f"Saved: {path}")
    if not illustrations:
        st.warning("No figures were generated.  Check that matplotlib is installed.")
