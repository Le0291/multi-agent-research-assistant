"""
src/ui/pipeline.py — Full Pipeline UI (Mode 1: Competition mode).

All 9 LangGraph agents run sequentially.  Results are streamed in real-time
into collapsible expander panels.  A Markdown + PDF report is produced at the end.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import config
from src.graph import build_graph, _dict_to_state, _state_to_full_dict, get_graph_config
from src.state import ResearchState
from src.utils.cost_tracker import format_cost_table
from src.ui.components import render_file_upload, render_entity_table

logger = logging.getLogger(__name__)


# ── LangGraph pipeline (cached so it is built only once per process) ──────────
@st.cache_resource(show_spinner=False)
def _get_pipeline_graph():
    """
    Build and compile the LangGraph StateGraph exactly once.

    Without this cache the graph is rebuilt on every button click,
    which is CPU-intensive and causes an OOM memory spike on the second
    run in memory-constrained environments like Railway.
    """
    logger.info("Building LangGraph pipeline (first run or cache miss)…")
    return build_graph()


# ── Demo topic callback ───────────────────────────────────────────────────────
def _set_demo_topic() -> None:
    """
    on_click callback for the "Demo Topic" button.

    Must run BEFORE the fp_topic text_input is instantiated so that
    setting st.session_state.fp_topic is legal (no StreamlitAPIException).
    """
    st.session_state.fp_topic = "Mixture of Experts in Large Language Models"


# ── Public entry point ────────────────────────────────────────────────────────
def render_full_pipeline_ui() -> None:
    """Render the Full Pipeline interface — topic input, file upload, run button."""
    st.markdown(
        '<span class="mode-pill mode-full">📋 Full Pipeline — Competition Mode</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "All **9 agents** execute in sequence via LangGraph.  "
        "A complete cited Markdown report is produced at the end."
    )

    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Transformer Architecture and Mixture of Experts",
        key="fp_topic",
    )

    render_file_upload(prefix="fp")

    col1, col2 = st.columns([3, 1])
    with col1:
        run_btn = st.button(
            "🚀 Generate Full Report", type="primary",
            use_container_width=True, key="fp_run",
        )
    with col2:
        st.button("💡 Demo Topic", key="fp_demo",
                  use_container_width=True, on_click=_set_demo_topic)

    if run_btn:
        if not topic.strip():
            st.error("Please enter a research topic.")
        else:
            _stream_full_pipeline(topic.strip())

    render_history_section()


# ── Private helpers ───────────────────────────────────────────────────────────
def _stream_full_pipeline(topic: str) -> None:
    """Execute the pipeline and stream each agent's result into expander panels."""
    progress_bar = st.progress(0, text="Initialising…")
    status_text  = st.empty()

    STEPS = [
        ("orchestrator",         "🧠 Orchestrator"),
        ("research_agent",       "🔍 Research Agent"),
        ("classification_agent", "📂 Classification Agent"),
        ("ner_agent",            "🏷️ NER Agent"),
        ("browser_agent",        "🌐 Browser Agent"),
        ("analyzer_agent",       "📊 Analyzer Agent"),
        ("illustration_agent",   "🎨 Illustration Agent"),
        ("writer_agent",         "✍️ Writer Agent"),
        ("critic_agent",         "🔎 Critic Agent"),
        ("finalize_node",        "✅ Finalising"),
    ]
    TOTAL = len(STEPS)

    _is_dark  = st.session_state.get("app_theme", "Dark") == "Dark"
    _head_col = "#92ccff" if _is_dark else "#006397"
    st.markdown(
        f"<h3 style='font-size:1rem;font-weight:600;color:{_head_col};"
        f"letter-spacing:-0.01em;margin:16px 0 8px 0;'>🔄 Agent Progress</h3>",
        unsafe_allow_html=True,
    )
    expanders: dict[str, Any] = {label: st.expander(label, expanded=False)
                                  for _, label in STEPS}

    st.markdown("### 📝 Live Preview")
    preview = st.empty()

    # ── Inject uploaded file as a source ─────────────────────────────────────
    initial   = ResearchState(topic=topic)
    file_text = st.session_state.get("uploaded_file_text", "")
    file_name = st.session_state.get("uploaded_file_name", "uploaded_file")
    if file_text:
        from src.state import SourceRecord  # noqa: PLC0415
        initial.raw_sources.append(SourceRecord(
            url=f"file://{file_name}",
            title=f"Uploaded: {file_name}",
            snippet=file_text[:500],
            full_content=file_text,
            relevance_score=9.0,
            source_type="documentation",
            domain="general",
            relevance_tier="high",
        ))
        initial.classified_sources.append(initial.raw_sources[-1])
        st.info(f"📁 File '{file_name}' added as a source to the pipeline.")

    state_dict  = _state_to_full_dict(initial)
    app         = _get_pipeline_graph()
    step_idx    = 0
    final_state: ResearchState | None = None

    try:
        for event in app.stream(state_dict, stream_mode="updates", config=get_graph_config()):
            for node_name, node_output in event.items():
                step_idx   += 1
                label       = next((lbl for nm, lbl in STEPS if nm == node_name), f"⚙️ {node_name}")
                progress_bar.progress(min(step_idx / TOTAL, 1.0), text=f"Running {label}…")
                status_text.info(f"▶ {label}")

                state_dict  = dict(node_output)
                state       = _dict_to_state(state_dict)
                final_state = state

                _fill_expander(node_name, label, state, expanders)

                if state.draft:
                    preview.markdown(
                        state.draft[:2500] + "\n\n*(live preview — full report below when done)*"
                    )

        progress_bar.progress(1.0, text="Complete ✓")
        status_text.success("✅ Pipeline complete!")

    except Exception as exc:
        import traceback  # noqa: PLC0415
        progress_bar.progress(1.0, text="Error")
        status_text.error(f"❌ {type(exc).__name__}")
        st.error(f"**Pipeline error:** {exc}")
        with st.expander("🔍 Full traceback (for debugging)", expanded=True):
            st.code(traceback.format_exc(), language="python")
        logger.exception("Full pipeline error")
        # Still render whatever was collected before the crash
        if final_state and final_state.final_report:
            st.warning("⚠️ Partial result available (pipeline crashed before finishing):")
            _render_final_report(final_state, topic)
        return

    if final_state:
        _save_to_history(final_state, topic)
        _render_final_report(final_state, topic)


def _fill_expander(node_name: str, label: str, state: ResearchState, expanders: dict) -> None:
    """Populate the expander for the given agent node with its output."""
    if label not in expanders:
        return
    with expanders[label]:
        if node_name == "orchestrator":
            st.success(f"{len(state.sub_questions)} sub-questions generated")
            for q in state.sub_questions:
                st.markdown(f"- {q}")

        elif node_name == "research_agent":
            st.success(f"Collected **{len(state.raw_sources)}** sources")
            for src in state.raw_sources[:6]:
                st.markdown(
                    f"**{src.title}** — score {src.relevance_score:.1f}/10  \n<{src.url}>"
                )

        elif node_name == "classification_agent":
            kept  = len(state.classified_sources)
            disc  = len(state.classification_log)
            st.success(f"{kept} sources kept · {disc} discarded")
            counts = Counter(s.source_type for s in state.classified_sources)
            if counts:
                st.table({"Source Type": list(counts.keys()), "Count": list(counts.values())})
            if state.classification_log:
                with st.expander("Discarded sources log", expanded=False):
                    for entry in state.classification_log:
                        st.caption(entry)

        elif node_name == "ner_agent":
            st.success(f"{len(state.entities)} unique entities extracted")
            render_entity_table(state, max_rows=15)
            if state.entity_relationships:
                st.caption(f"Co-occurrence pairs: {len(state.entity_relationships)}")

        elif node_name == "browser_agent":
            st.success(f"Browser visited {len(state.browser_results)} pages")
            for br in state.browser_results:
                st.markdown(f"**{br.get('title', 'N/A')}** — `{br.get('url', '')}`")
                if br.get("screenshot_path"):
                    try:
                        st.image(br["screenshot_path"], width=420)
                    except Exception:
                        st.caption(f"Screenshot: {br['screenshot_path']}")
                if "error" in br:
                    st.warning(f"Browser error: {br['error']}")

        elif node_name == "analyzer_agent":
            st.success(f"{len(state.themes)} themes · {len(state.contradictions)} contradictions")
            for t in state.themes:
                st.markdown(f"- {t}")
            for c in state.contradictions:
                st.markdown(f"- ⚠️ {c}")

        elif node_name == "illustration_agent":
            _img_errors = [e for e in (state.errors or []) if "Figure" in e and "failed" in e]
            if state.illustrations:
                st.success(f"✨ {len(state.illustrations)} figures generated via OpenAI")
            else:
                st.warning("No figures generated.")
            if _img_errors:
                with st.expander("⚠️ Figure generation errors — expand to see details", expanded=True):
                    for _e in _img_errors:
                        st.error(_e)
            for i, path in enumerate(state.illustrations):
                try:
                    st.image(path, caption=Path(path).stem.replace("_", " ").title(), width=480)
                    img_bytes = Path(path).read_bytes()
                    st.download_button(
                        f"⬇️ Download Figure {i + 1}",
                        data=img_bytes,
                        file_name=Path(path).name,
                        mime="image/png",
                        key=f"dl_fig_exp_{i}_{hash(path)}",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption(f"Figure: {path}")

        elif node_name == "writer_agent":
            st.success(f"Draft: {len(state.draft):,} characters")
            st.markdown(state.draft[:1500] + "\n\n*(preview truncated)*")

        elif node_name == "critic_agent":
            score = state.critic_score
            cls   = "score-high" if score >= config.critic_pass_score else "score-low"
            st.markdown(
                f"<span class='{cls}'>{score}/10</span> → **{state.critic_decision}**",
                unsafe_allow_html=True,
            )
            st.markdown(f"*{state.critic_feedback}*")

        elif node_name == "finalize_node":
            st.success("Report saved to `reports/`")


def _save_to_history(state: ResearchState, topic: str) -> None:
    """Persist the last 3 pipeline results in session_state."""
    if "pipeline_history" not in st.session_state:
        st.session_state["pipeline_history"] = []

    entry = {
        "topic":         topic,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "final_report":  state.final_report,
        "illustrations": list(state.illustrations),
        "cost_metrics":  state.cost_metrics,
        "entity_count":  len(state.entities),
        "source_count":  len(state.classified_sources),
        "critic_score":  state.critic_score,
    }
    history: list = st.session_state["pipeline_history"]
    history.insert(0, entry)
    st.session_state["pipeline_history"] = history[:3]  # keep last 3


def render_history_section() -> None:
    """Render the last 3 pipeline runs as collapsible sections."""
    history: list = st.session_state.get("pipeline_history", [])
    if not history:
        return

    _is_dark  = st.session_state.get("app_theme", "Dark") == "Dark"
    _head_col = "#92ccff" if _is_dark else "#006397"
    st.markdown("---")
    st.markdown(
        f"<h3 style='font-size:1rem;font-weight:600;color:{_head_col};"
        f"margin:8px 0;'>📚 Recent Runs ({len(history)}/3)</h3>",
        unsafe_allow_html=True,
    )

    for i, run in enumerate(history):
        score = run.get("critic_score", 0)
        score_color = "#61de8a" if score >= 7 else "#ffba4b" if score >= 5 else "#ffb4ab"
        label = (
            f"{'🕐' if i == 0 else '🕑' if i == 1 else '🕒'}  "
            f"**{run['topic'][:45]}** — "
            f"{run['timestamp']} · "
            f"{run['source_count']} sources · "
            f"{run['entity_count']} entities"
        )
        with st.expander(label, expanded=(i == 0 and len(history) == 1)):
            # Score badge
            st.markdown(
                f"<span style='font-size:0.85rem;font-weight:600;"
                f"color:{score_color};'>Critic Score: {score}/10</span>",
                unsafe_allow_html=True,
            )

            # Report preview
            st.markdown(run["final_report"][:3000] +
                        ("\n\n*(truncated — download for full report)*"
                         if len(run["final_report"]) > 3000 else ""))

            # Figures
            if run["illustrations"]:
                st.markdown("**Figures:**")
                img_cols = st.columns(min(len(run["illustrations"]), 3))
                for j, path in enumerate(run["illustrations"]):
                    with img_cols[j % 3]:
                        try:
                            st.image(path, caption=Path(path).stem[:25], use_container_width=True)
                            img_bytes = Path(path).read_bytes()
                            st.download_button(
                                "⬇️ Figure",
                                data=img_bytes,
                                file_name=Path(path).name,
                                mime="image/png",
                                key=f"hist_fig_{i}_{j}",
                                use_container_width=True,
                            )
                        except Exception:
                            st.caption(Path(path).name)

            # Download buttons
            slug = run["topic"][:30].replace(" ", "_")
            col_md, col_pdf = st.columns(2)
            with col_md:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=run["final_report"].encode("utf-8"),
                    file_name=f"{slug}_report.md",
                    mime="text/markdown",
                    key=f"hist_md_{i}",
                    use_container_width=True,
                )
            with col_pdf:
                from src.utils.report_exporter import save_pdf  # noqa: PLC0415
                pdf_path = save_pdf(
                    run["final_report"],
                    run["topic"],
                    image_paths=run.get("illustrations", []),
                )
                if pdf_path and Path(pdf_path).exists():
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF (with images)",
                            data=f.read(),
                            file_name=f"{slug}_report.pdf",
                            mime="application/pdf",
                            key=f"hist_pdf_{i}",
                            use_container_width=True,
                        )


def _render_final_report(state: ResearchState, topic: str) -> None:
    """Render the complete report, cost table, and download buttons."""
    from src.utils.report_exporter import markdown_to_html, save_pdf  # noqa: PLC0415

    st.markdown("---")
    st.markdown("## 📄 Final Report")

    # ── Render report with markdown2 — images embedded as base64 ─────────────
    html_report = markdown_to_html(state.final_report, state.illustrations)
    st.markdown(html_report, unsafe_allow_html=True)

    # ── Generated figures section with individual download buttons ────────────
    if state.illustrations:
        st.markdown("### 🖼️ Generated Figures")
        img_cols = st.columns(min(len(state.illustrations), 3))
        for i, path in enumerate(state.illustrations):
            with img_cols[i % 3]:
                try:
                    st.image(path, caption=f"Figure {i + 1}",
                             use_container_width=True)
                    img_bytes = Path(path).read_bytes()
                    st.download_button(
                        f"⬇️ Figure {i + 1}",
                        data=img_bytes,
                        file_name=Path(path).name,
                        mime="image/png",
                        key=f"dl_fig_final_{i}",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption(f"Figure: {path}")

    st.markdown("---")
    st.markdown("### 💰 Cost & Token Summary")
    st.table(format_cost_table(state.cost_metrics))

    st.markdown("### 📥 Download Report")
    col_md, col_pdf = st.columns(2)
    with col_md:
        st.download_button(
            "⬇️ Download Markdown",
            data=state.final_report.encode("utf-8"),
            file_name=f"{topic[:30].replace(' ', '_')}_report.md",
            mime="text/markdown",
            key="dl_report_md",
            use_container_width=True,
        )
    with col_pdf:
        pdf_path = save_pdf(
            state.final_report,
            topic,
            image_paths=list(state.illustrations),
        )
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF (with images)",
                    data=f.read(),
                    file_name=f"{topic[:30].replace(' ', '_')}_report.pdf",
                    mime="application/pdf",
                    key="dl_report_pdf",
                    use_container_width=True,
                )

    if state.errors:
        with st.expander("⚠️ Non-fatal pipeline errors"):
            for e in state.errors:
                st.warning(e)
