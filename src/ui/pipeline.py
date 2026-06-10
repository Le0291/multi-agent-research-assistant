"""
src/ui/pipeline.py — Full Pipeline UI (Mode 1: Competition mode).

All 9 LangGraph agents run sequentially.  Results are streamed in real-time
into collapsible expander panels.  A Markdown + PDF report is produced at the end.

Stability notes:
  - The final result is persisted in st.session_state and rendered OUTSIDE the
    run branch — download clicks (which rerun the script) no longer wipe it.
  - No download buttons are rendered while the pipeline is streaming: clicking
    any button mid-run aborts the running Streamlit script.
  - History is mirrored to disk (src/ui/history.py) so it survives refreshes.
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
from src.utils.report_exporter import save_pdf, embed_images_base64, markdown_to_html
from src.ui.components import render_file_upload, render_entity_table
from src.ui.history import get_history, add_history_entry, update_entry

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
            width="stretch", key="fp_run",
        )
    with col2:
        st.button("💡 Demo Topic", key="fp_demo",
                  width="stretch", on_click=_set_demo_topic)

    if st.session_state.get("fp_running") and not run_btn:
        st.session_state["fp_running"] = False

    if run_btn:
        if not topic.strip():
            st.error("Please enter a research topic.")
        else:
            _stream_full_pipeline(topic.strip())

    # Rendered OUTSIDE the run branch on purpose: every download-button click
    # triggers a full Streamlit rerun, and previously that wiped the result
    # from the screen (it only existed inside the `if run_btn:` branch — the
    # app looked like it "switched off" after a download).  Persisting the
    # result in session_state and rendering it here keeps it on screen across
    # any number of reruns/downloads.
    last = st.session_state.get("fp_last_result")
    if last:
        _render_result(last, key_prefix="last")

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

    st.session_state["fp_running"] = True
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
        # Still keep whatever was collected before the crash (not saved to history)
        if final_state and final_state.final_report:
            st.warning("⚠️ Partial result available (pipeline crashed before finishing):")
            st.session_state["fp_last_result"] = _build_result(final_state, topic)
        st.session_state["fp_running"] = False
        _cleanup_after_run()
        return

    if final_state:
        result = _build_result(final_state, topic)
        add_history_entry(result)
        st.session_state["fp_last_result"] = result

    st.session_state["fp_running"] = False
    # Free memory so a second back-to-back run doesn't get the container
    # OOM-killed (which looks like the app restarting itself)
    _cleanup_after_run()


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
            # NO download buttons here — clicking ANY button while the
            # pipeline is streaming makes Streamlit abort the running script
            # (the app appeared to "switch off" mid-run after a download).
            # Downloads are offered in the final-result section instead.
            for path in state.illustrations:
                try:
                    st.image(path, caption=Path(path).stem.replace("_", " ").title(), width=480)
                except Exception:
                    st.caption(f"Figure: {path}")
            if state.illustrations:
                st.caption("⬇️ Download buttons appear in the result section once the run completes.")

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


def _build_result(state: ResearchState, topic: str) -> dict:
    """
    Serialise a finished run into a small JSON-safe dict.

    Stored in session_state (survives reruns from download clicks) and in the
    on-disk history (survives page refreshes).  The PDF is generated ONCE here
    — previously it was regenerated on every Streamlit rerun, hammering CPU.
    """
    pdf_path = ""
    try:
        pdf_path = save_pdf(state.final_report, topic,
                            image_paths=list(state.illustrations)) or ""
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)

    return {
        "topic":         topic,
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "final_report":  state.final_report,
        "illustrations": [p for p in state.illustrations if p],
        "cost_table":    format_cost_table(state.cost_metrics),
        "entity_count":  len(state.entities),
        "source_count":  len(state.classified_sources),
        "critic_score":  state.critic_score,
        "errors":        list(state.errors),
        "pdf_path":      pdf_path,
    }


def _cleanup_after_run() -> None:
    """
    Free memory between runs.

    Back-to-back runs used to accumulate enough RAM (matplotlib figure state,
    big intermediate state dicts) that small hosts OOM-killed the container on
    the second run — which the user experiences as the app restarting itself.
    """
    import gc  # noqa: PLC0415
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        plt.close("all")
    except Exception:
        pass
    gc.collect()


def render_history_section() -> None:
    """Render the last 3 pipeline runs (disk-persisted — survives page refresh)."""
    history: list = get_history()
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
        score  = run.get("critic_score", 0)
        report = run.get("final_report", "")
        score_color = "#61de8a" if score >= 7 else "#ffba4b" if score >= 5 else "#ffb4ab"
        label = (
            f"{'🕐' if i == 0 else '🕑' if i == 1 else '🕒'}  "
            f"**{run.get('topic', '?')[:45]}** — "
            f"{run.get('timestamp', '')} · "
            f"{run.get('source_count', 0)} sources · "
            f"{run.get('entity_count', 0)} entities"
        )
        with st.expander(label, expanded=False):
            # Score badge
            st.markdown(
                f"<span style='font-size:0.85rem;font-weight:600;"
                f"color:{score_color};'>Critic Score: {score}/10</span>",
                unsafe_allow_html=True,
            )

            # Report preview
            st.markdown(report[:3000] +
                        ("\n\n*(truncated — download for full report)*"
                         if len(report) > 3000 else ""))

            # Figures (paths may be stale after a redeploy — degrade gracefully)
            for_figs = run.get("illustrations", [])
            if for_figs:
                st.markdown("**Figures:**")
                img_cols = st.columns(min(len(for_figs), 3))
                for j, path in enumerate(for_figs):
                    with img_cols[j % 3]:
                        try:
                            st.image(path, caption=Path(path).stem[:25], width="stretch")
                            st.download_button(
                                "⬇️ Figure",
                                data=Path(path).read_bytes(),
                                file_name=Path(path).name,
                                mime="image/png",
                                key=f"hist_fig_{i}_{j}",
                                width="stretch",
                            )
                        except Exception:
                            st.caption(Path(path).name)

            # Download buttons — images embedded in the MD; PDF generated at
            # most ONCE per entry (the old code rebuilt every PDF on every
            # Streamlit rerun, a big CPU/memory drain on each interaction)
            slug = run.get("topic", "report")[:30].replace(" ", "_")
            col_md, col_pdf = st.columns(2)
            with col_md:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=embed_images_base64(report).encode("utf-8"),
                    file_name=f"{slug}_report.md",
                    mime="text/markdown",
                    key=f"hist_md_{i}",
                    width="stretch",
                )
            with col_pdf:
                pdf_path = run.get("pdf_path", "")
                if not (pdf_path and Path(pdf_path).exists()):
                    try:
                        pdf_path = save_pdf(report, run.get("topic", "report"),
                                            image_paths=run.get("illustrations", [])) or ""
                        update_entry(i, pdf_path=pdf_path)
                    except Exception:
                        pdf_path = ""
                if pdf_path and Path(pdf_path).exists():
                    st.download_button(
                        "⬇️ Download PDF (with images)",
                        data=Path(pdf_path).read_bytes(),
                        file_name=f"{slug}_report.pdf",
                        mime="application/pdf",
                        key=f"hist_pdf_{i}",
                        width="stretch",
                    )


def _render_result(res: dict, key_prefix: str) -> None:
    """
    Render a finished run from its JSON-safe dict (session_state / history).

    Because this renders from persisted data — not from variables that only
    exist during the run — the result stays on screen across the rerun that
    every download-button click triggers.
    """
    st.markdown("---")
    st.markdown("## 📄 Final Report")

    report        = res.get("final_report", "")
    illustrations = res.get("illustrations", [])

    # Render with markdown2 — figures embedded as base64 <img> inside the report
    html_report = markdown_to_html(report, illustrations)
    st.markdown(html_report, unsafe_allow_html=True)

    if illustrations:
        st.markdown("### 🖼️ Generated Figures")
        img_cols = st.columns(min(len(illustrations), 3))
        for i, path in enumerate(illustrations):
            with img_cols[i % 3]:
                try:
                    st.image(path, caption=f"Figure {i + 1}",
                             width="stretch")
                    st.download_button(
                        f"⬇️ Figure {i + 1}",
                        data=Path(path).read_bytes(),
                        file_name=Path(path).name,
                        mime="image/png",
                        key=f"{key_prefix}_fig_{i}",
                        width="stretch",
                    )
                except Exception:
                    st.caption(f"Figure unavailable: {Path(path).name}")

    st.markdown("---")
    st.markdown("### 💰 Cost & Token Summary")
    if res.get("cost_table"):
        st.table(res["cost_table"])

    st.markdown("### 📥 Download Report")
    slug = res.get("topic", "report")[:30].replace(" ", "_")
    col_md, col_pdf = st.columns(2)
    with col_md:
        st.download_button(
            "⬇️ Download Markdown (figures embedded)",
            data=embed_images_base64(report).encode("utf-8"),
            file_name=f"{slug}_report.md",
            mime="text/markdown",
            key=f"{key_prefix}_report_md",
            width="stretch",
        )
    with col_pdf:
        pdf_path = res.get("pdf_path", "")
        if pdf_path and Path(pdf_path).exists():
            st.download_button(
                "⬇️ Download PDF (with images)",
                data=Path(pdf_path).read_bytes(),
                file_name=f"{slug}_report.pdf",
                mime="application/pdf",
                key=f"{key_prefix}_report_pdf",
                width="stretch",
            )

    if res.get("errors"):
        with st.expander("⚠️ Non-fatal pipeline errors"):
            for e in res["errors"]:
                st.warning(e)
