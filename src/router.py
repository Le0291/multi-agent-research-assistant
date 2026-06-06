"""
router.py — Smart mode router for the Multi-Agent Research Assistant.

WHY TWO MODES EXIST
====================
Full Pipeline  (project requirement)
    The course specification mandates a complete multi-agent LangGraph workflow
    with orchestrator → research → classification → NER → browser → analysis →
    illustration → writer → critic.  This must remain the default execution path.

Agent Playground  (innovation / usability feature)
    During development and live demos it is useful to run a SINGLE agent in
    isolation to inspect its output, debug its behaviour, or show the audience
    exactly what each agent contributes.  The playground satisfies the
    "innovation" rubric criterion without breaking the mandatory pipeline.

HOW ROUTING WORKS
=================
The `route()` function receives a mode string and a dict of user-supplied
inputs.  It dispatches to the appropriate runner function.

    Full Pipeline          → src.graph.run_pipeline()   (LangGraph)
    Any playground mode    → direct Python function call (no LangGraph needed)

For playground modes that require upstream data (e.g. NER needs text, Analysis
needs classified sources) the router automatically runs the minimum required
prerequisite agents BEFORE invoking the selected agent.  A log of which
prerequisites were auto-resolved is returned so the UI can display it.

DEPENDENCY MAP
==============
Mode                Prerequisites auto-resolved
─────────────────   ──────────────────────────────────────────────────────
Research Only       orchestrator  (for sub-questions only)
Classification      orchestrator → research
NER Only            orchestrator → research   (or raw text if provided)
Browser Only        NONE — user supplies URLs directly
Analysis Only       orchestrator → research → classification → NER
Writer Only         orchestrator → research → classification → NER → analysis
                    → illustration
Critic Only         NONE — user supplies draft text directly
Illustration Only   NONE if prompts supplied; else orchestrator → research
                    → classification → NER → analysis (to get image_prompts)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.state import ResearchState, SourceRecord, CostMetrics

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Mode catalogue
# ─────────────────────────────────────────────────────────────────────────────

# All available modes shown in the sidebar
ALL_MODES: list[str] = [
    "Full Pipeline",        # Default — complete LangGraph workflow
    "Research Only",        # Research agent in isolation
    "Classification Only",  # Classification agent in isolation
    "NER Only",             # NER (SpaCy) agent in isolation
    "Browser Only",         # Playwright browser agent in isolation
    "Analysis Only",        # Analyzer agent in isolation
    "Writer Only",          # Writer agent in isolation
    "Critic Only",          # Critic agent in isolation
    "Illustration Only",    # Illustration agent in isolation
]

# These modes appear under "Agent Playground" in the sidebar
PLAYGROUND_MODES: list[str] = [m for m in ALL_MODES if m != "Full Pipeline"]

# Human-readable descriptions shown in the UI help text
MODE_DESCRIPTIONS: dict[str, str] = {
    "Full Pipeline": (
        "Complete multi-agent workflow: research → classify → NER → browser → "
        "analysis → illustrate → write → critique.  Produces a full PDF report."
    ),
    "Research Only": (
        "Run the Research Agent alone.  Searches the web, scrapes pages, and "
        "scores source relevance.  Returns URLs, titles, and relevance scores."
    ),
    "Classification Only": (
        "Classify sources by type (academic, blog, news…) and domain.  "
        "Auto-searches if no sources are cached."
    ),
    "NER Only": (
        "Extract named entities (people, orgs, technologies, dates, locations) "
        "from text or auto-fetched sources.  Returns entity frequency table."
    ),
    "Browser Only": (
        "Open real browser pages via Playwright.  Paste URLs to capture "
        "metadata, headings, and screenshots without running the full pipeline."
    ),
    "Analysis Only": (
        "Synthesise themes, detect contradictions, and build a report outline "
        "from automatically collected sources."
    ),
    "Writer Only": (
        "Generate a full Markdown report draft.  Auto-runs research and "
        "analysis prerequisites first."
    ),
    "Critic Only": (
        "Paste any draft text.  The critic scores it 1–10 against the "
        "competition rubric and returns actionable feedback."
    ),
    "Illustration Only": (
        "Generate academic figures.  Provide image prompts directly, or let the "
        "system derive them from a topic automatically."
    ),
}

# Which inputs each playground mode accepts from the user
MODE_INPUTS: dict[str, list[str]] = {
    "Research Only":        ["topic"],
    "Classification Only":  ["topic", "urls"],
    "NER Only":             ["topic", "text"],
    "Browser Only":         ["urls"],
    "Analysis Only":        ["topic"],
    "Writer Only":          ["topic"],
    "Critic Only":          ["draft"],
    "Illustration Only":    ["topic", "image_prompts"],
}


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helper — build a minimal ResearchState
# ─────────────────────────────────────────────────────────────────────────────

def _empty_state(topic: str = "") -> ResearchState:
    """Create a ResearchState with default values for playground use."""
    return ResearchState(topic=topic, started_at=datetime.utcnow().isoformat())


def _inject_file_source(state: ResearchState, inputs: dict) -> None:
    """
    If the user uploaded a file (extra_file_text in inputs), inject it as
    a SourceRecord into both raw_sources and classified_sources so every
    downstream agent can use it alongside web-fetched content.
    """
    file_text = inputs.get("extra_file_text", "").strip()
    file_name = inputs.get("extra_file_name", "uploaded_file")

    if not file_text:
        return

    file_source = SourceRecord(
        url=f"file://{file_name}",
        title=f"Uploaded document: {file_name}",
        snippet=file_text[:500],
        full_content=file_text,
        relevance_score=9.0,       # user-supplied content is always highly relevant
        source_type="documentation",
        domain="general",
        relevance_tier="high",
    )
    # Add to both lists so agents that read either list will find it
    state.raw_sources.insert(0, file_source)
    state.classified_sources.insert(0, file_source)
    logger.info("[Router] Injected uploaded file '%s' as a source.", file_name)


def _result(mode: str, state: ResearchState, **extra: Any) -> dict[str, Any]:
    """Build a standardised result dict returned by every runner."""
    return {
        "mode":         mode,
        "status":       "success",
        "topic":        state.topic,
        "errors":       state.errors,
        "cost_metrics": state.cost_metrics,
        **extra,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Prerequisite helpers (auto dependency resolution)
# ─────────────────────────────────────────────────────────────────────────────

def _run_orchestrator(state: ResearchState) -> None:
    """Populate sub_questions via the orchestrator agent (in-place)."""
    from src.agents.orchestrator import orchestrator_node  # noqa: PLC0415
    updates = orchestrator_node(state)
    state.sub_questions = updates.get("sub_questions", [])


def _run_research(state: ResearchState) -> None:
    """Run Research Agent in-place; populates raw_sources."""
    from src.agents.research_agent import research_agent_node  # noqa: PLC0415
    if not state.sub_questions:
        _run_orchestrator(state)
    updates = research_agent_node(state)
    state.raw_sources = updates.get("raw_sources", [])


def _run_classification(state: ResearchState) -> None:
    """Run Classification Agent in-place; populates classified_sources."""
    from src.agents.classification_agent import classification_agent_node  # noqa: PLC0415
    if not state.raw_sources:
        _run_research(state)
    updates = classification_agent_node(state)
    state.classified_sources   = updates.get("classified_sources", [])
    state.classification_log   = updates.get("classification_log", [])


def _run_ner(state: ResearchState) -> None:
    """Run NER Agent in-place; populates entities and relationships."""
    from src.agents.ner_agent import ner_agent_node  # noqa: PLC0415
    if not state.classified_sources:
        _run_classification(state)
    updates = ner_agent_node(state)
    state.entities             = updates.get("entities", [])
    state.entity_relationships = updates.get("entity_relationships", [])
    state.entity_source_map    = updates.get("entity_source_map", {})


def _run_analysis(state: ResearchState) -> None:
    """Run Analyzer Agent in-place; populates themes, outline, etc."""
    from src.agents.analyzer_agent import analyzer_agent_node  # noqa: PLC0415
    if not state.entities:
        _run_ner(state)
    updates = analyzer_agent_node(state)
    state.themes              = updates.get("themes", [])
    state.contradictions      = updates.get("contradictions", [])
    state.evidence_summary    = updates.get("evidence_summary", {})
    state.outline             = updates.get("outline", [])
    state.image_prompts       = updates.get("image_prompts", [])
    state.transformer_config  = updates.get("transformer_config", {})
    state.moe_analysis        = updates.get("moe_analysis", {})


def _run_illustration(state: ResearchState) -> None:
    """Run Illustration Agent in-place; populates illustrations."""
    from src.agents.illustration_agent import illustration_agent_node  # noqa: PLC0415
    if not state.image_prompts:
        _run_analysis(state)
    updates = illustration_agent_node(state)
    state.illustrations = updates.get("illustrations", [])


# ─────────────────────────────────────────────────────────────────────────────
#  Individual mode runners
# ─────────────────────────────────────────────────────────────────────────────

def run_research_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Research Agent in isolation.

    Deps auto-resolved: orchestrator (sub-questions).
    Input : topic (str)
    Output: raw_sources list with URL, title, relevance score.
    """
    topic = inputs.get("topic", "").strip()
    if not topic:
        return {"mode": "Research Only", "status": "error",
                "errors": ["A research topic is required."]}

    state = _empty_state(topic)
    logger.info("[Research Only] Starting for topic: %s", topic)
    _inject_file_source(state, inputs)

    _run_orchestrator(state)  # generates sub_questions used by research agent
    _run_research(state)

    return _result("Research Only", state, sources=state.raw_sources,
                   sub_questions=state.sub_questions)


def run_classification_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Classification Agent in isolation.

    Deps auto-resolved: orchestrator + research (if no URLs supplied).
    Input : topic OR comma-separated URLs
    Output: classified_sources with source_type, domain, relevance_tier.
    """
    topic = inputs.get("topic", "").strip()
    urls  = [u.strip() for u in inputs.get("urls", []) if u.strip()]

    if not topic and not urls:
        return {"mode": "Classification Only", "status": "error",
                "errors": ["Provide a topic or at least one URL."]}

    from src.tools.scrape_tool import scrape_page  # noqa: PLC0415

    state = _empty_state(topic or "User-supplied URLs")
    _inject_file_source(state, inputs)   # inject uploaded file if present

    if urls:
        # Build SourceRecords from user-supplied URLs instead of searching
        logger.info("[Classification Only] Using %d user-supplied URLs.", len(urls))
        for url in urls:
            snippet = scrape_page(url, max_chars=500) or url
            state.raw_sources.append(
                SourceRecord(url=url, title=url, snippet=snippet,
                             full_content=snippet, relevance_score=7.0)
            )
    else:
        # Auto-research to get sources
        logger.info("[Classification Only] Auto-researching topic: %s", topic)
        _run_orchestrator(state)
        _run_research(state)

    from src.agents.classification_agent import classification_agent_node  # noqa: PLC0415
    updates = classification_agent_node(state)
    state.classified_sources = updates.get("classified_sources", [])
    state.classification_log = updates.get("classification_log", [])

    return _result("Classification Only", state,
                   classified_sources=state.classified_sources,
                   classification_log=state.classification_log)


def run_ner_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    NER Agent in isolation.

    Deps auto-resolved:
      - If raw text supplied → create a single SourceRecord and run NER.
      - If topic supplied   → auto-research, then NER.
    Input : topic OR free text
    Output: entities, entity_relationships, entity_source_map.
    """
    topic = inputs.get("topic", "").strip()
    text  = inputs.get("text", "").strip()

    if not topic and not text:
        return {"mode": "NER Only", "status": "error",
                "errors": ["Provide a topic or paste text to analyse."]}

    state = _empty_state(topic or "User-supplied text")

    if text:
        # Skip web search — run NER directly on the provided text
        logger.info("[NER Only] Running on user-provided text (%d chars).", len(text))
        _inject_file_source(state, inputs)   # also inject any uploaded file
        state.classified_sources = [
            SourceRecord(url="user_input", title="User-provided text",
                         snippet=text[:500], full_content=text,
                         relevance_score=8.0)
        ]
    else:
        # Auto-research to get source text
        logger.info("[NER Only] Auto-researching for NER: %s", topic)
        _run_orchestrator(state)
        _run_research(state)
        # Use raw_sources as classified_sources so ner_agent has content
        state.classified_sources = state.raw_sources

    from src.agents.ner_agent import ner_agent_node  # noqa: PLC0415
    updates = ner_agent_node(state)
    state.entities             = updates.get("entities", [])
    state.entity_relationships = updates.get("entity_relationships", [])
    state.entity_source_map    = updates.get("entity_source_map", {})

    return _result("NER Only", state,
                   entities=state.entities,
                   entity_relationships=state.entity_relationships,
                   entity_source_map=state.entity_source_map)


def run_browser_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Browser Agent in isolation.

    No prerequisites.  User supplies URLs directly.
    Input : list of URLs
    Output: browser_results (metadata + screenshot paths).
    """
    urls = [u.strip() for u in inputs.get("urls", []) if u.strip()]
    if not urls:
        return {"mode": "Browser Only", "status": "error",
                "errors": ["Provide at least one URL to visit."]}

    state = _empty_state("Browser visit")
    # Build minimal SourceRecords from the user-supplied URLs
    state.classified_sources = [
        SourceRecord(url=url, title=url, snippet="", full_content="",
                     relevance_score=8.0)
        for url in urls
    ]

    from src.agents.browser_agent import browser_agent_node  # noqa: PLC0415
    updates = browser_agent_node(state)
    state.browser_results = updates.get("browser_results", [])

    return _result("Browser Only", state, browser_results=state.browser_results)


def run_analysis_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Analyzer Agent in isolation.

    Deps auto-resolved: orchestrator → research → classification → NER.
    Input : topic
    Output: themes, contradictions, evidence_summary, outline, image_prompts.
    """
    topic = inputs.get("topic", "").strip()
    if not topic:
        return {"mode": "Analysis Only", "status": "error",
                "errors": ["A research topic is required."]}

    state = _empty_state(topic)
    logger.info("[Analysis Only] Running prerequisites for: %s", topic)
    _inject_file_source(state, inputs)   # inject uploaded file before research

    # Auto-resolve full chain up to NER
    _run_ner(state)   # internally calls orchestrator → research → classification

    from src.agents.analyzer_agent import analyzer_agent_node  # noqa: PLC0415
    updates = analyzer_agent_node(state)
    state.themes             = updates.get("themes", [])
    state.contradictions     = updates.get("contradictions", [])
    state.evidence_summary   = updates.get("evidence_summary", {})
    state.outline            = updates.get("outline", [])
    state.image_prompts      = updates.get("image_prompts", [])

    return _result("Analysis Only", state,
                   themes=state.themes,
                   contradictions=state.contradictions,
                   evidence_summary=state.evidence_summary,
                   outline=state.outline,
                   image_prompts=state.image_prompts,
                   sources_used=len(state.classified_sources))


def run_writer_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Writer Agent in isolation.

    Deps auto-resolved: orchestrator → research → classification → NER →
                        analysis → illustration.
    Input : topic
    Output: draft (Markdown string) + references list.
    """
    topic = inputs.get("topic", "").strip()
    if not topic:
        return {"mode": "Writer Only", "status": "error",
                "errors": ["A research topic is required."]}

    state = _empty_state(topic)
    logger.info("[Writer Only] Running prerequisites for: %s", topic)
    _inject_file_source(state, inputs)   # inject uploaded file as primary source

    # Auto-resolve all prerequisites
    _run_illustration(state)  # internally: orchestrator→research→classify→NER→analysis

    from src.agents.writer_agent import writer_agent_node  # noqa: PLC0415
    from src.utils.citations import build_references_section  # noqa: PLC0415

    updates = writer_agent_node(state)
    state.draft = updates.get("draft", "")

    return _result("Writer Only", state,
                   draft=state.draft,
                   references=state.references,
                   outline=state.outline,
                   sources_used=len(state.classified_sources))


def run_critic_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Critic Agent in isolation.

    No prerequisites — user supplies a draft text directly.
    Input : draft (str)
    Output: critic_score, critic_decision, critic_feedback.
    """
    draft = inputs.get("draft", "").strip()
    topic = inputs.get("topic", "User-supplied draft").strip()

    if not draft:
        return {"mode": "Critic Only", "status": "error",
                "errors": ["Paste a draft report to review."]}

    state = _empty_state(topic)
    state.draft = draft
    # Provide minimal context so the critic can score properly
    state.classified_sources = []
    state.entities = []

    from src.agents.critic_agent import critic_agent_node  # noqa: PLC0415
    updates = critic_agent_node(state)

    return _result("Critic Only", state,
                   critic_score=updates.get("critic_score", 0),
                   critic_decision=updates.get("critic_decision", "REVISE"),
                   critic_feedback=updates.get("critic_feedback", ""))


def run_illustration_only(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Illustration Agent in isolation.

    Deps auto-resolved:
      - If image_prompts supplied → generate images directly.
      - If only topic supplied    → auto-run analysis to derive prompts first.
    Input : image_prompts (list of str) OR topic
    Output: illustrations (list of file paths).
    """
    raw_prompts   = inputs.get("image_prompts", [])
    topic         = inputs.get("topic", "").strip()
    image_prompts = [p.strip() for p in raw_prompts if p.strip()]

    if not image_prompts and not topic:
        return {"mode": "Illustration Only", "status": "error",
                "errors": ["Provide image prompts or a topic to auto-derive them."]}

    state = _empty_state(topic or "Illustrations")

    if image_prompts:
        # Use user-provided prompts directly — no web search needed
        state.image_prompts = image_prompts
        logger.info("[Illustration Only] Using %d user prompts.", len(image_prompts))
    else:
        # Auto-derive prompts by running analysis prerequisites
        logger.info("[Illustration Only] Auto-deriving prompts for: %s", topic)
        _run_analysis(state)  # internally: orchestrator→research→classify→NER

    from src.agents.illustration_agent import illustration_agent_node  # noqa: PLC0415
    updates = illustration_agent_node(state)
    state.illustrations = updates.get("illustrations", [])

    return _result("Illustration Only", state,
                   illustrations=state.illustrations,
                   image_prompts=state.image_prompts)


# ─────────────────────────────────────────────────────────────────────────────
#  Main dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def route(mode: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch to the correct execution function based on the selected mode.

    For "Full Pipeline" → delegates to src.graph.run_pipeline() (LangGraph).
    For all other modes → calls the appropriate direct runner function.

    Returns a standardised dict:
        mode          – the selected mode string
        status        – "success" | "error"
        topic         – topic used
        errors        – list of non-fatal error strings
        cost_metrics  – CostMetrics object
        <mode-specific keys> – see each runner's docstring
    """
    logger.info("Router dispatching mode=%r", mode)

    runners: dict[str, Any] = {
        "Research Only":        run_research_only,
        "Classification Only":  run_classification_only,
        "NER Only":             run_ner_only,
        "Browser Only":         run_browser_only,
        "Analysis Only":        run_analysis_only,
        "Writer Only":          run_writer_only,
        "Critic Only":          run_critic_only,
        "Illustration Only":    run_illustration_only,
    }

    if mode == "Full Pipeline":
        # Full Pipeline is handled by the Streamlit streaming loop directly;
        # this branch is only used when called from non-streaming contexts.
        from src.graph import run_pipeline  # noqa: PLC0415
        state = run_pipeline(inputs.get("topic", ""))
        return {"mode": "Full Pipeline", "status": "success",
                "state": state, "errors": state.errors,
                "cost_metrics": state.cost_metrics}

    runner = runners.get(mode)
    if not runner:
        return {"mode": mode, "status": "error",
                "errors": [f"Unknown mode: {mode!r}"]}

    try:
        return runner(inputs)
    except Exception as exc:
        import traceback  # noqa: PLC0415
        logger.exception("Router: mode %r raised", mode)
        return {"mode": mode, "status": "error",
                "errors": [f"Execution failed: {exc}"],
                "traceback": traceback.format_exc()}
