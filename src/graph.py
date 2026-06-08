"""
graph.py — LangGraph StateGraph definition.

Root-cause fix for infinite recursion
======================================
The previous version had nodes returning PARTIAL state dicts (only the keys
each agent changed).  With some LangGraph / Python versions, StateGraph(dict)
treats a partial return as a full replacement, silently dropping every key the
node did not include — including `revision_count`.  That reset revision_count
to 0 on every iteration, so the critic→writer loop never terminated.

The definitive fix: every _wrap call serialises the full ResearchState back to
a dict, then merges the node's specific updates on top before returning.
LangGraph therefore always receives a complete state dict with every field,
including the correctly-incremented revision_count.

Additional safeguards
---------------------
  • route_after_critic reads directly from the raw state dict (no dataclass
    conversion) — immune to any reconstruction bugs.
  • _wrap has a try/except around BOTH the conversion and the node call.
  • recursion_limit = 100 (pipeline + revision loop max ~20 steps).
  • Hard emergency cap stops the loop if revision_count somehow exceeds 10.
"""

from __future__ import annotations

import dataclasses
import logging
import traceback
from typing import Any

from langgraph.graph import StateGraph, END

from src.config import config
from src.state import ResearchState, CostMetrics, SourceRecord, EntityRecord
from src.agents.orchestrator import orchestrator_node
from src.agents.research_agent import research_agent_node
from src.agents.classification_agent import classification_agent_node
from src.agents.ner_agent import ner_agent_node
from src.agents.browser_agent import browser_agent_node
from src.agents.analyzer_agent import analyzer_agent_node
from src.agents.illustration_agent import illustration_agent_node
from src.agents.writer_agent import writer_agent_node
from src.agents.critic_agent import critic_agent_node
from src.utils.report_exporter import save_markdown, save_pdf

logger = logging.getLogger(__name__)

_RECURSION_LIMIT   = 100   # total node-execution budget for the whole run
_EMERGENCY_REV_CAP = 10    # absolute last-resort loop terminator


# ─────────────────────────────────────────────────────────────────────────────
#  State helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_source(s: Any) -> SourceRecord:
    if isinstance(s, SourceRecord):
        return s
    if isinstance(s, dict):
        valid = {f.name for f in dataclasses.fields(SourceRecord)}
        return SourceRecord(**{k: v for k, v in s.items() if k in valid})
    return SourceRecord(url="", title="unknown", snippet="", full_content="",
                        relevance_score=0.0)


def _to_entity(e: Any) -> EntityRecord:
    if isinstance(e, EntityRecord):
        return e
    if isinstance(e, dict):
        valid = {f.name for f in dataclasses.fields(EntityRecord)}
        return EntityRecord(**{k: v for k, v in e.items() if k in valid})
    return EntityRecord(text="", label="MISC", category="concept")


def _to_cost(c: Any) -> CostMetrics:
    if isinstance(c, CostMetrics):
        return c
    if isinstance(c, dict):
        return CostMetrics(
            input_tokens=int(c.get("input_tokens", 0)),
            output_tokens=int(c.get("output_tokens", 0)),
            input_price_per_1m=float(c.get("input_price_per_1m", 1.0)),
            output_price_per_1m=float(c.get("output_price_per_1m", 5.0)),
        )
    return CostMetrics()


def _dict_to_state(d: dict[str, Any]) -> ResearchState:
    """
    Reconstruct ResearchState from a LangGraph state dict.
    Every field is read via explicit .get() — unknown LangGraph-internal keys
    are simply ignored (no **d expansion that could raise TypeError).
    """
    raw_rel = d.get("entity_relationships", [])
    entity_relationships = [
        tuple(r) if isinstance(r, (list, tuple)) else r for r in raw_rel
    ]
    return ResearchState(
        topic                = d.get("topic", ""),
        sub_questions        = list(d.get("sub_questions", [])),
        raw_sources          = [_to_source(s) for s in d.get("raw_sources", [])],
        classified_sources   = [_to_source(s) for s in d.get("classified_sources", [])],
        entities             = [_to_entity(e) for e in d.get("entities", [])],
        entity_relationships = entity_relationships,
        entity_source_map    = dict(d.get("entity_source_map", {})),
        browser_results      = list(d.get("browser_results", [])),
        themes               = list(d.get("themes", [])),
        contradictions       = list(d.get("contradictions", [])),
        evidence_summary     = dict(d.get("evidence_summary", {})),
        outline              = list(d.get("outline", [])),
        image_prompts        = list(d.get("image_prompts", [])),
        transformer_config   = dict(d.get("transformer_config", {})),
        moe_analysis         = dict(d.get("moe_analysis", {})),
        draft                = d.get("draft", ""),
        references           = list(d.get("references", [])),
        illustrations        = list(d.get("illustrations", [])),
        critic_feedback      = d.get("critic_feedback", ""),
        critic_score         = int(d.get("critic_score", 0)),
        critic_decision      = d.get("critic_decision", "REVISE"),
        revision_count       = int(d.get("revision_count", 0)),
        final_report         = d.get("final_report", ""),
        report_path          = d.get("report_path", ""),
        status               = d.get("status", "idle"),
        errors               = list(d.get("errors", [])),
        cost_metrics         = _to_cost(d.get("cost_metrics", {})),
        started_at           = d.get("started_at", ""),
        classification_log   = list(d.get("classification_log", [])),
    )


def _state_to_full_dict(state: ResearchState) -> dict[str, Any]:
    """
    Serialise a ResearchState to a complete flat dict.

    Returning the FULL dict from every node ensures LangGraph never silently
    drops fields that a node did not mention — regardless of LangGraph version
    or whether it treats partial returns as replacements.
    """
    try:
        return dataclasses.asdict(state)
    except Exception as exc:
        logger.error("dataclasses.asdict failed (%s); using manual fallback.", exc)
        # Manual fallback — guaranteed to work
        return {
            "topic":                state.topic,
            "sub_questions":        state.sub_questions,
            "raw_sources":          [dataclasses.asdict(s) for s in state.raw_sources],
            "classified_sources":   [dataclasses.asdict(s) for s in state.classified_sources],
            "entities":             [dataclasses.asdict(e) for e in state.entities],
            "entity_relationships": [list(r) for r in state.entity_relationships],
            "entity_source_map":    state.entity_source_map,
            "browser_results":      state.browser_results,
            "themes":               state.themes,
            "contradictions":       state.contradictions,
            "evidence_summary":     state.evidence_summary,
            "outline":              state.outline,
            "image_prompts":        state.image_prompts,
            "transformer_config":   state.transformer_config,
            "moe_analysis":         state.moe_analysis,
            "draft":                state.draft,
            "references":           state.references,
            "illustrations":        state.illustrations,
            "critic_feedback":      state.critic_feedback,
            "critic_score":         state.critic_score,
            "critic_decision":      state.critic_decision,
            "revision_count":       state.revision_count,
            "final_report":         state.final_report,
            "report_path":          state.report_path,
            "status":               state.status,
            "errors":               state.errors,
            "cost_metrics":         dataclasses.asdict(state.cost_metrics),
            "started_at":           state.started_at,
            "classification_log":   state.classification_log,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Node wrapper  — THE critical change
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(fn):
    """
    Wrap a ResearchState-typed agent function for LangGraph's dict state.

    Key behaviour:
      1. Convert incoming state dict → ResearchState.
      2. Call the agent.
      3. Serialise the FULL state back to dict.
      4. Overlay the agent's specific updates.
      5. Return the complete dict.

    This guarantees LangGraph always has the entire state after every node,
    with no risk of fields being silently dropped or reset.
    """
    def wrapped(state_dict: dict[str, Any]) -> dict[str, Any]:

        # ── Step 1: dict → ResearchState ──────────────────────────────────────
        try:
            state = _dict_to_state(state_dict)
        except Exception as exc:
            logger.error("State conversion failed in %s: %s", fn.__name__, exc)
            result = dict(state_dict)
            result["errors"] = list(result.get("errors", [])) + \
                               [f"State conversion ({fn.__name__}): {exc}"]
            return result

        # ── Step 2: run the agent ─────────────────────────────────────────────
        try:
            updates: dict[str, Any] = fn(state)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Node %s raised:\n%s", fn.__name__, tb)
            full = _state_to_full_dict(state)
            full["errors"] = list(state.errors) + [f"{fn.__name__}: {exc}"]
            full["status"] = "error"
            return full

        # ── Step 3: build the full return dict ────────────────────────────────
        # Start from the complete serialised state (captures any in-place
        # mutations, e.g. cost_metrics updated by invoke_claude inside the node)
        full = _state_to_full_dict(state)

        # Overlay the agent's explicit updates (these win over the base state)
        full.update(updates)

        # Log revision_count after every node for debugging
        logger.debug(
            "%s → revision_count=%s critic_decision=%s",
            fn.__name__,
            full.get("revision_count"),
            full.get("critic_decision"),
        )
        return full

    wrapped.__name__ = fn.__name__
    return wrapped


# ─────────────────────────────────────────────────────────────────────────────
#  Finalize node
# ─────────────────────────────────────────────────────────────────────────────

def finalize_node(state: ResearchState) -> dict[str, Any]:
    """Save the final approved (or max-revision) report as MD + PDF."""
    final = state.draft or state.final_report

    if state.critic_decision != "APPROVE":
        warning = (
            f"\n\n> ⚠️ **Note:** Maximum revision limit reached "
            f"({state.revision_count} iterations). "
            f"Critic score: {state.critic_score}/10. "
            "Manual review recommended.\n\n"
        )
        final = warning + final

    md_path = ""
    try:
        md_path = save_markdown(final, state.topic)
    except Exception as exc:
        logger.error("Markdown save failed: %s", exc)

    try:
        save_pdf(final, state.topic)
    except Exception as exc:
        logger.error("PDF save failed: %s", exc)

    logger.info("Pipeline done. Report saved: %s", md_path)
    return {"final_report": final, "report_path": md_path, "status": "done"}


# ─────────────────────────────────────────────────────────────────────────────
#  Conditional routing  — reads raw dict, no conversion
# ─────────────────────────────────────────────────────────────────────────────

def route_after_critic(state_dict: dict[str, Any]) -> str:
    """
    Determine the next node after the critic reviews the draft.

    Reads directly from the raw state dict — NO dataclass conversion — so
    revision_count is always read correctly regardless of conversion bugs.

    Termination guarantees (checked in priority order):
      1. critic_decision == "APPROVE"              → finalize
      2. revision_count >= config.max_revisions    → finalize
      3. revision_count >= _EMERGENCY_REV_CAP      → finalize (corruption guard)
      4. Otherwise                                 → back to writer
    """
    try:
        revision_count  = int(state_dict.get("revision_count", 0))
        critic_decision = str(state_dict.get("critic_decision", "REVISE"))
        critic_score    = int(state_dict.get("critic_score", 0))
    except (TypeError, ValueError) as exc:
        logger.error("route_after_critic: could not read state fields (%s) → forcing finalize", exc)
        return "finalize_node"

    logger.info(
        "route_after_critic | decision=%s score=%d revision=%d/%d",
        critic_decision, critic_score, revision_count, config.max_revisions,
    )

    if critic_decision == "APPROVE":
        logger.info("  → FINALIZE (approved, score=%d)", critic_score)
        return "finalize_node"

    if revision_count >= config.max_revisions:
        logger.warning("  → FINALIZE (max_revisions=%d reached)", config.max_revisions)
        return "finalize_node"

    if revision_count >= _EMERGENCY_REV_CAP:
        logger.error("  → FINALIZE (emergency cap %d hit — possible state corruption)", _EMERGENCY_REV_CAP)
        return "finalize_node"

    logger.info("  → REVISE (revision %d/%d)", revision_count, config.max_revisions)
    return "writer_agent"


# ─────────────────────────────────────────────────────────────────────────────
#  Graph factory
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """Construct and compile the LangGraph StateGraph."""
    graph = StateGraph(dict)

    graph.add_node("orchestrator",         _wrap(orchestrator_node))
    graph.add_node("research_agent",       _wrap(research_agent_node))
    graph.add_node("classification_agent", _wrap(classification_agent_node))
    graph.add_node("ner_agent",            _wrap(ner_agent_node))
    graph.add_node("browser_agent",        _wrap(browser_agent_node))
    graph.add_node("analyzer_agent",       _wrap(analyzer_agent_node))
    graph.add_node("illustration_agent",   _wrap(illustration_agent_node))
    graph.add_node("writer_agent",         _wrap(writer_agent_node))
    graph.add_node("critic_agent",         _wrap(critic_agent_node))
    graph.add_node("finalize_node",        _wrap(finalize_node))

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator",          "research_agent")
    graph.add_edge("research_agent",        "classification_agent")
    graph.add_edge("classification_agent",  "ner_agent")
    graph.add_edge("ner_agent",             "browser_agent")
    graph.add_edge("browser_agent",         "analyzer_agent")
    graph.add_edge("analyzer_agent",        "illustration_agent")
    graph.add_edge("illustration_agent",    "writer_agent")
    graph.add_edge("writer_agent",          "critic_agent")
    graph.add_edge("finalize_node",         END)

    # route_after_critic receives the raw state dict (no conversion needed)
    graph.add_conditional_edges(
        "critic_agent",
        route_after_critic,
        {"writer_agent": "writer_agent", "finalize_node": "finalize_node"},
    )

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
#  Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_graph_config() -> dict[str, Any]:
    """LangGraph config dict — pass to every .invoke() / .stream() call."""
    return {"recursion_limit": _RECURSION_LIMIT}


def run_pipeline(topic: str) -> ResearchState:
    """Execute the full pipeline and return the final ResearchState."""
    initial      = ResearchState(topic=topic)
    initial_dict = _state_to_full_dict(initial)   # start with full dict
    app          = build_graph()
    final_dict   = app.invoke(initial_dict, config=get_graph_config())
    return _dict_to_state(final_dict)
