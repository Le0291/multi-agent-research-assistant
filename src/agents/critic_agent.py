"""
critic_agent.py — Report quality reviewer.

Evaluates the draft report against the competition rubric and returns:
  - A score 1–10
  - A decision: APPROVE or REVISE
  - Specific actionable feedback

If score >= CRITIC_PASS_SCORE → APPROVE (graph routes to finalize)
Else                          → REVISE  (graph routes back to writer)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import config
from src.llm import invoke_claude
from src.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = f"""You are a strict academic report reviewer evaluating against a rubric.

Rubric categories (each 1–10, overall average = final score):
  1. accuracy        — Are facts correct and well-supported by citations?
  2. completeness    — Are all major aspects of the topic covered?
  3. clarity         — Is the writing clear, structured, and academic?
  4. citations       — Are inline citations [N] used consistently?
  5. ner_usage       — Is the Named Entities section present and informative?
  6. classification  — Is the Source Classification section present?
  7. illustrations   — Are figure references present and described?

Return ONLY a JSON object with:
  score:     integer 1-10 (average of categories)
  decision:  "APPROVE" if score >= {config.critic_pass_score}, else "REVISE"
  feedback:  string with specific, actionable improvement instructions
  breakdown: object mapping each category name to its 1-10 sub-score
"""


def critic_agent_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: review the draft and decide to APPROVE or REVISE.

    Input state fields used : draft, classified_sources, entities, topic
    Output state fields set  : critic_feedback, critic_score, critic_decision, status
    """
    logger.info(
        "Critic Agent: reviewing draft (revision %d, length=%d).",
        state.revision_count, len(state.draft),
    )

    # Pass a truncated draft to stay within context limits
    draft_excerpt = state.draft[:6000]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research topic: {state.topic}\n\n"
                f"Number of classified sources: {len(state.classified_sources)}\n"
                f"Number of entities extracted: {len(state.entities)}\n\n"
                f"Draft report (may be truncated):\n\n{draft_excerpt}"
            )
        ),
    ]

    try:
        raw = invoke_claude(messages, cost_metrics=state.cost_metrics, temperature=0.1)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON in critic response")

        result = json.loads(match.group())
        score = int(result.get("score", 5))
        decision = result.get("decision", "REVISE")
        feedback = result.get("feedback", "Please improve the report.")

    except Exception as exc:
        logger.error("Critic Agent parse failed: %s", exc)
        # Conservative fallback: send back to writer unless we've tried enough
        score = 5
        decision = "REVISE"
        feedback = f"Review parsing failed ({exc}). Please ensure all rubric sections are present."

    # Override decision based on configured threshold
    if score >= config.critic_pass_score:
        decision = "APPROVE"
    else:
        decision = "REVISE"

    logger.info("Critic score: %d/10 → %s", score, decision)
    return {
        "critic_score": score,
        "critic_decision": decision,
        "critic_feedback": feedback,
        "revision_count": state.revision_count + 1,
        "status": "finalizing" if decision == "APPROVE" else "revising",
    }
