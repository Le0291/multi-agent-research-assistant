"""
orchestrator.py — Master orchestrator agent.

Responsibilities:
  1. Decompose the research topic into focused sub-questions.
  2. Set initial pipeline status.
  3. Decide which agents run (currently sequential; can be made parallel).

The orchestrator is the first node in the LangGraph.  It populates
state.sub_questions so every downstream agent knows what to look for.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import invoke_claude
from src.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research orchestrator.  Given a research topic,
generate 5–7 focused sub-questions that together would produce a comprehensive
academic report.  Return ONLY a JSON array of strings, no extra text."""


def orchestrator_node(state: ResearchState) -> dict[str, Any]:
    """
    LangGraph node: decompose topic → sub_questions.

    Input state fields used : topic
    Output state fields set  : sub_questions, status
    """
    logger.info("Orchestrator: decomposing topic '%s'", state.topic)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Research topic: {state.topic}"),
    ]

    try:
        raw = invoke_claude(messages, cost_metrics=state.cost_metrics, temperature=0.2)

        # Parse JSON from Claude's response
        start = raw.find("[")
        end = raw.rfind("]") + 1
        sub_questions: list[str] = json.loads(raw[start:end]) if start != -1 else []
    except Exception as exc:
        logger.warning("Orchestrator JSON parse failed: %s", exc)
        # Fallback: generate basic sub-questions from the topic
        sub_questions = [
            f"What is {state.topic}?",
            f"What are the key components of {state.topic}?",
            f"What are the latest developments in {state.topic}?",
            f"What are the challenges and limitations of {state.topic}?",
            f"What are the real-world applications of {state.topic}?",
        ]

    logger.info("Orchestrator generated %d sub-questions.", len(sub_questions))
    return {
        "sub_questions": sub_questions,
        "status": "researching",
    }
