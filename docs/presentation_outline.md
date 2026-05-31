# 10-Minute Live Demo Presentation Outline

## Slide 1 — Title (30 seconds)
**Multi-Agent Research Assistant**  
Powered by Anthropic Claude · LangGraph · SpaCy · Playwright

Team: [Your Name]  
Course: [Course Name]  
Date: [Date]

---

## Slide 2 — Problem Statement (45 seconds)
- Manual research is slow, inconsistent, and hard to scale.
- LLMs alone hallucinate without grounding in real sources.
- **Solution**: A multi-agent pipeline that searches, validates, extracts, and writes — automatically.

---

## Slide 3 — Architecture Overview (60 seconds)
Show the 9-agent pipeline diagram:

```
Orchestrator → Research → Classify → NER → Browser
     → Analyzer → Illustrate → Writer → Critic → Final Report
```

- Highlight the **LangGraph** state machine with conditional routing.
- Show the revision loop (writer ↔ critic).

---

## Slide 4 — Live Demo Part 1: Search & Classify (90 seconds)
1. Open the Streamlit app.
2. Type topic: **"Mixture of Experts in Large Language Models"**
3. Click **Run Research Assistant**.
4. Show real-time progress bar.
5. Open the **Research Agent** panel → show 10+ sources with relevance scores.
6. Open the **Classification Agent** panel → show source type table.

---

## Slide 5 — Live Demo Part 2: NER & Browser (60 seconds)
1. Open the **NER Agent** panel → show entity table (ORG, PERSON, PRODUCT).
2. Open the **Browser Agent** panel → show Playwright screenshot of a real source.
3. Explain: *"This proves the system visited real pages, not just search results."*

---

## Slide 6 — Live Demo Part 3: Report & Critic (90 seconds)
1. Open the **Analyzer Agent** panel → show 5 themes + contradictions.
2. Open the **Writer Agent** panel → show the draft with inline citations [1][2]…
3. Open the **Critic Agent** panel → show score and feedback.
4. If REVISE: show the revision loop in action.
5. Scroll to the **Final Report** → show full structured Markdown.

---

## Slide 7 — Innovation Highlights (45 seconds)
| Feature | Implementation |
|---------|---------------|
| ReAct pattern | Research agent (reason → search → observe → repeat) |
| MCP-style tools | `mcp_tools.py` with JSON-schema descriptors |
| LAM browser | Playwright headless automation with screenshots |
| Cost tracker | Live USD estimate from token usage |
| Revision loop | Critic-driven writer improvement (up to 3 iterations) |
| ChromaDB RAG | Semantic source retrieval for theme evidence |

---

## Slide 8 — Results & Metrics (30 seconds)
Show the benchmark CSV:
- Runtime: < 5 minutes per topic
- Sources: 10–15 per report
- Critic scores: 7–9/10
- Cost: $0.05–$0.20 per report

---

## Slide 9 — Documentation (30 seconds)
- `docs/transformer_analysis.md` — Transformer architecture deep dive
- `docs/moe_comparison.md` — Dense vs. MoE comparison table
- `docs/technical_documentation.md` — Full system architecture
- `README.md` — Setup, run, and competition strategy

---

## Slide 10 — Conclusion & Q&A (30 seconds)
- 9 specialised agents collaborating via a shared state graph.
- Grounded in real web sources with full citation trail.
- Extensible: add new agents, swap LLMs, add new tools.
- **Questions?**
