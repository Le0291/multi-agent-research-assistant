# Multi-Agent Research Assistant

A competition-ready, production-quality multi-agent research pipeline built with
**Anthropic Claude**, **LangGraph**, **LangChain**, **SpaCy**, and **Playwright**.

---

## Two Operating Modes

### Mode 1 — Full Pipeline *(default · competition requirement)*

Runs all **9 agents in sequence** via LangGraph and produces a complete cited
Markdown + PDF report.  This is the mandatory mode required by the course
project specification.

### Mode 2 — Agent Playground *(innovation feature)*

Run **any single agent in isolation** directly from the sidebar.  Dependencies
are resolved automatically — you only see the output of the agent you selected.
This lets you inspect each agent's behaviour, debug it independently, and
impress the demo audience by showing exactly what each component contributes.

> Switching between modes is instant — use the sidebar dropdown.

---

## Project Overview

Given any research topic, the system orchestrates **9 specialised AI agents** that:

1. Decompose the topic into focused sub-questions
2. Search and scrape ≥10 real web sources
3. Classify sources by type, domain, and quality
4. Extract named entities (people, orgs, technologies, …)
5. Enrich sources via real browser automation (Playwright)
6. Synthesise 4–6 themes with evidence
7. Generate academic figures (matplotlib / DALL-E)
8. Write a polished, fully-cited Markdown report
9. Review the report against a rubric (score 1–10); revise if needed

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph StateGraph (src/graph.py)         │
│                                                         │
│  [orchestrator] → [research_agent] → [classification]   │
│       → [ner_agent] → [browser_agent] → [analyzer]      │
│       → [illustration] → [writer_agent] → [critic]      │
│              ↑                    ↓ REVISE               │
│              └──────── (revision loop, max 3) ──────────┘
│                                   ↓ APPROVE              │
│                           [finalize_node]                │
└─────────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
         reports/               generated_images/
         *.md  *.pdf             *.png
```

---

## Agent Descriptions

| Agent | File | Role |
|-------|------|------|
| Orchestrator | `src/agents/orchestrator.py` | Decomposes topic into 5–7 sub-questions |
| Research Agent | `src/agents/research_agent.py` | ReAct-style web search + scraping (≥10 sources) |
| Classification Agent | `src/agents/classification_agent.py` | Labels source_type, domain, relevance; filters low-quality |
| NER Agent | `src/agents/ner_agent.py` | SpaCy NER → entity frequency table + co-occurrence |
| Browser Agent | `src/agents/browser_agent.py` | Playwright LAM — screenshots, structured metadata |
| Analyzer Agent | `src/agents/analyzer_agent.py` | Themes, contradictions, outline, image prompts |
| Illustration Agent | `src/agents/illustration_agent.py` | Matplotlib placeholders or DALL-E figures |
| Writer Agent | `src/agents/writer_agent.py` | Full Markdown report with inline citations |
| Critic Agent | `src/agents/critic_agent.py` | Rubric score 1–10; APPROVE or REVISE decision |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Yes | Your Anthropic API key |
| `TAVILY_API_KEY` | Recommended | Tavily search (best quality) |
| `BRAVE_API_KEY` | Optional | Brave Search fallback |
| `OPENAI_API_KEY` | Optional | DALL-E 3 image generation |
| `ANTHROPIC_MODEL` | Optional | Default: `claude-3-5-sonnet-20241022` |
| `MIN_SOURCES` | Optional | Default: `10` |
| `MAX_REVISIONS` | Optional | Default: `3` |
| `CRITIC_PASS_SCORE` | Optional | Default: `7` |

---

## Setup Instructions

### Step 1 — Clone / download the project

```bash
cd "C:\Users\leo\Desktop\final project\multi_agent_research_assistant"
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Download SpaCy model

```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Install Playwright browsers (optional, for browser automation)

```bash
playwright install chromium
```

### Step 6 — Configure environment variables

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` and add your `ANTHROPIC_API_KEY` (and optionally `TAVILY_API_KEY`).

### Step 7 — Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## How to Run the Benchmark

```bash
python -m src.evaluation.benchmark
```

Results are saved to `benchmark_results.csv`.

---

## Competition Strategy

### Rubric coverage

| Requirement | Implementation |
|-------------|---------------|
| Report quality | Writer + Critic revision loop; academic Markdown formatting |
| ≥10 diverse sources | Research agent collects 10–15, scored 1–10 |
| Inline citations | Every factual claim tagged [N] |
| NER | SpaCy en_core_web_sm + entity frequency table |
| Source classification | Claude + keyword fallback; 8 source types |
| Illustrations | matplotlib (always) + DALL-E (optional) |
| Speed | < 5 min per topic with Tavily |
| Cost efficiency | Local NER/embedding; cost tracker; cheap sentence-transformers |
| Error handling | `@safe_node`, API fallbacks, never crashes |
| Innovation | ReAct pattern, MCP tools, LAM browser, ChromaDB RAG, revision loop |
| Clean code | Type hints, module docstrings, inline comments |
| Documentation | README + 3 deep-dive docs + presentation outline |

### Innovation highlights

- **ReAct research loop**: reason → search → scrape → score → repeat
- **MCP-style tools**: JSON-schema descriptors compatible with Claude's tool_use API
- **LAM browser**: Playwright headless Chromium with screenshots as proof
- **Cost tracker**: Live USD estimate shown in Streamlit
- **Revision loop**: Critic-driven writer improvement (up to 3 iterations)
- **ChromaDB RAG**: Semantic retrieval for evidence-backed theme synthesis
- **MoE analysis**: Dedicated section in every AI report comparing dense vs. MoE
- **Agent Playground**: Run any single agent in isolation with auto dependency resolution

---

## Agent Playground — Full Reference

### Why this feature exists

The Playground lets you test and demonstrate each agent independently, without
running the entire pipeline every time.  This is valuable for:

- **Debugging**: isolate an agent's behaviour without 5-minute wait times
- **Demo**: show the audience exactly what NER, classification, or browser automation does
- **Development**: test prompt changes to a single agent quickly
- **Innovation score**: demonstrates architectural maturity and modularity

### Available playground modes

| Mode | Input needed | Auto-resolves dependencies | Output |
|------|-------------|---------------------------|--------|
| Research Only | Topic | Orchestrator (sub-questions) | Sources + relevance scores |
| Classification Only | Topic or URLs | Research Agent (if no URLs) | Source type/domain/relevance table |
| NER Only | Topic or text | Research Agent (if no text) | Entity table + co-occurrence pairs |
| Browser Only | URLs | None | Metadata + screenshots |
| Analysis Only | Topic | Research + Classification + NER | Themes, contradictions, outline |
| Writer Only | Topic | Full chain up to Illustration | Markdown draft |
| Critic Only | Draft text | None | Score 1–10 + feedback |
| Illustration Only | Prompts or topic | Analysis (if no prompts) | PNG figures |

### Smart dependency resolution

The router (`src/router.py`) automatically runs the minimum required
predecessor agents before the selected agent.  For example:

```
User selects: NER Only
User provides: topic text (no raw text)

Router automatically runs:
  1. Orchestrator  → generates sub-questions
  2. Research Agent → fetches web sources
  3. NER Agent     → extracts entities  ← shown to user
```

The user only sees the NER output.  The intermediate steps happen silently.

### How routing decisions are made

```python
# src/router.py
if mode == "Full Pipeline":
    run via LangGraph graph (all 9 agents)
elif mode == "NER Only":
    if text provided: run NER directly
    else: orchestrator → research → NER
elif mode == "Critic Only":
    run critic directly on the provided draft
# etc.
```

---

## Screenshots

*(Run the app and add screenshots here before your presentation)*

| Screenshot | Description |
|-----------|-------------|
| `docs/screenshot_full_pipeline.png` | Full Pipeline streaming progress |
| `docs/screenshot_playground_ner.png` | NER Only mode — entity table |
| `docs/screenshot_playground_classification.png` | Classification result |
| `docs/screenshot_playground_browser.png` | Browser automation + screenshot |
| `docs/screenshot_playground_critic.png` | Critic score panel |
| `docs/screenshot_final_report.png` | Complete generated report |

---

## Known Limitations

- Playwright requires a separate browser binary download (`playwright install chromium`).
- DALL-E image generation requires an OpenAI API key.
- DuckDuckGo fallback search may return fewer/lower-quality results than Tavily.
- Very long documents may be truncated to fit Claude's context window.
- PDF export on Windows uses ReportLab (no WeasyPrint dependency).

---

## File Structure

```
multi_agent_research_assistant/
├── app.py                        ← Streamlit UI entry point
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── config.py                 ← All settings from environment
│   ├── state.py                  ← Shared ResearchState schema
│   ├── graph.py                  ← LangGraph pipeline definition
│   ├── router.py                 ← Smart mode router + dependency resolution
│   ├── llm.py                    ← Anthropic Claude client
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── research_agent.py
│   │   ├── classification_agent.py
│   │   ├── ner_agent.py
│   │   ├── browser_agent.py
│   │   ├── analyzer_agent.py
│   │   ├── writer_agent.py
│   │   ├── illustration_agent.py
│   │   └── critic_agent.py
│   ├── tools/
│   │   ├── search_tool.py        ← Tavily / Brave / DDG
│   │   ├── scrape_tool.py        ← httpx + BeautifulSoup
│   │   ├── browser_tool.py       ← Playwright LAM
│   │   ├── mcp_tools.py          ← Tool registry + dispatcher
│   │   └── vector_store.py       ← ChromaDB RAG
│   ├── utils/
│   │   ├── citations.py
│   │   ├── cost_tracker.py
│   │   ├── error_handler.py
│   │   └── report_exporter.py    ← Markdown + PDF
│   └── evaluation/
│       ├── benchmark.py
│       ├── classification_eval.py
│       └── ner_eval.py
├── reports/                      ← Generated .md and .pdf files
├── generated_images/             ← Figures and screenshots
└── docs/
    ├── transformer_analysis.md
    ├── moe_comparison.md
    ├── technical_documentation.md
    └── presentation_outline.md
```

---

## Screenshots

*(Run the app and add screenshots here before your presentation)*

- `docs/screenshot_ui.png` — Main Streamlit interface
- `docs/screenshot_report.png` — Sample generated report
- `docs/screenshot_ner.png` — Entity extraction panel
- `docs/screenshot_browser.png` — Playwright browser panel
