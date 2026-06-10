# Multi-Agent Research Assistant

A competition-ready, production-quality multi-agent research pipeline built with
**Anthropic Claude**, **LangGraph**, **LangChain**, **SpaCy**, **Playwright**,
**OpenAI gpt-image**, and **Streamlit**.

Give it any research topic and it orchestrates **9 specialised AI agents** to
produce a fully-cited academic report (Markdown + PDF) complete with named-entity
analysis, data-driven figures, and a self-critique revision loop.

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

> Switch modes instantly from the sidebar navigation buttons.

---

## Project Overview

Given any research topic, the system orchestrates **9 specialised AI agents** that:

1. Decompose the topic into focused sub-questions
2. Search and scrape ≥10 real web sources
3. Classify sources by type, domain, and quality
4. Extract named entities (people, orgs, technologies, …) with a multi-layer filter
5. Enrich the top source via real browser automation (Playwright)
6. Synthesise 4–6 themes with supporting evidence
7. Generate academic figures — **data-driven Matplotlib charts** by default
   (real legible labels from pipeline data); optional OpenAI **gpt-image-1 /
   gpt-image-2** when `USE_AI_IMAGES=true`
8. Write a polished, fully-cited Markdown report
9. Review the report against a rubric (score 1–10); revise if needed

---

## Key Features

- 🧠 **9-agent LangGraph pipeline** with a critic-driven revision loop (max 3)
- 🔍 **ReAct research loop** — reason → search → scrape → score → repeat
- 🏷️ **High-precision NER** — SpaCy + an 8-layer filter that strips URLs, table
  garbage, stop-words, and sentence fragments, then remaps tech terms & dedupes
- 🌐 **LAM browser automation** — Playwright headless Chromium with screenshots
- 🎨 **Topic-centric figures** — Matplotlib charts about the topic itself
  (knowledge map, research scope, theme map) with real legible labels;
  embedded inside the exported MD (base64) and PDF (real images);
  optional OpenAI gpt-image-1/2 via `USE_AI_IMAGES=true`
- 🌓 **Dark / Light theme toggle** — fully themed Stitch-style UI, zero white flashes
- 📚 **Run history** — the last 3 runs persisted to disk (survive page refreshes)
- ⬇️ **Per-figure & report downloads** — PNG figures, Markdown, and PDF export
- 💰 **Live cost tracker** — real-time USD/token estimate in the UI
- 🛡️ **Never crashes** — per-node try/except + a graph-level node wrapper, request timeouts, graceful degradation
- ☁️ **Deploy-ready** — Railway config (`railway.toml`) with Playwright auto-install

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                │
│        sidebar.py · pipeline.py · playground.py          │
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
| NER Agent | `src/agents/ner_agent.py` | SpaCy NER → cleaned entity frequency table + co-occurrence |
| Browser Agent | `src/agents/browser_agent.py` | Playwright LAM — screenshot + structured metadata (thread-isolated, timed out) |
| Analyzer Agent | `src/agents/analyzer_agent.py` | Themes, contradictions, outline, image prompts |
| Illustration Agent | `src/agents/illustration_agent.py` | Data-driven Matplotlib charts (default); optional gpt-image via `USE_AI_IMAGES` |
| Writer Agent | `src/agents/writer_agent.py` | Full Markdown report with inline citations |
| Critic Agent | `src/agents/critic_agent.py` | Rubric score 1–10; APPROVE or REVISE decision |

---

## Environment Variables

Create a `.env` file in the project root (it is git-ignored — never commit it):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Yes | — | Your Anthropic API key |
| `TAVILY_API_KEY` | Recommended | — | Tavily search (best quality) |
| `BRAVE_API_KEY` | Optional | — | Brave Search fallback |
| `USE_AI_IMAGES` | Optional | `false` | `true` = gpt-image figures; `false` = legible Matplotlib charts |
| `OPENAI_API_KEY` | Optional | — | Required only when `USE_AI_IMAGES=true` |
| `OPENAI_IMAGE_MODEL` | Optional | `gpt-image-1` | `gpt-image-1` or `gpt-image-2` (when AI images on) |
| `OPENAI_IMAGE_QUALITY` | Optional | `medium` | `low` / `medium` / `high` / `auto` (when AI images on) |
| `ANTHROPIC_MODEL` | Optional | `claude-haiku-4-5-20251001` | Claude model |
| `ANTHROPIC_MAX_TOKENS` | Optional | `8192` | Max output tokens |
| `ANTHROPIC_TEMPERATURE` | Optional | `0.3` | Sampling temperature |
| `MIN_SOURCES` | Optional | `10` | Minimum sources collected |
| `MAX_SOURCES` | Optional | `15` | Maximum sources collected |
| `MAX_REVISIONS` | Optional | `3` | Critic→Writer revision cap |
| `CRITIC_PASS_SCORE` | Optional | `7` | Score needed to APPROVE |
| `BROWSER_VISIT_COUNT` | Optional | `3` | Pages the browser agent visits |

> **Note on figures:** By default the app generates **data-driven Matplotlib
> charts** with real, legible labels built from your pipeline data (entities,
> sources, themes) — ideal for an academic report. To use AI image generation
> instead, set `USE_AI_IMAGES=true` and provide `OPENAI_API_KEY`.  (AI models
> render text as gibberish, which is why charts are the default.)  Note that
> `dall-e-3` is unavailable on new accounts, so the **gpt-image** family is used;
> it returns base64 image data which the agent decodes automatically.

### Approximate AI-image cost (only when `USE_AI_IMAGES=true`, per figure 1024×1024)

| Quality | Cost / image | Per run (3 figures) |
|---------|--------------|---------------------|
| `low` | ~$0.02 | ~$0.06 |
| `medium` *(default)* | ~$0.04 | ~$0.12 |
| `high` | ~$0.17 | ~$0.51 |

---

## Setup Instructions (local)

### Step 1 — Open the project

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

### Step 4 — Download the SpaCy model

```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Install the Playwright browser (for browser automation)

```bash
playwright install chromium
```

### Step 6 — Configure environment variables

Create a `.env` file in the project root and add at least your Anthropic key:

```env
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
# Figures are legible Matplotlib charts by default. To use AI images instead:
USE_AI_IMAGES=false               # set true to enable gpt-image
OPENAI_API_KEY=sk-proj-...        # required only if USE_AI_IMAGES=true
OPENAI_IMAGE_MODEL=gpt-image-1    # optional (gpt-image-1 | gpt-image-2)
OPENAI_IMAGE_QUALITY=medium       # optional (low | medium | high | auto)
```

### Step 7 — Run the Streamlit app

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Deployment (Railway)

The repo is configured for one-click deployment on **Railway**.

1. Connect the GitHub repo to a new Railway project (it deploys from `main`).
2. In **Settings → Variables**, add your keys:
   `ANTHROPIC_API_KEY`, `TAVILY_API_KEY`, and (optionally) `OPENAI_API_KEY`,
   `OPENAI_IMAGE_MODEL`, `OPENAI_IMAGE_QUALITY`.
3. Deploy. The start command (`railway.toml`) installs the Playwright browser
   before launching Streamlit:

   ```toml
   [deploy]
   startCommand = "playwright install chromium --with-deps && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"
   ```

> Every push to `main` triggers an automatic rebuild and redeploy. After
> changing any environment variable, redeploy so the running process picks it up.

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
| NER | SpaCy `en_core_web_sm` + 8-layer cleaning filter + frequency table |
| Source classification | Claude + keyword fallback; 8 source types |
| Illustrations | Matplotlib data charts (default, always legible) + optional gpt-image |
| Speed | < 5 min per topic with Tavily |
| Cost efficiency | Local NER/embedding; live cost tracker; configurable image quality |
| Error handling | Per-node try/except + graph node wrapper, request timeouts, API fallbacks, never crashes |
| Innovation | ReAct pattern, MCP tools, LAM browser, ChromaDB RAG, revision loop |
| Clean code | Type hints, module docstrings, inline comments |
| Documentation | README + deep-dive docs + presentation outline |

### Innovation highlights

- **ReAct research loop**: reason → search → scrape → score → repeat
- **MCP-style tools**: JSON-schema descriptors compatible with Claude's tool_use API
- **LAM browser**: Playwright headless Chromium with screenshots as proof
- **High-precision NER**: multi-layer filter pipeline for clean, meaningful entities
- **Data-driven figures**: Matplotlib charts from real pipeline data with legible labels (optional gpt-image via `USE_AI_IMAGES`)
- **Cost tracker**: live USD estimate shown in Streamlit
- **Revision loop**: critic-driven writer improvement (up to 3 iterations)
- **ChromaDB RAG**: sources embedded locally (sentence-transformers); the analyzer retrieves the most relevant passages per sub-question to ground its theme synthesis
- **Run history**: instant recall of the last 3 reports within a session
- **Agent Playground**: run any single agent in isolation with auto dependency resolution

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

---

## Known Limitations

- Playwright requires a separate browser binary (`playwright install chromium`);
  on Railway this is handled automatically by `railway.toml`.
- Figures are legible Matplotlib charts by default. AI images are opt-in
  (`USE_AI_IMAGES=true` + `OPENAI_API_KEY`) — note AI models render text as
  gibberish. `dall-e-3` is unavailable on new accounts, so the **gpt-image**
  family is used.
- DuckDuckGo fallback search may return fewer/lower-quality results than Tavily.
- Very long documents may be truncated to fit Claude's context window.
- PDF export on Windows uses ReportLab (WeasyPrint is Linux/macOS only).
- The browser agent visits the top 3 sources by default (`BROWSER_VISIT_COUNT`);
  lower it to 1 on very memory-constrained hosts.

---

## File Structure

```
multi_agent_research_assistant/
├── app.py                        ← Streamlit UI entry point
├── requirements.txt
├── railway.toml                  ← Railway deploy config (Playwright + Streamlit)
├── nixpacks.toml                 ← Nixpacks build config
├── Procfile / runtime.txt        ← Process + Python runtime pins
├── README.md
├── .streamlit/
│   └── config.toml               ← Native dark theme tokens
├── src/
│   ├── config.py                 ← All settings from environment
│   ├── state.py                  ← Shared ResearchState schema
│   ├── graph.py                  ← LangGraph pipeline definition
│   ├── router.py                 ← Smart mode router + dependency resolution
│   ├── llm.py                    ← Anthropic Claude client (90s timeout)
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── research_agent.py
│   │   ├── classification_agent.py
│   │   ├── ner_agent.py
│   │   ├── browser_agent.py
│   │   ├── analyzer_agent.py
│   │   ├── illustration_agent.py ← Matplotlib data charts (default) + optional gpt-image
│   │   ├── writer_agent.py
│   │   └── critic_agent.py
│   ├── tools/
│   │   ├── search_tool.py        ← Tavily / Brave / DDG
│   │   ├── scrape_tool.py        ← httpx + BeautifulSoup
│   │   ├── browser_tool.py       ← Playwright LAM (thread-isolated + timeout)
│   │   ├── mcp_tools.py          ← Tool registry + dispatcher
│   │   └── vector_store.py       ← ChromaDB RAG: index + semantic retrieval
│   ├── ui/
│   │   ├── sidebar.py            ← Nav buttons, theme toggle, recent runs
│   │   ├── pipeline.py           ← Full Pipeline UI, history, downloads
│   │   ├── playground.py         ← Single-agent playground
│   │   ├── components.py         ← Shared widgets (file upload, entity table)
│   │   ├── theme.py              ← Theme/CSS injection
│   │   └── styles/
│   │       ├── dark.css
│   │       ├── light.css
│   │       └── shared.css
│   ├── utils/
│   │   ├── citations.py
│   │   ├── cost_tracker.py
│   │   ├── error_handler.py      ← error fallbacks + API-key checks
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

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Anthropic Claude (via `langchain-anthropic`) |
| Orchestration | LangGraph `StateGraph` |
| Search | Tavily · Brave · DuckDuckGo fallback |
| Scraping | httpx + BeautifulSoup |
| Browser automation | Playwright (headless Chromium) |
| NER / NLP | SpaCy `en_core_web_sm` |
| Vector store / RAG | ChromaDB + sentence-transformers |
| Image generation | OpenAI gpt-image-1 / gpt-image-2 |
| Charts | Matplotlib |
| UI | Streamlit |
| Export | ReportLab (PDF) · Markdown |
| Deployment | Railway |
