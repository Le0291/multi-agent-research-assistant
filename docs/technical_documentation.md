# Technical Documentation

## 1. System Architecture Overview

```
User (Streamlit UI)
       │
       ▼
  app.py  ──────────────────────────────────────────────────
       │
       ▼
  src/graph.py  ← LangGraph StateGraph (pipeline orchestration)
       │
       ├─► orchestrator.py        (sub-question decomposition)
       ├─► research_agent.py      (ReAct web search + scraping)
       ├─► classification_agent.py(source type + quality filter)
       ├─► ner_agent.py           (SpaCy NER + entity registry)
       ├─► browser_agent.py       (Playwright LAM automation)
       ├─► analyzer_agent.py      (theme synthesis + outline)
       ├─► illustration_agent.py  (matplotlib / DALL-E figures)
       ├─► writer_agent.py        (Markdown report generation)
       ├─► critic_agent.py        (rubric-based review + routing)
       └─► finalize_node          (save MD + PDF)
       │
       ▼
  src/tools/                  src/utils/
  ├── search_tool.py          ├── citations.py
  ├── scrape_tool.py          ├── cost_tracker.py
  ├── browser_tool.py         ├── error_handler.py
  ├── mcp_tools.py            └── report_exporter.py
  └── vector_store.py
```

---

## 2. LangGraph Workflow

### 2.1 What is LangGraph?

LangGraph is a library for building stateful, multi-agent workflows as directed graphs.
Each **node** is a Python function that receives the shared state and returns partial
updates.  **Edges** define the flow between nodes.  **Conditional edges** allow dynamic
routing based on state values.

### 2.2 State Management

We use a single `ResearchState` dataclass (in `src/state.py`) as the shared memory.
LangGraph serialises it as a dict between nodes.  Each node only reads what it needs
and writes back only the fields it modifies.

### 2.3 Conditional Routing

After the critic node, the graph branches:

```python
graph.add_conditional_edges(
    "critic_agent",
    route_after_critic,          # Returns "writer_agent" or "finalize_node"
    {"writer_agent": ..., "finalize_node": ...}
)
```

`route_after_critic()` checks:
- `critic_decision == "APPROVE"` → finalize
- `revision_count >= MAX_REVISIONS` → finalize (with warning)
- Otherwise → back to writer

---

## 3. LangChain Usage

We use LangChain for:
- `ChatAnthropic` — the Anthropic Claude wrapper with retry logic and streaming support.
- `HumanMessage` / `SystemMessage` / `AIMessage` — standardised message types.
- `langchain_core.runnables` — the `.invoke()` interface used uniformly across agents.

We do **not** use LangChain's deprecated `AgentExecutor` or `Chain` classes — the
pipeline is managed explicitly by LangGraph for full observability.

---

## 4. MCP-Style Tools

The Model Context Protocol (MCP) standardises how LLMs invoke external tools via
JSON-schema descriptors.  In `src/tools/mcp_tools.py`, each tool has:

```python
TOOL_SPEC = {
    "name": "web_search",
    "description": "Search the web for information.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}
```

The `call_tool(name, **kwargs)` dispatcher routes calls to the appropriate Python
function.  This pattern allows tool specs to be passed directly to Claude's
`tool_use` API for autonomous tool selection.

---

## 5. Named Entity Recognition (NER)

### Pipeline
1. **SpaCy** (`en_core_web_sm`) processes each source's text.
2. Entities are mapped from SpaCy labels to semantic categories:
   - `ORG` → `organization`
   - `PERSON` → `person`
   - `PRODUCT` / `WORK_OF_ART` → `technology`
   - `GPE` / `LOC` → `location`
3. **Entity frequency table**: counts how many sources each entity appears in.
4. **Source-to-entity map**: records which URLs mention each entity.
5. **Co-occurrence relationships**: entity pairs that appear in the same sentence.

### Why SpaCy?
- No API cost (runs locally).
- Fast (processes thousands of tokens per second on CPU).
- Accurate for standard named entities.
- Supports custom entity rulers for domain-specific terms.

---

## 6. Source Classification

The classification agent uses Claude to classify each source into:
- **source_type**: `academic_paper | documentation | blog | news | tutorial | dataset | opinion | unknown`
- **domain**: `technical | business | healthcare | education | AI | policy | general`
- **relevance**: `high | medium | low`

Keyword-based heuristics provide an instant fallback if the Claude call fails.
Low-relevance sources are logged and removed from the active set.

---

## 7. Browser Automation (LAM)

**Playwright** provides headless Chromium automation.  The browser agent:
1. Navigates to the top-3 classified source URLs.
2. Extracts structured metadata: title, Open Graph tags, headings (h1–h2).
3. Takes a screenshot saved to `generated_images/`.
4. Returns richer body text (for JS-rendered pages that static HTTP misses).

This demonstrates **Large Action Model (LAM)** behaviour — the model instructs
the browser to perform real-world actions beyond text generation.

Playwright gracefully degrades: if the binary is not installed, the agent logs a
warning and skips browser enrichment without crashing the pipeline.

---

## 8. Illustration Component

Three-tier image generation strategy:

| Tier | Method | Requires |
|------|--------|---------|
| 1    | DALL-E 3 (OpenAI API) | `OPENAI_API_KEY` |
| 2    | Matplotlib placeholder | Python only |
| 3    | Mermaid code blocks   | None (embedded in Markdown) |

Figures are embedded in the report as:
```markdown
![Figure 1](generated_images/figure_1_topic.png)
```

---

## 9. Vector Store (ChromaDB)

After the research agent collects sources, they are embedded with
`sentence-transformers/all-MiniLM-L6-v2` (local model, no API cost) and stored in
a persistent ChromaDB collection.

The analyzer agent can query the store with a theme or sub-question to retrieve the
most semantically relevant source chunks — enabling evidence-based theme synthesis
rather than purely prompt-based reasoning.

---

## 10. Error Handling Philosophy

Every potential failure point has an explicit handler:

| Error Type | Handler |
|------------|---------|
| Missing API key | `check_api_keys()` in UI + `validate()` in config |
| Search API failure | Fallback to next provider, then DuckDuckGo |
| Scrape failure | Log and continue with snippet only |
| Playwright unavailable | Skip browser enrichment, log warning |
| Claude rate limit | `max_retries=3` with exponential backoff |
| Empty source list | Log warning, continue with available sources |
| JSON parse failure | Regex extraction fallback + hardcoded defaults |
| PDF export failure | Fall back to plain-text PDF |

The `@safe_node` decorator catches unhandled exceptions in any node and records
them to `state.errors` without crashing the graph.

---

## 11. Testing & Benchmarking

### Unit-level evaluation
- `src/evaluation/classification_eval.py` — Precision/recall on 5 labelled sources.
- `src/evaluation/ner_eval.py` — P/R/F1 on 3 gold-standard sentences.

### End-to-end benchmarking
- `src/evaluation/benchmark.py` — Runs all 5 topics and records to `benchmark_results.csv`.

### Metrics recorded per run
- Runtime (seconds)
- Number of sources (raw and classified)
- Report length (chars)
- Token usage and estimated cost
- Critic score
- Revision count
- Source type distribution

---

## 12. Configuration Reference

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | **Required** |
| `TAVILY_API_KEY` | — | Primary search |
| `BRAVE_API_KEY` | — | Secondary search |
| `OPENAI_API_KEY` | — | Optional DALL-E images |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20241022` | Model ID |
| `ANTHROPIC_MAX_TOKENS` | `8192` | Max output tokens |
| `MIN_SOURCES` | `10` | Minimum sources required |
| `MAX_REVISIONS` | `3` | Max writer→critic loops |
| `CRITIC_PASS_SCORE` | `7` | Minimum score to approve |
