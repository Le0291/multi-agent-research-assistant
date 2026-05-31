# Mixture of Experts: Comparative Analysis

## 1. Dense Transformers vs. Mixture of Experts

### 1.1 Dense Transformers

In a standard ("dense") Transformer, **every parameter is activated for every token**.
If a model has 70 billion parameters, all 70B are involved in processing each token.

- **Pros**: Simple training, predictable behaviour, strong per-token compute.
- **Cons**: Inference cost scales linearly with parameter count; diminishing returns
  at very large sizes.

### 1.2 Mixture of Experts (MoE)

MoE replaces the dense Feed-Forward Network (FFN) in each Transformer layer with a
collection of *N* independent "expert" FFNs.  A lightweight **gating network** routes
each token to only *k* of the *N* experts (typically k=1 or k=2).

```
MoE(x) = Σᵢ Gᵢ(x) · Eᵢ(x)
```

Where `Gᵢ(x)` is the gating weight for expert *i* and `Eᵢ(x)` is expert *i*'s output.

Key insight: **total parameters ↑, but activated parameters per token stay small**.

---

## 2. Major MoE Models

### 2.1 Switch Transformer (Google, 2021)

- **Experts per layer**: 64–2048
- **Routing**: Top-1 (each token goes to exactly 1 expert)
- **Scale**: Up to 1.6 trillion parameters
- **Innovation**: Simplified sparse routing with an auxiliary load-balancing loss to
  prevent all tokens routing to the same expert (expert collapse).
- **Result**: 7× faster pre-training than T5-XXL at equivalent quality.

### 2.2 GLaM (Google, 2022)

- **Experts**: 64 per layer, top-2 routing
- **Parameters**: 1.2T total, ~97B activated per token
- **Efficiency**: Matches GPT-3 quality with 1/3 the energy consumption during training.

### 2.3 Mixtral 8×7B (Mistral AI, 2024)

- **Architecture**: 8 experts per layer, top-2 routing
- **Activated parameters**: ~12.9B (out of 46.7B total)
- **Performance**: Matches or exceeds Llama 2 70B on most benchmarks while being 6×
  faster at inference.
- **Open weights**: Released publicly; runs on consumer hardware.
- **Key advantage**: Very competitive cost/quality tradeoff for deployment.

### 2.4 Mixtral 8×22B (Mistral AI, 2024)

- **Experts**: 8 per layer, 39B activated per token out of 141B total
- **Capabilities**: Strong coding, maths, and multilingual reasoning
- **Context**: 64K token context window

### 2.5 DeepSeek-MoE (DeepSeek AI, 2024)

- **Innovation**: Fine-grained expert segmentation — more, smaller experts with
  higher-granularity routing.
- **Shared experts**: Some experts are always activated (shared knowledge);
  others are routed (specialised knowledge).
- **Result**: Higher expert specialisation and less redundancy than coarse MoE.
- **Scale**: DeepSeek-V2 uses 236B total parameters, 21B activated.

### 2.6 Grok-1 (xAI, 2024)

- **Architecture**: 314B total parameters, MoE with 8 experts, top-2 routing
- **Open weights**: Released by Elon Musk's xAI.
- **Notable**: One of the largest open-weight MoE models.

---

## 3. Cost / Performance Tradeoffs

| Model | Total Params | Activated Params | Inference Cost | Quality (vs. Dense) |
|-------|-------------|-----------------|----------------|---------------------|
| LLaMA 2 70B (dense) | 70B | 70B | High | Baseline |
| Mixtral 8×7B | 46.7B | 12.9B | Low | ≥ LLaMA 2 70B |
| Switch-C (1.6T) | 1.6T | ~110B | Medium | Strong on NLP tasks |
| DeepSeek-V2 | 236B | 21B | Very Low | Competitive with GPT-4 class |
| Grok-1 | 314B | ~87B | High | Strong coding/reasoning |

### Key Tradeoffs

1. **Communication overhead**: In distributed inference, each token must be routed to
   the correct expert, which may reside on a different GPU.  This "all-to-all"
   communication can bottleneck MoE at large batch sizes.

2. **Load balancing**: Without a balancing loss, some experts receive far more tokens
   than others ("expert collapse"), wasting capacity.

3. **Training stability**: MoE models can be harder to train — routing decisions are
   discrete (non-differentiable), requiring tricks like soft routing or straight-through
   estimators.

4. **Memory**: The total parameter count must fit in VRAM even if only k/N experts run
   per token.  A 1.6T Switch Transformer requires enormous distributed memory.

---

## 4. Relevance to Multi-Agent Research Systems

In an agentic system like ours, MoE offers an interesting architectural metaphor:

| MoE Concept | Research Agent Analogue |
|-------------|------------------------|
| Expert FFN  | Specialised sub-agent (NER agent, classification agent, …) |
| Gating network | Orchestrator (routes the task to the right agent) |
| Sparse activation | Not all agents run for every sub-task |
| Shared experts | Shared tools (search, scrape) used by all agents |

This "agents as experts" framing suggests that multi-agent systems are a natural
software-level instantiation of the MoE principle: parallelism through specialisation.

---

## 5. Conclusion

MoE architectures represent the frontier of scalable LLM design.  By decoupling
parameter count from per-token compute, they allow models to specialise different
experts for different input types while maintaining inference efficiency.  DeepSeek-V2
and Mixtral demonstrate that open-weight MoE models are competitive with dense models
many times their activated-parameter size — a crucial finding for cost-effective
deployment of large-scale AI research systems.
