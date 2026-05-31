# Transformer Architecture Analysis

## 1. Introduction

The Transformer architecture, introduced by Vaswani et al. in the landmark 2017 paper
*"Attention Is All You Need"*, fundamentally changed the landscape of natural language
processing and, subsequently, all of AI.  Unlike recurrent neural networks (RNNs) or
convolutional networks, the Transformer relies **entirely on attention mechanisms** to
model relationships between tokens, regardless of their distance in the sequence.

---

## 2. Core Components

### 2.1 Self-Attention

Self-attention allows every token in a sequence to "attend to" every other token.
Each token is projected into three vectors:

| Vector | Symbol | Role |
|--------|--------|------|
| Query  | Q      | "What am I looking for?" |
| Key    | K      | "What do I contain?" |
| Value  | V      | "What information do I carry?" |

The attention weight between position *i* and *j* is:

```
Attention(Q, K, V) = softmax( QKᵀ / √dₖ ) · V
```

The `√dₖ` scaling factor prevents the dot products from growing too large and
collapsing the softmax distribution.

**Multi-Head Attention** runs *h* parallel attention heads, each with different learned
projections.  This allows the model to simultaneously attend to information from different
representational subspaces.

### 2.2 Feed-Forward Sub-Layer

Each Transformer layer also contains a position-wise feed-forward network (FFN):

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

The FFN is applied independently to each token position, doubling the model's capacity
to transform representations after attention.

### 2.3 Positional Encoding

Because self-attention is permutation-invariant, positional encodings are added to
the token embeddings to inject sequence order.  The original paper used sinusoidal
encodings; modern models use learned positional embeddings or Rotary Position Embeddings
(RoPE).

### 2.4 Layer Normalisation & Residual Connections

Each sub-layer is wrapped with:
1. A **residual (skip) connection** — adds the sub-layer input to its output.
2. **Layer Normalisation** — stabilises training by normalising the feature dimension.

This combination (`LayerNorm(x + SubLayer(x))`) enables very deep stacks (96+ layers in
GPT-3).

---

## 3. Decoder-Only Architectures

Modern LLMs (GPT-4, Claude, Llama, Mistral) use a **decoder-only** architecture — a
simplified Transformer that retains only the decoder stack from the original
encoder-decoder design.

Key properties:
- **Causal (autoregressive) attention**: each token can only attend to previous tokens.
  This is enforced by masking the upper triangle of the attention matrix.
- **Next-token prediction training**: the model is trained to predict the next token
  given all preceding tokens (language modelling objective).
- **No cross-attention**: there is no separate encoder; all context is handled via the
  masked self-attention.

This simplicity allows decoder-only models to scale extremely well and generalise to
tasks they were never explicitly trained on (in-context learning).

---

## 4. Claude as a Transformer Backbone

Anthropic's Claude models are decoder-only Transformers trained with a combination of:

1. **Constitutional AI (CAI)** — a feedback process that makes the model more helpful,
   harmless, and honest by having it critique its own outputs using a set of principles.
2. **RLHF (Reinforcement Learning from Human Feedback)** — human preference labels guide
   the model toward outputs that humans rate as higher quality.
3. **Massive pretraining** on diverse text corpora.

In the context of this research assistant, Claude serves as the cognitive backbone for:
- Decomposing research topics (orchestrator)
- Scoring source relevance (research agent)
- Classifying sources (classification agent)
- Augmenting NER (NER agent)
- Synthesising themes (analyzer agent)
- Writing reports (writer agent)
- Critiquing quality (critic agent)

The Transformer's ability to hold long contexts and reason over them makes it uniquely
suited for multi-step agentic workflows.

---

## 5. Why Transformers Are Ideal for Multi-Agent Research

| Property | Benefit for Research Agents |
|----------|-----------------------------|
| Long context window (100K+ tokens) | Agents can process many sources at once |
| In-context learning | Agents follow instructions without fine-tuning |
| Structured output (JSON) | Agents produce parseable outputs reliably |
| Tool use / function calling | Agents can invoke external APIs (search, scrape) |
| Few-shot prompting | Agents can be guided by examples in the prompt |

The self-attention mechanism is particularly valuable: when the writer agent summarises
10+ sources, self-attention allows it to track relationships between distant pieces of
information — something RNNs struggled with due to vanishing gradients.

---

## 6. Scaling Laws

Research by Kaplan et al. (2020) showed that Transformer performance follows predictable
**power-law scaling**:

```
Loss ∝ N^(-α)   where N = parameter count
```

This means that larger models are reliably better — but the training compute and
inference cost scale accordingly.  This drives the exploration of Mixture of Experts
(MoE) as a way to increase effective model size without proportionally increasing
inference cost (see `moe_comparison.md`).

---

## 7. Summary

The Transformer's combination of self-attention, residual connections, and positional
encoding creates a universally applicable sequence model.  Decoder-only variants like
Claude are the workhorse of modern AI systems, and their long-context, instruction-
following capabilities make them ideal orchestrators for complex multi-agent pipelines.
