# Research Notes

Everything here in **my own words**. No copy-paste from papers or blogs. If I cannot restate it
myself, it does not belong here yet — it belongs in [00-open-questions.md](00-open-questions.md).

---

## Reading list and status

| # | Source | Status | Notes file / section below |
|---|---|---|---|
| 1 | Language modelling as next-token prediction | not read | §1 |
| 2 | *Attention Is All You Need* (Vaswani et al., 2017) — pass 1 (shape) | not read | §2 |
| 2 | pass 2 (equations) | not read | §2 |
| 2 | pass 3 (details I skipped) | not read | §2 |
| 3 | Karpathy — "Let's build GPT" / nanoGPT / micrograd (concept only) | not read | §3 |
| 4 | Alammar — *The Illustrated Transformer* | not read | §4 |
| 5 | Backprop as reverse-mode autodiff | not read | §5 |

---

## §1 — What a language model actually is

- The task, stated as probability:
- Why "next token" is enough to produce whole paragraphs:
- What the model outputs at each position, and its shape:
- What training data looks like (input vs target):

## §2 — Attention Is All You Need

- The one-sentence idea of the paper:
- Encoder–decoder in the paper vs **decoder-only** for text generation (what I am building, and
  what I am dropping):
- The attention equation, restated in my own words:
- Why scaling by √d_k:
- Why multiple heads:
- Why the position-wise feed-forward network:
- What the paper does *not* explain that I had to find elsewhere:

## §3 — Concept notes from implementations (no code copied)

- The overall training loop shape:
- Character-level vs subword tokenisation:
- Things I noticed frameworks hide from you:

## §4 — Visual intuitions

- Embedding space, pictured:
- Attention as a lookup / soft dictionary:
- The N×N attention grid and what the causal mask does to it:
- Residual stream as a "highway" that each block reads from and writes back to:

## §5 — Backpropagation

- The chain rule as a **local** rule:
- What a computational graph is, and why a topological ordering is required:
- Forward pass vs backward pass — what is stored, and why memory grows with depth:
- What `zero_grad` corresponds to, and why gradients accumulate by default:

---

## Vocabulary I had to learn

| Term | My definition | First time I met it |
|---|---|---|
| logit |  |  |
| token |  |  |
| embedding |  |  |
| d_model |  |  |
| head |  |  |
| context / block size |  |  |
| residual stream |  |  |
| temperature |  |  |

---

## Phase 0 understanding gate

Answer without notes, then move the answers into [VIVA.md](VIVA.md):

1. Why does text generation need a causal mask?
2. Why divide attention scores by √d_k, and what goes wrong numerically without it?
3. Why is there a residual connection around every sublayer?
4. Why do Transformers need positional information when RNNs do not?
