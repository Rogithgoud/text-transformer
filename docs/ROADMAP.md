# Roadmap

The build order. Each phase ends with an **understanding gate** — a set of questions to answer
without notes before moving on. Answer them in [VIVA.md](VIVA.md).

Rule for the whole project: **nothing gets built before it is understood, and nothing is
considered done before it is documented and tested.**

---

## Phase 0 — Research

Read, in order:

1. Language modelling as next-token prediction: `P(next token | all previous tokens)`.
2. *Attention Is All You Need* (Vaswani et al., 2017) — three passes: shape, equations, details.
3. Karpathy's "Let's build GPT" / nanoGPT / micrograd write-ups — **conceptual reference only,
   do not copy code.**
4. Alammar's *The Illustrated Transformer* — visual intuition.
5. Backpropagation as reverse-mode automatic differentiation — the chain rule on a graph.

**Deliverables:** `00-research-notes.md`, `00-open-questions.md`, hand-drawn architecture
diagram with tensor shapes on every arrow.

**Gate:** why a causal mask? why divide by √d_k? why residual connections? why positional
information at all?

---

## Phase 1 — Repo + documentation skeleton

Set up the documentation machinery *before* coding, so documenting is a habit and not a
cleanup job. Templates live in [_templates/](_templates/).

**Commit discipline:** one commit per understood unit. The message says what was learned, not
just what was added. Push daily.

**Error-log rule:** the moment a bug appears — *before fixing it* — create
`03-errors/NNN-short-name.md` from the template.

---

## Phase 2 — Math primitives

Bottom-up. Implement → hand-verify → document → commit.

1. Scalar helpers: `exp`, `log`, `sqrt`, `tanh`. *Optional, high value:* implement them
   yourself (Taylor series, Newton's method) before falling back to `math`.
2. Vector ops: add, scale, dot product, elementwise multiply. Document the dot product
   **geometrically** — projection and similarity. This is the core intuition for attention.
3. Matrix ops: matmul (triple loop), transpose, the broadcasting rules you rely on.
4. Softmax — with the max-subtraction stability trick. Expect `OverflowError` without it; log it.
5. Layer normalisation — mean, variance, normalise, learnable scale and shift.
6. Cross-entropy loss — as negative log-likelihood. **Derive the softmax+cross-entropy
   gradient by hand** and see it collapse to `predicted − actual`.

**Gate:** on paper, with no computer: dot product of two 3-vectors, softmax over 3 scores,
cross-entropy for a known target.

---

## Phase 3 — Autodiff engine (the crux)

There is no `.backward()`. Two routes — pick one and document why.

- **Route A (recommended):** a mini autograd engine. Every op records its inputs and a local
  gradient function, forming a graph; topologically sort and propagate gradients backwards.
- **Route B:** hand-written manual backward passes per layer. Simpler to grasp, brutal to debug.

Steps: chain rule as a *local* rule → scalar graph first → extend to matrices → derive and
document backward passes for add, multiply, matmul, softmax, layernorm, activation.

**Gradient checking is non-negotiable:** compare each analytic gradient against the numerical
estimate `(f(x+h) − f(x−h)) / 2h`. Must agree to ~1e-5. Document the chosen `h` and why a
too-small `h` fails (floating-point cancellation).

**Deliverable:** `01-math/backprop.md` with a computational graph drawn for one tiny example,
forward values and backward gradients annotated on every edge.

---

## Phase 4 — Data + tokenizer

1. Small public-domain plain-text corpus; record source and licence in `data/README.md`.
2. Character-level tokenizer: vocabulary, char↔id maps, encode/decode. Document why
   character-level fits this project, and what BPE would buy later.
3. Batching: random windows of length `block_size`; targets are inputs shifted left by one.
   Draw the shift on paper — this is where off-by-one bugs live.

**Gate:** why are targets the inputs shifted by one, and how does one length-64 sequence give
64 independent prediction problems at once?

---

## Phase 5 — Transformer components, one at a time

Build, test, and document each in isolation. Do **not** assemble the whole model and then debug.

1. Token embedding — a learnable lookup table.
2. Positional encoding — sinusoidal or learned; why alternating frequencies.
3. Single-head self-attention — Q/K/V projections → `Q·Kᵀ` → scale by √d_k → causal mask →
   softmax → weighted sum of V. Document all six steps separately, with shapes.
4. Causal mask — set future positions to −∞ *before* the softmax. Why −∞, and why masking
   after the softmax is wrong.
5. Multi-head attention — split `d_model` into heads, run in parallel, concatenate, project.
6. Feed-forward network — two linears with a nonlinearity, inner dim ≈ 4×. Prove algebraically
   that two stacked linears without a nonlinearity collapse into one.
7. Residual connections — the gradient highway.
8. Pre-norm vs post-norm — choose pre-norm; document the stability reason.
9. The full block.
10. Output head — project to vocab size for logits.

**Tests per component:** shapes correct; attention rows sum to 1; the mask genuinely blocks the
future (change a later token, confirm earlier outputs are unchanged — this single test catches
the most common bug in the project).

---

## Phase 6 — Assembly and forward pass

1. Stack: embedding → positional → N blocks → final norm → output head.
2. Sanity check: initial loss on random weights should be ≈ `ln(vocab_size)` — a uniformly
   clueless model. If not, something is broken. Cheapest bug-catcher available.
3. Count parameters by hand from the formulas, then verify against the code.

---

## Phase 7 — Training

1. Loss: cross-entropy over all positions.
2. Backward: gradient-check the *whole model* on a tiny config, not just single ops.
3. Optimizer: plain SGD first, understand it, then Adam — document momentum, second moment,
   bias correction (why it exists), and epsilon.
4. Training loop: batch → forward → loss → zero grads → backward → step → log. Document why
   gradients must be zeroed.
5. **Overfit one batch first.** Train on a single tiny batch until loss → ~0. If it cannot,
   the backward pass is wrong. This is *the* diagnostic.
6. Real training run; log losses to a text file; render an ASCII/SVG loss curve (no matplotlib);
   checkpoint weights as JSON.
7. Document every run in `04-experiments/`.

Expect minutes-to-hours where a framework takes seconds. Measure it, and explain why
(interpreted loops vs vectorised C/BLAS).

---

## Phase 8 — Generation

1. Prompt → logits at last position → probabilities → sample → append → repeat.
2. Implement and document greedy (argmax), temperature scaling (what T<1 and T>1 do to the
   distribution's shape — draw both), and top-k sampling.
3. Document the failure modes honestly: a tiny model on a small corpus yields near-gibberish
   with occasional real words and plausible spacing. Explain exactly why.

---

## Phase 9 — Visual understanding artifacts

- Attention heatmaps per head (ASCII or hand-built SVG/HTML) **plus a written interpretation**
  of what each head appears to have learned.
- Embedding nearest neighbours by cosine similarity, before vs after training.
- Annotated loss curve.
- One-page shape-flow diagram: input ids → logits, every shape labelled.
- Hand-annotated computational graph for one tiny forward + backward.

---

## Phase 10 — Final documentation

README (what, why, how, results, honest limitations, reading order), `LEARNING-LOG.md`,
a populated `03-errors/` (expect 10–20 real entries), `00-open-questions.md` with every
question answered and dated, `VIVA.md` completed, clean commit history.

---

## Guardrails

- **Sequence matters more than speed.** Do not build attention before the autodiff engine is
  gradient-checked.
- Rough timeline: Phase 0–1 ≈ 2–3 days · Phase 2–3 ≈ a week or more (autodiff is genuinely
  hard) · Phase 4–8 ≈ a week · Phase 9–10 ≈ 2–3 days.
- **No scope creep.** Extras belong in the README's "Future work" section.
