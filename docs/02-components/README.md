# 02 — Components

One file per Transformer part, created from
[`../_templates/component-template.md`](../_templates/component-template.md).
Each component is built, tested, and documented **in isolation** before the next one starts.

| # | File | Component | Built | Isolated test | Invariant test | Documented |
|---|---|---|---|---|---|---|
| 01 | `01-tokenizer.md` | character-level tokenizer | ☐ | ☐ | round-trip encode/decode ☐ | ☐ |
| 02 | `02-batching.md` | random windows, targets shifted by one | ☐ | ☐ | off-by-one check ☐ | ☐ |
| 03 | `03-token-embedding.md` | learnable lookup table | ☐ | ☐ | shape ☐ | ☐ |
| 04 | `04-positional-encoding.md` | sinusoidal / learned positions | ☐ | ☐ | uniqueness per position ☐ | ☐ |
| 05 | `05-linear-layer.md` | weight · x + bias | ☐ | ☐ | gradient check ☐ | ☐ |
| 06 | `06-single-head-attention.md` | Q·Kᵀ → scale → mask → softmax → ·V | ☐ | ☐ | rows sum to 1 ☐ | ☐ |
| 07 | `07-causal-mask.md` | −∞ on future positions, before softmax | ☐ | ☐ | future cannot leak ☐ | ☐ |
| 08 | `08-multi-head-attention.md` | split → attend → concat → project | ☐ | ☐ | equals single-head when n=1 ☐ | ☐ |
| 09 | `09-feed-forward.md` | linear → nonlinearity → linear | ☐ | ☐ | gradient check ☐ | ☐ |
| 10 | `10-residual-and-norm.md` | pre-norm block wiring | ☐ | ☐ | identity at zero-init ☐ | ☐ |
| 11 | `11-transformer-block.md` | the repeatable block | ☐ | ☐ | shape in = shape out ☐ | ☐ |
| 12 | `12-output-head.md` | project to vocab logits | ☐ | ☐ | initial loss ≈ ln(vocab) ☐ | ☐ |
| 13 | `13-full-model.md` | assembled model | ☐ | ☐ | overfit one batch ☐ | ☐ |
| 14 | `14-sampling.md` | greedy, temperature, top-k | ☐ | ☐ | probabilities sum to 1 ☐ | ☐ |

## The single most valuable test in this project

**"The mask genuinely blocks the future."** Take an input sequence, record the output at
position *i*. Change any token at a position *later* than *i*. The output at position *i* must be
bit-for-bit identical. If it changed, information is leaking backwards through time and every
subsequent result is meaningless.

Write this test early. It catches the most common bug in a hand-built Transformer.
