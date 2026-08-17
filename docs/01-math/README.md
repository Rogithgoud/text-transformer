# 01 — Mathematics

One file per concept, created from [`../_templates/math-template.md`](../_templates/math-template.md).
Build order is top to bottom; do not skip ahead.

| # | File | Concept | Implemented | Hand-verified | Gradient-checked | Documented |
|---|---|---|---|---|---|---|
| 01 | `01-scalar-functions.md` | exp, log, sqrt, tanh (Taylor / Newton) | ☐ | ☐ | ☐ | ☐ |
| 02 | `02-vectors.md` | add, scale, elementwise multiply | ☐ | ☐ | ☐ | ☐ |
| 03 | `03-dot-product.md` | dot product as projection & similarity | ☐ | ☐ | ☐ | ☐ |
| 04 | `04-matmul.md` | matrix multiply, transpose, shape rules | ☐ | ☐ | ☐ | ☐ |
| 05 | `05-softmax.md` | softmax + max-subtraction stability | ☐ | ☐ | ☐ | ☐ |
| 06 | `06-layernorm.md` | mean, variance, normalise, scale & shift | ☐ | ☐ | ☐ | ☐ |
| 07 | `07-cross-entropy.md` | negative log-likelihood | ☐ | ☐ | ☐ | ☐ |
| 08 | `08-backprop.md` | chain rule on a computational graph | ☐ | ☐ | ☐ | ☐ |
| 09 | `09-gradient-checking.md` | finite differences, choice of `h` | ☐ | ☐ | — | ☐ |
| 10 | `10-initialisation.md` | why weights are scaled, not just random | ☐ | ☐ | — | ☐ |
| 11 | `11-sgd.md` | gradient descent, learning rate | ☐ | ☐ | — | ☐ |
| 12 | `12-adam.md` | moments, bias correction, epsilon | ☐ | ☐ | — | ☐ |

## The two derivations that matter most

Do these on paper, photograph them into [`../05-visuals/`](../05-visuals/), and transcribe the
steps into the relevant file:

1. **softmax + cross-entropy gradient** → collapses to `predicted − actual`. The cleanest
   result in the whole project.
2. **matmul backward** → `dA = dY · Bᵀ` and `dB = Aᵀ · dY`. Derive it by writing one output
   element as an explicit sum and differentiating that sum.

## Rule

No formula gets used in `src/` before its file here exists and its hand-worked example matches
the code.
