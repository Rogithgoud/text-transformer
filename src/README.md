# src — build order

**No code yet, by design.** Nothing here gets written before its documentation file exists and
its hand-worked example matches. See [../docs/ROADMAP.md](../docs/ROADMAP.md).

Planned modules, in the order they must be built. Each one is only started once the previous one
is gradient-checked and documented.

| Order | File | Contents | Depends on | Phase |
|---|---|---|---|---|
| 1 | `mathfuncs.py` | exp, log, sqrt, tanh — from series / Newton's method | — | 2 |
| 2 | `vector.py` | add, scale, dot, elementwise multiply | 1 | 2 |
| 3 | `matrix.py` | matmul, transpose, shape helpers | 2 | 2 |
| 4 | `autograd.py` | the graph, topological sort, backward propagation | 3 | 3 |
| 5 | `ops.py` | forward + backward for add, mul, matmul, softmax, layernorm, activation | 4 | 3 |
| 6 | `gradcheck.py` | finite-difference gradient checking | 5 | 3 |
| 7 | `tokenizer.py` | character-level vocabulary, encode, decode | — | 4 |
| 8 | `dataset.py` | random windows, inputs and shifted targets | 7 | 4 |
| 9 | `layers.py` | embedding, positional encoding, linear, layernorm | 5 | 5 |
| 10 | `attention.py` | single-head, causal mask, multi-head | 9 | 5 |
| 11 | `block.py` | pre-norm block: attention + feed-forward + residuals | 10 | 5 |
| 12 | `model.py` | the full decoder-only Transformer, parameter collection | 11 | 6 |
| 13 | `loss.py` | cross-entropy, forward and backward | 5 | 7 |
| 14 | `optim.py` | SGD, then Adam | 12 | 7 |
| 15 | `train.py` | the training loop, logging, checkpointing to JSON | 13, 14 | 7 |
| 16 | `generate.py` | sampling: greedy, temperature, top-k | 12 | 8 |
| 17 | `visualise.py` | ASCII heatmaps, loss curve, embedding neighbours | 12 | 9 |

## Conventions to fix now, before any code exists

Decide these once and write them down, so shape bugs stay findable:

- **Tensor representation:** nested Python lists. Document the exact nesting for each rank.
- **Shape convention:** `(batch, time, channels)` — i.e. `(B, T, C)`. Never deviate silently.
- **Beware `[[0] * n] * m`** — it creates `m` references to *one* row. Use a comprehension.
- **Every function documents its input and output shapes in its docstring.** No exceptions.
- **Determinism:** one seeded `random.Random` instance, seed recorded in every experiment file.
- **No hidden mutation:** operations return new structures rather than editing inputs in place,
  except where documented for performance (and then, say so).
