# tests

Plain Python assertion scripts — no pytest, no test framework (same spirit as the rest of the
project). Each file is runnable on its own and prints what it checked.

Every component gets **at least** these three:

1. **Shape test** — output dimensions are exactly what the docs claim.
2. **Hand-calculation test** — a tiny input whose correct output was computed on paper.
3. **Gradient check** — analytic vs `(f(x+h) − f(x−h)) / 2h`, agreeing to ~1e-5.

Plus the component-specific invariants:

| Test file | Invariant it protects |
|---|---|
| `test_softmax.py` | outputs sum to 1; no overflow on large inputs; shift-invariance |
| `test_matmul.py` | matches hand calculation; `(AB)ᵀ = BᵀAᵀ` |
| `test_layernorm.py` | output mean ≈ 0, variance ≈ 1 along the normalised axis |
| `test_crossentropy.py` | loss on a uniform distribution ≈ `ln(vocab_size)` |
| `test_autograd.py` | gradients on a graph with a reused node accumulate correctly |
| `test_tokenizer.py` | `decode(encode(s)) == s` for the whole corpus |
| `test_dataset.py` | target[i] is exactly input[i+1]; no window runs off the end |
| `test_causal_mask.py` | **changing a later token cannot change an earlier output** |
| `test_attention.py` | attention rows sum to 1; multi-head with n=1 equals single-head |
| `test_model.py` | initial loss ≈ `ln(vocab_size)`; shape in = shape out per block |
| `test_overfit.py` | loss on one fixed tiny batch drives to ~0 |

## Why `test_causal_mask.py` matters most

It is the only test that catches information leaking backwards through time. That bug does not
crash and does not look wrong — it makes the loss suspiciously *good*, because the model is
reading the answer. Write this test early and run it after every change to attention.

## Running

```bash
python tests/test_softmax.py
```

Record any failure in [../docs/03-errors/](../docs/03-errors/) before fixing it.
