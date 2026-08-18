# text-transformer

**A text-generation Transformer built from scratch in pure Python — no libraries.**

A character-level text-generation Transformer (decoder-only, GPT-style) implemented in
**pure Python with only the standard library**. No PyTorch, no TensorFlow, no JAX, no NumPy,
no SciPy. Every matrix multiply, every softmax, every gradient, and the optimizer are
written by hand from the mathematics.

This repository is as much a **learning document** as it is code. The goal is not a good
language model — it is complete, provable understanding of every operation inside one.

---

## Status

**It trains and it generates.** First real run: loss 5.02 → 2.19 val in 70 minutes on one CPU
core, 29,697 parameters. Details in
[docs/04-experiments/001-first-real-run.md](docs/04-experiments/001-first-real-run.md).

- [ ] Phase 0 — Research *(next-token prediction done, the paper itself still to read properly)*
- [x] Phase 1 — Repo + doc skeleton
- [x] Phase 2 — Math primitives *(matmul, softmax, layernorm, cross-entropy; my own `exp`/`sqrt`
      still an optional extra)*
- [x] Phase 3 — Backward passes + gradient checking *(hand-written per layer instead of a general
      autograd engine, see below)*
- [x] Phase 4 — Data + tokenizer
- [x] Phase 5 — Transformer components
- [x] Phase 6 — Model assembly / forward pass
- [x] Phase 7 — Training (loss, backward, Adam)
- [x] Phase 8 — Generation / sampling
- [x] Phase 9 — Visual artifacts *(loss curve, attention heatmaps, embedding neighbours done;
      positional and temperature plots still to do)*
- [ ] Phase 10 — Final documentation

**One deliberate change from the original plan:** the roadmap had a general autograd engine
(record a graph, topologically sort it, propagate backwards). I built Route B instead — every
layer has its own hand-written `backward()` that adds to its parameter gradients and returns the
gradient of its input. Same chain rule, no graph machinery, and it got to a working training run
much sooner. The trade-off: adding a new layer means deriving its backward by hand, and there is
no `.backward()` to lean on.

---

## The constraint, and why

> No imports that do the numerical work for you.

Allowed: `math`, `random`, `json`, `time`, `os`, `sys`, `pickle` (standard library only).
Not allowed: any array/tensor/autodiff/ML library.

**No GPU, and that is forced, not chosen.** A GPU is only reachable through CUDA, and CUDA is
only reachable through the libraries this project bans. So this is a single-CPU-core project by
definition — the constraint decided the hardware.

**Consequence, stated honestly:** pure interpreted Python is thousands of times slower than
a vectorised BLAS backend. So the model is deliberately tiny — character-level vocabulary,
~2 layers, small `d_model`, short context, ~100 KB of training text. A tiny model that is
fully understood is the deliverable. Scale is not.

The full rules I fixed for this project are at the top of
[docs/00-research-notes.md](docs/00-research-notes.md).

See [docs/01-math/README.md](docs/01-math/README.md) for the hand-written maths, and
[docs/03-errors/](docs/03-errors/) for every bug hit along the way.

---

## Reading order (for a reviewer)

Read the repository in this order to follow the actual learning path:

1. [docs/ROADMAP.md](docs/ROADMAP.md) — the plan, phase by phase
2. [docs/00-research-notes.md](docs/00-research-notes.md) — concepts in my own words
3. [docs/01-math/](docs/01-math/) — every formula, derived and hand-verified
4. [docs/02-components/](docs/02-components/) — each Transformer part in isolation
5. [docs/05-visuals/](docs/05-visuals/) — attention heatmaps, shape-flow diagram, loss curves
6. [docs/04-experiments/](docs/04-experiments/) — every training run and its result
7. [docs/03-errors/](docs/03-errors/) — the bug log: symptom, cause, maths reason, fix
8. [docs/LEARNING-LOG.md](docs/LEARNING-LOG.md) — chronological journal
9. [docs/00-open-questions.md](docs/00-open-questions.md) — what confused me, and the answers
10. [docs/VIVA.md](docs/VIVA.md) — self-examination: the questions I must answer cold

---

## Architecture

Shapes for the run in experiment 001. `T = 32` positions, `C = 32` channels, `V = 65` characters.

```
 ids                     (32,)        32 character ids
   |  embedding lookup                65 x 32 learnable table
 vectors                 (32, 32)     each character as 32 numbers
   |  + positional encoding           fixed sin/cos, no parameters
 vectors                 (32, 32)     now they also know where they are
   |  block 1                         2 heads
   |     |- layernorm -> causal self-attention -> + residual
   |     |- layernorm -> feed-forward 32->128->32 -> + residual
 vectors                 (32, 32)
   |  block 2                         same again
 vectors                 (32, 32)
   |  final layernorm
 vectors                 (32, 32)
   |  output head                     32 x 65
 logits                  (32, 65)     65 scores at every position
   |  softmax
 probabilities           (32, 65)     ready to sample from
```

The width never changes from the embedding to the head. That fixed-width channel is the residual
stream: every block reads from it and adds back into it.

Inside one attention head: `Q·Kᵀ` → `÷√head_size` → mask the future with `-inf` → softmax each row
→ weighted sum of `V`. The mask goes on **before** the softmax, so `exp(-inf) = 0` and the rows
still sum to 1.

## How to run

No dependencies. Python 3 standard library only.

```bash
python tests/test_gradcheck.py
```

```bash
python src/train.py overfit
```

```bash
python src/train.py train
```

```bash
python src/generate.py 400 0.8 10 "KING RICHARD:"
```

```bash
python src/visualise.py
```

Run the tests and then `overfit` before any real run. `overfit` memorises a single batch — if the
loss does not go to nearly zero, the backward pass is wrong and a long run would be wasted.
`train` needs `data/input.txt`, see [data/README.md](data/README.md) for where to get it.

## Results

Experiment 001, full write-up in
[docs/04-experiments/001-first-real-run.md](docs/04-experiments/001-first-real-run.md):

| | |
|---|---|
| parameters | 29,697 |
| corpus | Tiny Shakespeare, 1,115,394 chars, vocab 65 |
| config | `block_size` 32, `d_model` 32, 2 layers, 2 heads, batch 4, Adam 3e-3 |
| initial loss | 5.0217 |
| final loss | **2.2595 train, 2.1890 val** |
| time | 70.3 min, 1.93 s/step, one CPU core |

Correctness checks, which matter more here than the loss:

| Check | Result |
|---|---|
| gradient check vs finite differences | worst relative error **3.7e-08** over 128 cells |
| future cannot affect the past | passes, and each position does affect its own output |
| attention rows sum to 1, upper triangle exactly 0 | passes |
| overfit a single batch | 2.51 → **0.00096** in 200 steps |
| initial loss vs `ln(65) = 4.1744` | 4.35 with a small init, 5.02 with the head at `1/√d` |

Sample, temperature 0.8, top-k 10:

```
KING RICHARD:

Wit haw brest how his tast tour amar tworne,
And shive by tore shives, coue hanceld
Bun he she sond this him hall torre,
Shall ling blave lery thar thee with tas hat, lande wather all cand hastess ting
Heell shallost the st she my the scold.

COUCICESTUS:
Showes lond lichemees lich somerave
```

**What it learned:** the speech structure (a capitalised name, a colon, a newline, then a capital
letter — `COUCICESTUS:` is invented but exactly the right shape), real short words, sensible word
lengths and spacing, commas mid-line and full stops at line ends.

**What it did not learn, and why:** no grammar and no long-word spelling. The context is 32
characters, about six words, so it cannot see a sentence. It ran for 2000 steps at batch 4, so it
saw roughly 256k characters — about a quarter of the corpus, once. 29,697 parameters. The output is
what that budget buys, and validation loss stayed at or below train loss throughout, so it is
underfitting: the limit is model size and compute, not data.

Two findings from the visual artifacts, written up in
[docs/05-visuals/README.md](docs/05-visuals/README.md):

- **The heads divide the work.** Block 0 head 0 is a local previous-character head, nearly all of
  its weight on the diagonal and the cell to its left. Block 1 head 1 instead attends back to the
  start of the line — which is what produces the `NAME:` structure. Local first, then structural,
  which is why 2 layers beat 1.
- **The embedding table learned categories nobody programmed.** Cosine similarity puts `.` `?` `!`
  at 0.96/0.92 — nearly the same vector, because for predicting what comes next they are
  interchangeable. Upper and lower case pairs found each other too: `t`/`T` 0.66, `a`/`A` 0.55,
  `q`/`Q` 0.55.

## Future work

Deliberately out of scope for this project, listed so the boundary is explicit:
dropout, learning-rate schedules, BPE/subword tokenisation, weight tying, KV caching,
mixed precision, any form of parallelism.

## License

MIT — see [LICENSE](LICENSE).
