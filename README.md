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

`Phase 0 — Research`  *(update this line as you progress; see [docs/ROADMAP.md](docs/ROADMAP.md))*

- [ ] Phase 0 — Research
- [ ] Phase 1 — Repo + doc skeleton
- [ ] Phase 2 — Math primitives
- [ ] Phase 3 — Autodiff engine
- [ ] Phase 4 — Data + tokenizer
- [ ] Phase 5 — Transformer components
- [ ] Phase 6 — Model assembly / forward pass
- [ ] Phase 7 — Training (loss, backward, optimizer)
- [ ] Phase 8 — Generation / sampling
- [ ] Phase 9 — Visual artifacts
- [ ] Phase 10 — Final documentation

---

## The constraint, and why

> No imports that do the numerical work for you.

Allowed: `math`, `random`, `json`, `time`, `os`, `sys`, `pickle` (standard library only).
Not allowed: any array/tensor/autodiff/ML library.

**Consequence, stated honestly:** pure interpreted Python is thousands of times slower than
a vectorised BLAS backend. So the model is deliberately tiny — character-level vocabulary,
~2 layers, small `d_model`, short context, ~100 KB of training text. A tiny model that is
fully understood is the deliverable. Scale is not.

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

*(Replace this block with your own diagram once drawn — every arrow labelled with tensor shapes.
See [docs/05-visuals/](docs/05-visuals/).)*

```
token ids  ->  token embedding  ->  + positional encoding
           ->  [ Transformer block ] x N
                   |- layer norm -> multi-head causal self-attention -> + residual
                   |- layer norm -> feed-forward network             -> + residual
           ->  final layer norm  ->  output projection to vocab  ->  logits
           ->  softmax  ->  probability of the next character
```

## How to run

*(Fill in once `src/` exists.)*

```bash
python src/train.py
```

Requirements: **none.** Python 3.10+ standard library only.

## Results

*(Fill in after Phase 8: final loss, sample generations, wall-clock time, and an honest
statement of limitations — what the model cannot do, and exactly why.)*

## Future work

Deliberately out of scope for this project, listed so the boundary is explicit:
dropout, learning-rate schedules, BPE/subword tokenisation, weight tying, KV caching,
mixed precision, any form of parallelism.

## License

MIT — see [LICENSE](LICENSE).
