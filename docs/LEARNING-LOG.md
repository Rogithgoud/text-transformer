# Learning Log

Chronological journal, one entry per working session, newest at the bottom. Written the same day,
not reconstructed later.

Each entry answers four things:

- **Did** — what I built or read
- **Learned** — the concept that landed, in my own words
- **Stuck** — what confused me, and whether it's resolved
- **Next** — the single next step

---

## 2026-08-17 — Session 1: project setup

**Did:** set up the repo, the folder structure and the doc templates (component, math, error,
experiment). Wrote the roadmap for all 10 phases with an understanding gate at the end of each.
Listed the 12 math concepts, 14 components, 11 tests and 11 visual artifacts I'll need, and fixed
the shape convention `(B, T, C)` before writing any code so I don't create shape bugs for myself
later. No model code on purpose.

**Learned:** setting the documentation up first is the only way this stays documented. If I leave
it to the end I'll be reconstructing from memory and it'll show. Also pre-registered 10 bugs I
expect to hit (softmax overflow, off-by-one targets, mask after softmax, gradients not zeroed and
so on) so that when they happen I recognise them instead of panicking.

**Stuck:** spent a while deciding how strict "no libraries" should be. Settled on standard
library only and wrote the rules at the top of the research notes so it's not ambiguous later.

**Next:** Phase 0 reading item 1, next-token prediction.

---

## 2026-08-17 — Session 2: what a language model actually is (Phase 0, item 1)

**Did:** worked through reading item 1 and filled §1 of the research notes. Four things: the task
as a probability statement, why one-character prediction produces paragraphs, what comes out at
each position, and what a training example looks like. Did the "hello" input/target table by hand
with `block_size = 4`, then redid the whole thing for "banana" with `block_size = 3` from memory
to check I actually had it.

**Learned:** three things really landed.

1. The whole task is `P(next char | all previous chars)`, and a whole sentence's probability is
   just that one question multiplied down the chain. So one small question covers all of language.
2. The target is literally the input shifted left by one, and that single pair secretly contains
   `block_size` separate prediction problems. Writing out the "what can each position see" column
   gave me a triangle, and that triangle is the causal mask. I derived the mask from the
   definition of the task before reading the paper, which I did not expect.
3. Logits are not probabilities. They're raw unbounded scores, `V` of them per position, and
   softmax does two jobs: `exp` to force them positive and exaggerate the gaps, then divide by the
   sum to force the total to 1.

**Stuck:** two things I noted rather than solved. Why nobody fixes the mismatch between training
(model sees real text) and generation (model sees its own output), and whether softmax's gap
exaggeration makes an untrained model overconfident at the start. Both are in
[00-open-questions.md](00-open-questions.md) as Q1 and Q4.

Also spotted the off-by-one trap before writing any code. With 5 characters and `block_size = 4`
the only valid start index is 0, because the target window needs one extra character past the
input window. So the random start has to come from `0 .. len(data) - block_size - 1`. That's the
kind of bug that doesn't crash, it just silently trains on wrong data.

**Next:** Phase 0 reading item 2, Attention Is All You Need, pass 1, reading for shape only, not
equations. Question Q6 first: work out what I'm dropping by building decoder-only.

---

## 2026-08-18 — Session 3: built the whole thing and trained it

**Did:** wrote all 11 modules and 5 test files, then trained. matrix, tokenizer, dataset, layers,
attention, block, model, loss, optim, train, generate, visualise. Ran the gradient check, the mask
test and the overfit test before the real run, then 2000 steps on Tiny Shakespeare in 70 minutes,
final val loss 2.19 from a start of 5.02.

Two decisions I made to get there tonight instead of next week:

1. **No general autograd engine.** Each layer has its own hand-written `backward()` that adds to
   its parameter gradients and returns the gradient of its input. Same chain rule, applied
   locally, no graph to build or topologically sort. The roadmap had the engine as Route A and
   this as Route B, and Route B was the right call under a deadline.
2. **Batches are a loop, not a third dimension.** Every matrix stays 2D `(T, C)`. In pure Python
   the whole thing is loop-bound anyway, so a batch dimension would have cost complexity and
   bought nothing. It made every shape easy to hold in my head while debugging.

**Learned:** the backward pass is not mysterious once you write it. Reading my `attention.py`
forward and backward side by side, the backward is just the forward reversed, line for line:
`out = probs @ V` becomes `dprobs = dout @ V^T` and `dV = probs^T @ dout`. The shapes only fit
one way round, which is a useful check when I'm not sure I've got a transpose right.

Also: the tests earned their keep immediately.

- The **gradient check** is the only reason I trust any of this. 3.7e-08 worst relative error over
  128 cells means matmul, softmax, layernorm, attention, the residuals and the embedding backward
  are all correct. Without it I would have been guessing.
- The **overfit-one-batch** test (2.51 → 0.00096) proved forward, backward and Adam work together
  before I spent an hour on a real run.
- The **mask test** is the one that protects the result from being fake. If the future leaked into
  the past, the loss would look *better*, not worse, and I would have believed it.

Then the visual artifacts turned out to be the best part. The two attention heads in block 0 and
block 1 are doing visibly different jobs, local vs structural. And the embedding table grouped
`.` `?` `!` at 0.96 cosine similarity, and paired `t`/`T`, `a`/`A`, `q`/`Q` — categories nothing
in my code knows about. It got those purely from which characters are useful in the same places.

**Stuck:** two things, both written up rather than hand-waved.

- The gradient check failed on the first run at 3.3e-03 and I assumed layernorm was wrong. It
  wasn't. The test was dividing float noise by float noise on weights whose true gradient is zero.
  Written up in [03-errors/001](03-errors/001-gradcheck-noise-on-zero-gradients.md). 20 minutes
  lost to suspecting the wrong thing.
- I predicted the initial loss would be `ln(65) = 4.17` and got 5.02. The output head is
  initialised at `1/sqrt(32)`, so the first logits already have spread and the model starts
  confidently wrong, which costs more than being uniformly clueless. The sharper rule is that
  `ln(V)` is the loss when the logits are all *equal*. That's experiment 002's one change.

Also caught myself writing 41,761 as the parameter count from memory when the real number is
29,697. Fixed it in the file and left a note. Numbers go in these docs from the output.

**Next:** Phase 0 item 2, actually read the paper now that I've built the thing it describes —
I expect it to read completely differently from this side. Then experiment 002: head init 0.02,
one change only.
