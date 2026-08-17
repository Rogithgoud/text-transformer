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

## YYYY-MM-DD — Session 3:

**Did:**

**Learned:**

**Stuck:**

**Next:**
