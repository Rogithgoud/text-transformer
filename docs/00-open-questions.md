# Open Questions

Every time something is unclear it gets a row here straight away, even if it feels like a stupid
question, even if I expect to answer it in ten minutes. This file is supposed to grow through the
project. An answered question keeps its row, with the answer and the date.

Anything still unanswered at the end stays here marked `still unknown`. That's an honest result,
not a failure.

---

## Open

| # | Question | Raised on | Phase | Why it matters |
|---|---|---|---|---|
| Q1 | In training the model always sees real text, but in generation it sees its own output. Why does nobody fix this mismatch? Is there a standard fix and is it just too expensive? | 2026-08-17 | 0 | it's probably a big reason my output will degrade partway through, so I should know if it's my bug or a known property |
| Q2 | How do I pick `block_size`? I chose 32 for speed, but what actually breaks if it's too small vs too large, apart from time per step? | 2026-08-17 | 0 | it's a hyperparameter I'm choosing blind right now |
| Q3 | Why sample from the probability distribution at all instead of always taking the highest-probability character? Greedy sounds better on paper | 2026-08-17 | 0 | decides what I implement in Phase 8 and how I explain temperature |
| Q4 | Softmax exaggerates the gaps between scores because of `exp`. Early in training the weights are random, so does that make the model wrongly confident, and does that hurt the first few steps? | 2026-08-17 | 0 | might explain the shape of the start of my loss curve |
| Q5 | Should newline, space and punctuation be in the vocabulary as normal characters, or treated specially? | 2026-08-17 | 0 | changes vocab size, which changes the expected starting loss `ln(V)` that I use as a sanity check |
| Q6 | The paper is encoder-decoder but I'm building decoder-only. What exactly am I dropping, and does anything in the paper stop making sense once the encoder is gone? | 2026-08-17 | 0 | I need to know what to skip when I read the paper instead of getting stuck on it |

## Answered

| # | Question | Answer (short) | Raised | Answered | Where written up |
|---|---|---|---|---|---|
| A1 | Does one training sequence give one prediction problem or many? | Many. One pass gives logits at every position, so a length-32 window gives 32 problems at once, and the causal mask is what stops them cheating off each other | 2026-08-17 | 2026-08-17 | §1, "what training data looks like" |
| A2 | Where does the causal mask actually come from? Is it just a trick people use? | No, it falls straight out of the definition `P(next \| previous)`. Drawing the "what can each position see" table gives you the triangle directly | 2026-08-17 | 2026-08-17 | §1, step D |

## Still unknown at project end

| # | Question | What I tried | Best current guess |
|---|---|---|---|
| — |  |  |  |
