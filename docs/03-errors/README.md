# 03 — Error Log

Every bug gets its own file, numbered in the order encountered, from
[`../_templates/error-template.md`](../_templates/error-template.md).

**The rule:** open the file and fill in *symptom* and *what I first thought was wrong*
**before** attempting the fix. A log written after the fix loses the part that matters — the
reasoning, including the wrong reasoning.

Expect 10–20 real entries by the end of the project. Two entries means the logging was skipped,
not that the code was clean.

## Index

| # | File | Symptom | Root cause | Phase | Time lost |
|---|---|---|---|---|---|
| 001 |  |  |  |  |  |

## Bugs to expect (pre-registered predictions)

Written in advance, from the roadmap. When one of these actually happens, it still gets a full
file — and if a prediction never comes true, that is worth noting too.

| Predicted bug | Expected symptom | Expected cause |
|---|---|---|
| softmax overflow | `OverflowError: math range error` | `exp` of a large score without subtracting the max first |
| off-by-one in targets | loss falls suspiciously fast, or never below a floor | target window misaligned with the input window |
| mask applied after softmax | rows no longer sum to 1; the model appears to cheat | zeroing after normalising instead of −∞ before it |
| wrong transpose in `Q·Kᵀ` | shapes happen to fit but attention is meaningless | comparing the wrong pair of vectors |
| gradients not zeroed | loss explodes or oscillates from step 2 onwards | gradient accumulation across steps |
| gradient check fails on matmul | analytic vs numerical mismatch ~2× or transposed | `dA`/`dB` swapped or a missing transpose |
| shared mutable rows | changing one embedding changes several | a list-of-lists built with `[[0]*n]*m` |
| loss = `nan` | everything becomes `nan` after a few steps | `log(0)`, or a learning rate large enough to diverge |
| initial loss far from `ln(vocab)` | e.g. 12 instead of ~4.2 | bad initialisation scale, or a broken output head |
| unbearably slow steps | seconds-to-minutes per step | triple-nested Python loops; expected, must be measured |
