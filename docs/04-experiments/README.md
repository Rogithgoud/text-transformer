# 04 — Experiments

One file per training run, from
[`../_templates/experiment-template.md`](../_templates/experiment-template.md).
Every run, including the failed ones. Especially the failed ones.

**Rule:** the hypothesis is written *before* the run starts. Change **one** thing per
experiment, otherwise the result cannot be attributed.

## Index

| # | File | Change from previous | Steps | Final loss | Time | Verdict |
|---|---|---|---|---|---|---|
| 001 |  | baseline: overfit a single batch | | | | |

## Suggested progression

| # | Purpose | Success criterion |
|---|---|---|
| 001 | overfit one tiny batch | loss → ~0; if not, the backward pass is wrong |
| 002 | tiny corpus, few steps, verify the loop runs end to end | loss decreases at all |
| 003 | baseline real run | loss well below `ln(vocab)`; spacing looks like text |
| 004 | vary learning rate only | identify divergence threshold |
| 005 | SGD vs Adam, everything else fixed | measured difference in convergence |
| 006 | vary n_layers only | measured effect on loss and on time per step |
| 007 | vary n_heads only | measured effect, plus what the heatmaps show |
| 008 | vary block_size (context) only | measured effect on loss |
| 009 | final run for the README | best honest result achievable in reasonable time |
