# Experiment 001: first real run on Tiny Shakespeare

**Date:** 2026-08-18
**Commit:** the run started from the tree at `72f58a9` plus the corpus README
**Goal / hypothesis:** written before starting. The loss should begin near
`ln(65) = 4.1744` and fall fast for the first few hundred steps as the model picks up base
character frequencies (space 15%, 'e' 8%), then slow down as it starts on spelling. I expect
somewhere around 2.0-2.5 after a couple of thousand steps, output with plausible word lengths and
line breaks, some real short words, and no real grammar. Anything much below 2.0 at this size
would surprise me.

---

## Configuration

| Setting | Value |
|---|---|
| dataset | Tiny Shakespeare, 1,115,394 chars |
| vocab size | 65 |
| block_size (context) | 32 |
| n_layers | 2 |
| n_heads | 2 |
| d_model | 32 |
| d_ff | 128 (4x) |
| batch size | 4 |
| optimizer | Adam, b1 0.9, b2 0.999, eps 1e-8 |
| learning rate | 3e-3 |
| steps | 2000 |
| parameter count | *filled in from the run output* |
| random seed | 1337 |

## Environment

Pure Python 3, standard library only, single core, no GPU. Windows laptop.

## Results

| Metric | Value |
|---|---|
| initial loss | |
| expected initial loss `ln(65)` | 4.1744 |
| final train loss | |
| final val loss | |
| wall-clock time | |
| seconds per step | |

**Loss curve:** raw numbers are logged in `runs/train_log.txt`, plotted in
`docs/05-visuals/10-loss-curve.txt`.

## Sample output

*(after the run, from `python src/generate.py 400 0.8 10`)*

## Interpretation

*(did it match the hypothesis? what does the curve shape say? what does the sample text show the
model has and has not learned?)*

## What I would change next, and why

*(one change, with a prediction)*
