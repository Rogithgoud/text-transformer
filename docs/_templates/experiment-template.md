# Experiment NNN: <short name>

> Copy into `docs/04-experiments/NNN-short-name.md`.

**Date:** YYYY-MM-DD
**Commit:** `<git sha>`
**Goal / hypothesis:** what I expected to happen, and why, written *before* the run.

---

## Configuration

| Setting | Value |
|---|---|
| dataset / size |  |
| vocab size |  |
| block_size (context) |  |
| n_layers |  |
| n_heads |  |
| d_model |  |
| d_ff |  |
| batch size |  |
| optimizer |  |
| learning rate |  |
| steps |  |
| parameter count |  |
| random seed |  |

## Environment

Python version, machine, single-threaded.

## Results

| Metric | Value |
|---|---|
| initial loss |  |
| expected initial loss (`ln(vocab)`) |  |
| final train loss |  |
| final val loss |  |
| wall-clock time |  |
| seconds per step |  |

**Loss curve:** (ASCII plot or link into `docs/05-visuals/`)

```
```

## Sample output

Prompt, temperature, top-k, and the raw generated text — unedited, including the bad parts.

```
```

## Interpretation

Did it match the hypothesis? What does the curve's shape say — underfitting, overfitting,
diverging, plateauing? What does the sample text reveal about what the model has and has not
learned?

## What I would change next, and why

One change at a time, with a prediction of its effect. That prediction becomes the hypothesis
of the next experiment.
