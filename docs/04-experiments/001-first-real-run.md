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
| parameter count | 29,697 |
| random seed | 1337 |

## Environment

Pure Python 3, standard library only, single core, no GPU. Windows laptop.

## Results

| Metric | Value |
|---|---|
| initial loss | 5.0217 |
| expected initial loss `ln(65)` | 4.1744 |
| final train loss | 2.2595 |
| final val loss | 2.1890 |
| wall-clock time | 70.3 min |
| seconds per step | 1.93 (5.20 on step 1, that includes startup) |

**Loss curve:** raw numbers in `runs/train_log.txt`, drawn in
[../05-visuals/10-loss-curve.txt](../05-visuals/10-loss-curve.txt), read in the interpretations
section of [../05-visuals/README.md](../05-visuals/README.md).

### The initial loss missed my prediction

I predicted 4.17 and got 5.02. Not a bug. The output head is initialised with
`scale = 1/sqrt(32) = 0.177`, so the very first logits already have spread, which means the model
starts out confidently wrong rather than uniformly clueless, and confident wrong answers cost more
than `ln(V)`. The validation loss was already 4.40 one step later, so it corrects itself
immediately and the run was not worth restarting.

Worth knowing: `ln(V)` is the loss of a model whose logits are all *equal*, which is only the same
thing as an untrained model if the initialisation is small enough. That is a slightly different
claim from the one I wrote in the corpus README, and the sharper version is the right one.

## Sample output

`python src/generate.py 400 0.8 10 "KING RICHARD:"`

```
KING RICHARD:

Wit haw brest how his tast tour amar tworne,
And shive by tore shives, coue hanceld
Bun he she sond this him hall torre,
Shall ling blave lery thar thee with tas hat, lande wather all cand hastess ting
Heell shallost the st she my the scold.

COUCICESTUS:
Showes lond lichemees lich somerave
Thalls the sto he the heameserard
Shigh thath wis mold tunghe there cas whither it lour anding, st till ti
```

## Interpretation

The hypothesis held: 2.19 val, inside the 2.0-2.5 I predicted, and the failure modes are the ones
I expected.

What it clearly learned:

- **The line and speech structure.** `COUCICESTUS:` on its own line, in capitals, ending in a
  colon, followed by a line break and then a capital letter. It invented a speaker name that is
  not a word but has exactly the right shape. Nothing told it this pattern existed.
- **Real words.** his, how, And, by, with, the, she, my, shall, there, it, all, this, him, he.
  Mostly short and mostly common, which is what a 32 character context and 41k parameters can
  hold. (I first wrote 41,761 in this file from memory instead of reading it off the model. The
  real count from `model.num_params()` is 29,697. Numbers go in these docs from the output, not
  from what I think I remember.)
- **Word length and spacing.** Nothing is 20 characters long, and the spaces fall at plausible
  intervals, so the base frequency of space (15%) has been learned properly rather than crudely.
- **Punctuation position.** Commas mid-line, full stops at the end of lines. Consistent with the
  embedding result, where `.` `?` `!` collapsed into nearly the same vector.
- **Capitals after newlines,** which is a rule that needs two characters of context to spot.

What it clearly did not learn:

- **Grammar.** There is no subject-verb agreement, no clause structure, nothing beyond the local
  level. Expected: the whole context is 32 characters, which is about 6 words, so it cannot see a
  sentence.
- **Spelling of anything long.** "shive", "hanceld", "lichemees" are the right shape and the wrong
  letters. It has learned which letters plausibly follow which, not which sequences are words.
- **Meaning,** obviously.

Why the output is this good and no better, specifically: 29,697 parameters, a 32 character memory,
and 2000 steps, which at batch 4 and block 32 means the model saw about 256k characters, roughly a
quarter of the corpus, once. That is the entire budget. The output is what that budget buys.

Cross-checks that make me trust the number rather than just liking it:

- gradient check at 3.7e-08 before the run, so the gradients are right
- the mask test passes, so it is not reading the answer, which is the one bug that would make a
  loss look good for a fake reason
- validation loss at or below train loss the whole way, so it is not memorising the training text
- the samples are held-out-plausible rather than copied lines from the corpus

## What I would change next, and why

One change, with a prediction, for experiment 002:

**Initialise the output head with scale 0.02 instead of 1/sqrt(d_model).** Prediction: the initial
loss lands at about 4.17 instead of 5.02, and the first 50 steps stop being spent undoing a bad
starting point. I don't expect a big change in the final loss, maybe 0.02 to 0.05 better, but it
makes the `ln(V)` sanity check meaningful, which is worth more than the loss improvement.

After that, in order of what I expect to matter most: more steps (it was still improving when it
stopped), then `d_model` 32 to 64, then `block_size` 32 to 64. Each on its own, so the result can
be attributed.
