# Run 1: Tiny Shakespeare, 2000 steps

2026-08-18.

## What I predicted, before starting

The loss should start near `ln(65) = 4.1744` and fall fast for the first few hundred steps as the
model picks up base character frequencies, then slow down once it has to start on spelling. I expect
2.0 to 2.5 at the end, output with plausible word lengths and line breaks, some real short words,
and no grammar at all. Much below 2.0 at this size would surprise me.

Writing this down first so I can't rationalise whatever comes out.

## Setup

| | |
|---|---|
| corpus | Tiny Shakespeare, 1,115,394 characters, vocabulary 65 |
| context | 32 |
| d_model | 32 |
| layers | 2 |
| heads | 2 |
| feed forward | 128, so 4x |
| batch | 4 |
| optimizer | Adam, 3e-3, b1 0.9, b2 0.999, eps 1e-8 |
| steps | 2000 |
| parameters | 29,697 |
| seed | 1337 |

Pure Python, standard library only, one core, no GPU. Windows laptop.

## What happened

| | |
|---|---|
| loss at step 1 | 5.0217 |
| what I expected at step 1 | 4.1744 |
| final training loss | 2.2595 |
| final validation loss | 2.1890 |
| time | 70.3 minutes |
| per step | 1.93 seconds, the 5.20 on step 1 includes startup |

Raw numbers in `runs/train_log.txt`, curve and reading in [../visuals/notes.md](../visuals/notes.md).

### The starting loss missed my prediction

I predicted 4.17 and got 5.02. Not a bug. The output head is initialised with
`scale = 1/sqrt(32) = 0.177`, so the very first logits already have spread, which means the model
starts out confidently wrong rather than uniformly clueless, and confident wrong answers cost more
than `ln(V)`. Validation was already at 4.40 one step later, so it corrects itself immediately and
the run was not worth restarting.

The sharper version of the rule, which is what I should have written in the first place: `ln(V)` is
the loss when the logits are all *equal*. That only matches an untrained model if the initialisation
is small enough to make it so.

## Sample

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

## Reading it

The prediction held. 2.19 validation, inside the 2.0 to 2.5 I said, and the failure modes are the
ones I expected.

What it clearly learned. The speech layout, `COUCICESTUS:` on its own line in capitals with a colon,
then a line break, then a capital letter. That name is invented but has exactly the right shape, and
nothing told the model that pattern exists. Real words, mostly short and common: his, how, And, by,
with, the, she, my, shall, there, it, all, this, him, he. Plausible word lengths and spacing, so the
15% base rate of the space character has been learned properly rather than crudely. Commas mid line
and full stops at line ends, which is consistent with the embedding result where `.` `?` `!` collapsed
into nearly the same vector. And capitals after newlines, which needs two characters of context.

What it clearly did not learn. Grammar, in any form. The whole context is 32 characters, about six
words, so it cannot see a sentence. Spelling of anything long: shive, hanceld, lichemees are the
right shape and the wrong letters, because it has learned which letters plausibly follow which and
not which sequences are words.

Why it is this good and no better, in numbers. 29,697 parameters, a 32 character memory, and 2000
steps at batch 4 and context 32, so the model saw about 256k characters, roughly a quarter of the
corpus, once. That is the whole budget, and the output is what the budget buys.

Cross checks, so I trust the number rather than just liking it. The gradient check was at 3.7e-08
before the run, so the gradients are right. The mask test passes, so it is not reading the answer,
which is the one bug that makes a loss look good for a fake reason. Validation stayed at or below
training the whole way, so it is not memorising the training text. And the samples are new text
rather than lines copied out of the corpus.

One correction to this file: I first wrote the parameter count as 41,761 from memory. The real number
from `model.num_params()` is 29,697.

## Next run, one change

Initialise the output head at 0.02 instead of `1/sqrt(d_model)`. Prediction: the starting loss lands
near 4.17 instead of 5.02, and the first 50 steps stop being spent undoing a bad starting point. I
don't expect much difference in the final loss, maybe 0.02 to 0.05, but it makes the `ln(V)` check
meaningful, which is worth more than the loss.

After that, in the order I expect to matter: more steps, since it was still improving. Then `d_model`
32 to 64. Then context 32 to 64. One at a time, so each result can be attributed to something.
