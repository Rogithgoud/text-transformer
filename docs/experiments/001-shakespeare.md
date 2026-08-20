# Run 1: Tiny Shakespeare, 2000 steps

2026-08-18.

## What I predicted before starting

Loss should start near `ln(65) = 4.17`, drop fast while it picks up which characters are common,
then slow down when it has to start on spelling. I expect 2.0 to 2.5 at the end, with plausible word
lengths and line breaks, a few real short words, and no grammar. Below 2.0 would surprise me.

Writing it first so I cannot talk myself into whatever comes out.

## Setup

Tiny Shakespeare, 1,115,394 characters, 65 unique. Context 32, d_model 32, 2 layers, 2 heads, feed
forward 128, batch 4, Adam at 3e-3, 2000 steps, seed 1337. 29,697 parameters. One CPU core.

## What happened

Started at 5.0217. Finished at 2.2595 training, 2.1890 on held out text. 70.3 minutes, about 1.93
seconds a step.

The prediction held on the final number and missed on the starting one. I said 4.17 and got 5.02.
Not a bug. My output layer is initialised at `1/sqrt(32)`, so the very first logits already have
spread and the model starts out confidently wrong, which costs more than knowing nothing.
Validation was already at 4.40 one step later so it fixes itself immediately, and it was not worth
restarting an hour long run over.

The rule I should have written down is that `ln(65)` is the loss when all the logits are *equal*,
which only matches an untrained model if the starting weights are small enough.

Curve and the rest of the plots in [../visuals/notes.md](../visuals/notes.md).

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
```

## What I make of it

It got the layout of a play on its own. Capitals, colon, line break, capital letter.
`COUCICESTUS` is not a name but it is the right shape, and nothing told the model that pattern
exists. Real short words all through it (his, how, And, by, with, she, shall, there, it), spacing
and word lengths look right, commas mid line and full stops at the ends.

No grammar at all, and no spelling of anything long. shive, hanceld, lichemees are the right shape
with the wrong letters. That is what a loss of 2.19 means in practice: `e^2.19` is about 9, so at
every character it is effectively choosing between 9 options, and nine way guesses chained together
give you exactly this.

Why it is this good and no better, in numbers: 29,697 parameters, 32 characters of memory, and 2000
steps at batch 4 means it saw about 256k characters, a quarter of the corpus, once. That is the whole
budget.

Reasons I trust the number rather than just liking it. Gradient check at 3.7e-08 before the run.
Mask test passes, so it is not reading the answer, which is the one bug that makes a loss look good
for a fake reason. Validation stayed at or below training the whole way, so it is not memorising.

One correction: I first wrote the parameter count as 41,761 from memory. It is 29,697.

## Next run, one change

Output head starting at 0.02 instead of `1/sqrt(32)`. I expect the starting loss to land near 4.17
and the first 50 steps to stop being spent undoing a bad start. I do not expect much difference at
the end, maybe 0.02 to 0.05, but it makes the 4.17 check mean something.

Then more steps, since it was still improving. Then a wider model, then a longer context. One at a
time so I can tell what did what.
