# Log

What I did each session, and what I got wrong. Written the same day.

## 2026-08-17

Set up the repo and worked out the plan before writing code. Decided how strict to be about "no
libraries" and settled on standard library only, written down in the readme so it does not drift
later.

Then the first real bit of work: what a language model actually is. Wrote out the task as
`P(next | previous)`, the generation loop, what comes out at one position, and did the input and
target windows by hand for "hello", then again for "banana" from memory to check I had it rather
than was following along.

Three things landed today.

The whole task is one small question multiplied down a chain, so modelling all of language reduces
to answering "what comes next" well.

The target is literally the input shifted left by one, and that one pair contains as many separate
prediction problems as there are positions. Writing out what each position is allowed to see gave me
a triangle, and that triangle is the causal mask. I did not expect to get to the mask before reading
the paper.

Logits are not probabilities. They are raw unbounded scores, and softmax does two jobs on them:
`exp` to force them positive and exaggerate the gaps, then divide by the sum.

Also spotted the off by one in the window index before writing any code. With 5 characters and a
window of 4, the only legal start is 0, because the target needs one character past the input. That
is a bug that would not crash, it would just train on wrong data.

Two things I could not explain and wrote down instead of pretending: why nobody fixes the mismatch
between training on real text and generating from your own output, and whether softmax's gap
exaggeration makes an untrained model overconfident.

## 2026-08-18

Built the whole thing and trained it.

Wrote all 11 modules and 5 test files. Matrix helpers, tokenizer, dataset, layers, attention, block,
model, loss, optimizer, training loop, sampling, and the text based plots. Then 2000 steps on Tiny
Shakespeare, 70 minutes, loss 5.02 down to 2.19 on held out text.

Two decisions I made to get there in one day rather than a week.

No autograd engine. Each layer has its own backward that adds to the gradients of its own parameters
and returns the gradient of its input. Same chain rule, applied locally, no graph to build or sort.
I had planned to build the general engine first and changed my mind. The cost is that a new layer
means deriving its backward by hand.

Batches are a loop over sequences instead of an extra dimension, so every matrix stays 2D. Pure
Python is loop bound anyway, so the dimension would have cost complexity and bought nothing, and it
kept the shapes simple while debugging.

The backward pass stops being mysterious once you write it out. Reading my attention forward and
backward side by side, the backward is the forward reversed line for line. `out = probs @ V` becomes
`dprobs = dout @ V^T` and `dV = probs^T @ dout`. The shapes only fit one way round, which is how I
now check a transpose when I am unsure, and it caught two mistakes.

The tests earned their keep immediately, and each one caught a different class of problem.

The gradient check is the only reason I trust any of this. 3.7e-08 worst relative error over 128
sampled weights covers matmul, softmax, layernorm, attention, the residuals and the embedding.
Without it I would have been guessing whether my derivations were right.

Overfitting one batch, 2.51 down to 0.00096, proved forward, backward and Adam work together before
I spent an hour on a real run.

The mask test protects the result from being fake. If the future leaked into the past the loss would
look better, not worse, and I would have believed it.

Then the plots turned out to be the most interesting part. The two heads in each block are doing
visibly different jobs, local in the first block and structural in the second. And the embedding
table has grouped `.` `?` `!` at 0.96 cosine similarity and paired up upper and lower case versions
of the same letter, which is not something anything in my code knows about. It got there purely from
those characters being usable in the same places.

Two things went wrong.

The gradient check failed on the first attempt at 3.3e-03 and I assumed layernorm was wrong, since
it has the messiest derivation. It wasn't. The test was dividing float noise by float noise on
weights whose true gradient is zero. Written up in errors/001. About 20 minutes lost, most of it
suspecting the wrong thing.

I predicted the run would start at `ln(65) = 4.17` and it started at 5.02. The output head is
initialised at `1/sqrt(32)`, so the first logits already have spread and the model starts
confidently wrong, which costs more than being uniformly clueless. The sharper version of the rule
is that `ln(V)` is the loss when the logits are all equal. That is the one change for the next run.

Also caught myself writing 41,761 as the parameter count from memory when the real number is 29,697.
Fixed it and left a note in the file. Numbers go in these docs from the output.

Next: read the paper properly now that I have built the thing it describes, and run the next
experiment with the smaller output head init.
