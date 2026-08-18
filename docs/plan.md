# Plan

Six stages. The order matters more than the speed: I did not let myself build a thing before I
could explain it, and I did not call a thing done before it was tested.

## 1. Work out what the task actually is

Next character prediction, `P(next | everything before)`, and why one small question like that
covers whole sentences. What comes out of the model at one position and what a training example
looks like. Written up in [notes.md](notes.md).

Doing this first is what saved me later. Writing out the input and target windows by hand gave me
the causal mask before I had written a line of code, and it also showed me the off by one in the
window index, which is a bug that does not crash and just trains on wrong data.

## 2. The math primitives

Dot product, matmul, transpose, softmax with the max subtracted for stability, layernorm, cross
entropy. Each one verified against a calculation I did on paper first.

The one worth deriving properly is softmax with cross entropy, because the two derivatives cancel
into `predicted - actual` and nothing else in the project is that clean.

## 3. Backward passes and gradient checking

No autograd. Every layer gets its own backward, derived by hand: add, matmul, softmax, layernorm,
attention, the residuals, the embedding.

Then check all of it against `(loss(w+h) - loss(w-h)) / 2h`. This is the stage everything else
depends on. A wrong gradient still gives a loss curve that goes down, just to the wrong place, so
without this check I would be guessing. Mine came out at 3.7e-08 worst relative error.

## 4. The model

Tokenizer, the data windows, embedding, positional encoding, then attention, then the feed forward
layer, then a block, then the stack. Each piece on its own, tested on its own, before the next one.

Two tests that earn their keep here. Attention rows have to sum to 1, and a token in the future must
not be able to change an output in the past. That second one is the important one, because a leak
makes the loss look better rather than worse and I would have believed it.

## 5. Training and sampling

Cross entropy, Adam, the loop. Overfit a single batch first and drive the loss to nearly zero,
because if that fails the backward pass is wrong and a long run is wasted time.

Then the real run, with the prediction written down before it starts, and sampling with temperature
and top_k after.

## 6. Look at what it learned, and write it up

The loss curve, the attention grid for every head, the nearest neighbours in the embedding table.
All as text, since matplotlib is out.

A picture on its own proves nothing, so each one gets a paragraph saying what it shows. This turned
out to be the most interesting stage: the two blocks are visibly doing different jobs, and the
embedding table has grouped characters that nothing in the code knows are related.

## Where I actually am

Stages 2 to 6 are done and the model trains and generates. Stage 1 is done for the part I needed to
build, but I have not sat down properly with the original paper yet, and the questions in
[questions.md](questions.md) are still open.
