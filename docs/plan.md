# Plan

Six stages. I did not let myself build something before I could explain it, and did not call
anything done before it was tested.

**1. Work out what the task is.** Next character prediction, and what a training example looks like.
Doing this first paid off twice: writing the windows out by hand gave me the causal mask before I
touched attention, and it showed me the off by one in the window index, which is a bug that does not
crash and just trains on wrong data. In [notes.md](notes.md).

**2. Math primitives.** Dot product, matmul, transpose, softmax with the max subtracted so `exp`
does not overflow, layernorm, cross entropy. Each one checked against a calculation I did on paper.

**3. Backward passes.** No autograd, so every layer gets its own, derived by hand. Then check all of
it against finite differences. Everything else depends on this stage, because a wrong gradient still
gives you a loss curve that goes down, just to the wrong place. Mine came out at 3.7e-08. In
[math.md](math.md).

**4. The model.** Tokenizer, data windows, embedding, positions, attention, feed forward, block,
stack. Each piece tested on its own before the next one. Two tests matter here: attention rows have
to add to 1, and a character in the future must not be able to change an earlier output. The second
one is the important one, because if the mask leaks the loss looks better and I would believe it.

**5. Training and sampling.** Cross entropy, Adam, the loop. Overfit one batch first and drive the
loss to nearly zero, because if that fails a long run is wasted. Then the real run with the
prediction written down beforehand.

**6. Look at what it learned.** Loss curve, attention grids, embedding similarity, positional
encoding. This turned out to be the most interesting part: the two blocks are visibly doing
different jobs and the embedding table grouped characters that nothing in my code knows are related.
In [visuals/notes.md](visuals/notes.md).

Where I am: stages 2 to 6 are done, it trains and generates. I have not sat down with the original
paper properly yet, and the things in [questions.md](questions.md) are still open.
