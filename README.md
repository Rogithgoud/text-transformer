# text-transformer

A character level text generation transformer written in pure Python. No PyTorch, no TensorFlow,
no NumPy. The matrix multiply, the softmax, the layernorm, every backward pass and the optimizer
are all written out from the math.

The point was not to build a good language model. It was to understand every operation inside one,
which is why the notes and the error log are in the repo next to the code.

## Rules I set for myself

Standard library only: `math`, `random`, `json`, `time`, `os`. Nothing that does the numerical
work for me.

That also means no GPU, and not by choice. A GPU is only reachable through CUDA, and CUDA is only
reachable through the libraries I'm not using. So this runs on one CPU core, which is why the model
is small: 29,697 parameters, a 32 character context, 2 layers.

## What it produced

Trained on Tiny Shakespeare for 2000 steps, 70 minutes on my laptop. Loss went from 5.02 to 2.19
on held out text. Sampling at temperature 0.8 with top_k 10:

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

It learned the speech layout on its own: a name in capitals, a colon, a line break, then a capital
letter. `COUCICESTUS:` is not a real name but it is exactly the right shape. There are real short
words in there (how, his, And, by, with, thee, she, my, all), the word lengths are plausible,
commas fall mid line and full stops at the end of lines.

It did not learn grammar or how to spell anything long, and it was never going to. The loss of 2.19
means the model is choosing between about 9 characters at every step (e to the 2.19 is 8.9), so
long words come out the right shape with the wrong letters. The context is 32 characters, about six
words, so it cannot see a sentence. In 2000 steps at batch 4 it saw roughly 256k characters, about
a quarter of the corpus, once. Validation loss stayed at or below training loss the whole way, so
it is underfitting: the limit is model size and compute, not data.

## Checks that the thing is actually correct

This matters more here than the loss does, because a wrong backward pass still produces a loss
curve that goes down.

| Check | Result |
|---|---|
| every gradient against `(loss(w+h) - loss(w-h)) / 2h` | worst relative error 3.7e-08 over 128 cells |
| memorise one batch | 2.51 down to 0.00096 in 200 steps |
| a later token cannot change an earlier output | passes, and each position does still affect itself |
| attention rows sum to 1, upper triangle exactly 0 | passes |
| untrained loss against `ln(65) = 4.1744` | 4.35 with a small output head init |

The gradient check is the one I would point at first. It covers matmul, softmax, layernorm,
attention, the residuals and the embedding, and 3.7e-08 means the derivations are right rather
than approximately right.

## The shapes, end to end

For the run above: 32 positions, 32 channels, 65 characters in the vocabulary.

```
 ids                     (32,)        32 character ids
   |  embedding lookup                65 x 32 table, learned
 vectors                 (32, 32)     each character as 32 numbers
   |  + positional encoding           fixed sin/cos, nothing to learn
 vectors                 (32, 32)     now they also know where they are
   |  block 1                         2 heads
   |     |- layernorm, attention, add back to x
   |     |- layernorm, 32 -> 128 -> 32, add back to x
 vectors                 (32, 32)
   |  block 2                         same again
 vectors                 (32, 32)
   |  final layernorm
   |  output head                     32 x 65
 logits                  (32, 65)     65 scores at every position
   |  softmax
 probabilities           (32, 65)     sample from this
```

The width never changes between the embedding and the output head. Every block reads from that
fixed width channel and adds its result back into it.

Inside one attention head: three projections Q, K, V, then `Q` times `K` transposed to get a score
for every pair of positions, divide by the square root of the head size, set the future cells to
minus infinity, softmax each row, then take the weighted sum of V. The mask goes on before the
softmax, so `exp(-inf)` is 0 and the rows still add up to 1.

## Running it

Nothing to install. Python 3 and the standard library.

```bash
python tests/test_gradcheck.py
```

```bash
python src/train.py overfit
```

```bash
python src/train.py train
```

```bash
python src/generate.py 400 0.8 10 "KING RICHARD:"
```

```bash
python src/visualise.py
```

Run the tests and then `overfit` before any long run. `overfit` trains on a single batch until it
memorises it, and if the loss does not go to nearly zero then the backward pass is wrong and a
three hour run would be wasted. `train` needs `data/input.txt`, see [data/README.md](data/README.md)
for where I got it.

## The code

| File | What it holds |
|---|---|
| [src/matrix.py](src/matrix.py) | matmul, transpose, random init, column sums |
| [src/tokenizer.py](src/tokenizer.py) | characters to ids and back |
| [src/dataset.py](src/dataset.py) | windows of text and the targets shifted by one |
| [src/layers.py](src/layers.py) | linear, layernorm, embedding, softmax, relu, positions |
| [src/attention.py](src/attention.py) | causal self attention, one head and multi head |
| [src/block.py](src/block.py) | the feed forward layer and one full block |
| [src/model.py](src/model.py) | the whole thing assembled, save and load |
| [src/loss.py](src/loss.py) | cross entropy |
| [src/optim.py](src/optim.py) | SGD and Adam |
| [src/train.py](src/train.py) | the training loop |
| [src/generate.py](src/generate.py) | sampling with temperature and top_k |
| [src/visualise.py](src/visualise.py) | loss curve, attention grids, embedding neighbours, all as text |

There is no autograd engine. Each layer has its own `backward()` that adds to the gradients of its
own parameters and returns the gradient of its input, which is the chain rule applied locally. I
had planned to build a general graph based engine first, and changed my mind because this got to a
working training run much sooner. The cost is that adding a new layer means deriving its backward
by hand.

Batches are a loop over sequences rather than an extra dimension, so every matrix stays 2D. In pure
Python the whole thing is loop bound anyway, so a batch dimension would have cost complexity and
bought nothing, and it kept the shapes simple while I was debugging.

## Notes

- [docs/notes.md](docs/notes.md) is what the task actually is, worked out before I wrote any code
- [docs/log.md](docs/log.md) is what I did each session and what I got wrong
- [docs/questions.md](docs/questions.md) is what I still don't know
- [docs/errors/](docs/errors/) is one file per bug, cause and math reason
- [docs/experiments/](docs/experiments/) is one file per training run, prediction written
  before the run
- [docs/visuals/](docs/visuals/) is the attention grids and embedding neighbours, with what
  I think they show

## What I would do next

The loss was still dropping when the run stopped, so more steps is the cheapest improvement.
After that a wider `d_model`, then a longer context, one change at a time so the result can be
attributed. The one change already queued is a smaller init on the output head, because the run
started at 5.02 instead of the 4.17 I predicted.

Things I left out on purpose: dropout, learning rate schedules, subword tokenisation, weight
tying, KV caching.
