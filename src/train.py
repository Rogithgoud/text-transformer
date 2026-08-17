"""Training loop.

Two modes:

    python src/train.py overfit      one tiny batch, memorise it, loss -> ~0
    python src/train.py train        the real run on data/input.txt

Run overfit FIRST, always. If the model cannot memorise a single batch then the
backward pass is wrong, and there is no point starting a long run. It is the
cheapest possible check that the whole forward/backward/optimizer chain works.

One step is:

    zero the grads   (they accumulate, so leftovers from the last step would be
                      added to this one)
    for each sequence in the batch:
        forward, loss, backward         gradients pile up across the batch
    divide the grads by batch_size      so the batch size does not secretly
                                        change the learning rate
    optimizer.step()                    nudge every weight downhill
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dataset as D
from loss import cross_entropy
from model import Model
from optim import Adam
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "input.txt")
RUNS = os.path.join(ROOT, "runs")
CKPT = os.path.join(ROOT, "checkpoints")


def average_grads(model, n):
    """The batch was a loop, so the gradients are a sum. Turn it into a mean."""
    inv = 1.0 / n
    for p in model.parameters():
        for row in p.grad:
            for j in range(len(row)):
                row[j] *= inv


def run_step(model, opt, batch):
    """One optimizer step over a batch of (input ids, target ids) pairs."""
    model.zero_grad()
    total = 0.0
    for x, y in batch:
        logits = model.forward(x)
        loss, dlogits = cross_entropy(logits, y)
        model.backward(dlogits)
        total += loss
    average_grads(model, len(batch))
    opt.step()
    return total / len(batch)


def estimate_loss(model, ids, block_size, batch_size, rng):
    """Loss on data the optimizer is not being run on. No backward pass here."""
    batch = D.get_batch(ids, block_size, batch_size, rng)
    total = 0.0
    for x, y in batch:
        loss, _ = cross_entropy(model.forward(x), y)
        total += loss
    return total / len(batch)


def overfit():
    """Memorise one fixed batch. Success = loss goes to nearly zero."""
    print("overfit test: can the model memorise a single batch?")
    rng = random.Random(1234)
    text = "hello world, hello there. "
    tok = Tokenizer.from_text(text)
    ids = tok.encode(text * 4)

    block_size = 16
    model = Model(vocab_size=tok.vocab_size, block_size=block_size,
                  d_model=16, n_heads=2, n_layers=2, rng=rng)
    opt = Adam(model.parameters(), lr=0.01)
    batch = D.get_batch(ids, block_size, 2, rng)

    print("vocab %d, params %d" % (tok.vocab_size, model.num_params()))
    start = time.time()
    loss = None
    for step in range(1, 201):
        loss = run_step(model, opt, batch)          # same batch every time
        if step % 20 == 0 or step == 1:
            print("  step %3d   loss %.6f" % (step, loss))
    print("took %.1fs" % (time.time() - start))

    if loss < 0.05:
        print("PASS: loss went to ~0, so forward, backward and Adam all work")
    else:
        print("FAIL: loss stuck at %.4f, the backward pass is probably wrong" % loss)
        sys.exit(1)


def train():
    if not os.path.exists(DATA):
        print("no corpus at %s" % DATA)
        sys.exit(1)

    # config for one night on one CPU core
    cfg = {
        "block_size": 32,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "batch_size": 4,
        "lr": 3e-3,
        "steps": 2000,
        "eval_every": 25,
        "ckpt_every": 100,
        "seed": 1337,
    }

    rng = random.Random(cfg["seed"])
    text = D.load_text(DATA)
    tok = Tokenizer.from_text(text)
    ids = tok.encode(text)
    train_ids, val_ids = D.split_train_val(ids, 0.1)

    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    tok.save(os.path.join(CKPT, "tokenizer.json"))

    model = Model(vocab_size=tok.vocab_size, block_size=cfg["block_size"],
                  d_model=cfg["d_model"], n_heads=cfg["n_heads"],
                  n_layers=cfg["n_layers"], rng=rng)
    opt = Adam(model.parameters(), lr=cfg["lr"])

    import math
    print("corpus %d chars, vocab %d, train %d, val %d"
          % (len(text), tok.vocab_size, len(train_ids), len(val_ids)))
    print("params %d" % model.num_params())
    print("expected starting loss ~ln(%d) = %.4f" % (tok.vocab_size, math.log(tok.vocab_size)))
    print("config %s" % json.dumps(cfg))

    log_path = os.path.join(RUNS, "train_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# %s\n" % json.dumps(cfg))
        f.write("step\ttrain_loss\tval_loss\tsecs\n")

    start = time.time()
    for step in range(1, cfg["steps"] + 1):
        t0 = time.time()
        loss = run_step(model, opt, D.get_batch(train_ids, cfg["block_size"],
                                                cfg["batch_size"], rng))
        dt = time.time() - t0

        if step % cfg["eval_every"] == 0 or step == 1:
            val = estimate_loss(model, val_ids, cfg["block_size"], cfg["batch_size"], rng)
            print("step %5d   train %.4f   val %.4f   %.2fs/step   elapsed %.1fmin"
                  % (step, loss, val, dt, (time.time() - start) / 60.0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("%d\t%.6f\t%.6f\t%.3f\n" % (step, loss, val, dt))

        if step % cfg["ckpt_every"] == 0:
            model.save(os.path.join(CKPT, "model.json"))

    model.save(os.path.join(CKPT, "model.json"))
    print("done in %.1f min, checkpoint written" % ((time.time() - start) / 60.0))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "overfit"
    if mode == "overfit":
        overfit()
    elif mode == "train":
        train()
    else:
        print("usage: python src/train.py [overfit|train]")
        sys.exit(1)
