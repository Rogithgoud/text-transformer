"""Sampling text from a trained checkpoint.

    python src/generate.py                       200 chars, temperature 1.0
    python src/generate.py 400 0.8 10 "KING"     chars, temperature, top_k, prompt

The loop is the autoregressive one: take the logits at the LAST position, turn
them into probabilities, pick a character, append it to the input, repeat. The
model's own output becomes its next input, which is also why one bad character
can send the rest of the sample off the rails.

The context is cropped to the last block_size characters, because that is all the
positional table covers. The model has no memory beyond that window at all.

Three ways to pick a character:

  greedy      always take the highest probability. Sounds best, is worst: with no
              randomness it falls into loops like "the the the".
  temperature divide the logits by T before the softmax. T < 1 sharpens the
              distribution (safer, more repetitive), T > 1 flattens it (more
              variety, more nonsense). T = 1 is the model's honest opinion.
  top_k       keep only the k most likely characters and renormalise, so the long
              tail of nearly-impossible characters can never be picked.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Model
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(os.path.dirname(HERE), "checkpoints")


def pick(logits, temperature, top_k, rng):
    """One character id, sampled from the logits at a single position."""
    if temperature <= 0.0:
        return max(range(len(logits)), key=lambda i: logits[i])     # greedy

    scaled = [v / temperature for v in logits]

    keep = list(range(len(scaled)))
    if top_k and top_k < len(scaled):
        keep = sorted(keep, key=lambda i: scaled[i], reverse=True)[:top_k]

    m = max(scaled[i] for i in keep)                                # overflow guard
    exps = [(i, math.exp(scaled[i] - m)) for i in keep]
    total = sum(e for _, e in exps)

    # walk the cumulative probability until it passes a uniform draw
    r = rng.random() * total
    upto = 0.0
    for i, e in exps:
        upto += e
        if upto >= r:
            return i
    return exps[-1][0]


def generate(model, tok, prompt, n_chars, temperature, top_k, rng):
    ids = tok.encode(prompt) if prompt else [rng.randrange(tok.vocab_size)]
    out = list(ids)
    for _ in range(n_chars):
        context = out[-model.block_size:]        # crop to what the model can see
        logits = model.forward(context)
        out.append(pick(logits[-1], temperature, top_k, rng))   # last position only
    return tok.decode(out)


def main():
    n_chars = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    prompt = sys.argv[4] if len(sys.argv) > 4 else ""

    tok_path = os.path.join(CKPT, "tokenizer.json")
    model_path = os.path.join(CKPT, "model.json")
    if not os.path.exists(model_path):
        print("no checkpoint yet, run: python src/train.py train")
        sys.exit(1)

    tok = Tokenizer.load(tok_path)
    rng = random.Random()

    import json
    with open(model_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    model = Model(cfg["vocab_size"], cfg["block_size"], cfg["d_model"],
                  cfg["n_heads"], cfg["n_layers"], rng)
    model.load(model_path)

    print("temperature %.2f, top_k %s, prompt %r" % (temperature, top_k or "off", prompt))
    print("-" * 60)
    print(generate(model, tok, prompt, n_chars, temperature, top_k, rng))
    print("-" * 60)


if __name__ == "__main__":
    main()
