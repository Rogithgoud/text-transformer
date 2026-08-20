"""Plots, from the trained checkpoint and the training log.

    python src/visualise.py

Writes four PNGs into docs/visuals/. matplotlib is allowed here because it only
draws pictures, it does not do any of the model's math.

    loss.png        training and validation loss
    attention.png   one attention grid per head
    embedding.png   how close the learned character vectors are to each other
    positions.png   the positional encoding table
"""

import json
import math
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")               # no window, just save files
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Model
from tokenizer import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "docs", "visuals")


def label(ch):
    return {"\n": "\\n", " ": "sp"}.get(ch, ch)


def load_model():
    tok = Tokenizer.load(os.path.join(CKPT, "tokenizer.json"))
    with open(os.path.join(CKPT, "model.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    m = Model(cfg["vocab_size"], cfg["block_size"], cfg["d_model"],
              cfg["n_heads"], cfg["n_layers"], random.Random(0))
    m.load(os.path.join(CKPT, "model.json"))
    return m, tok


def loss_plot():
    steps, train, val = [], [], []
    with open(os.path.join(RUNS, "train_log.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("step"):
                continue
            a, b, c, _ = line.split("\t")
            steps.append(int(a))
            train.append(float(b))
            val.append(float(c))

    plt.figure(figsize=(9, 5))
    plt.plot(steps, train, linewidth=0.9, alpha=0.7, label="train")
    plt.plot(steps, val, linewidth=1.6, label="validation")
    plt.axhline(math.log(65), linestyle="--", color="grey",
                label="ln(65) = 4.17, knows nothing")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("loss, 2000 steps on Tiny Shakespeare")
    plt.legend()
    plt.grid(alpha=0.3)
    path = os.path.join(OUT, "loss.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print("wrote %s  (start %.2f, end train %.2f val %.2f)"
          % (path, train[0], train[-1], val[-1]))


def attention_plot(sentence="First Citizen:\nBefore we go"):
    """Each grid cell (i, j) is how much position i attended to position j.
    Only the lower triangle can be non zero, that is the causal mask."""
    m, tok = load_model()
    ids = tok.encode(sentence)[:m.block_size]
    m.forward(ids)
    ticks = [label(tok.itos[i]) for i in ids]

    heads = [(b_i, h_i, h) for b_i, b in enumerate(m.blocks)
             for h_i, h in enumerate(b.attn.heads)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for ax, (b_i, h_i, head) in zip(axes.flat, heads):
        ax.imshow(head.probs, cmap="viridis")
        ax.set_title("block %d, head %d" % (b_i, h_i))
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels(ticks, fontsize=6)
        ax.set_yticks(range(len(ticks)))
        ax.set_yticklabels(ticks, fontsize=6)
        ax.set_xlabel("looking at")
        ax.set_ylabel("from position")
    fig.suptitle('attention for "First Citizen:\\nBefore we go"')
    path = os.path.join(OUT, "attention.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print("wrote %s" % path)


def cosine(u, v):
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return 0.0 if nu == 0 or nv == 0 else sum(a * b for a, b in zip(u, v)) / (nu * nv)


def embedding_plot(probes="etaoinshrlu ATQ.?!,;\n"):
    """Cosine similarity between the learned vectors of some characters.
    Nothing in the code groups characters, so any pattern here was learned."""
    m, tok = load_model()
    table = m.tok_emb.weight.data
    chars = [c for c in probes if c in tok.stoi]
    rows = [table[tok.stoi[c]] for c in chars]
    grid = [[cosine(u, v) for v in rows] for u in rows]

    plt.figure(figsize=(9, 8))
    plt.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(label="cosine similarity")
    plt.xticks(range(len(chars)), [label(c) for c in chars], fontsize=8)
    plt.yticks(range(len(chars)), [label(c) for c in chars], fontsize=8)
    plt.title("how close the learned character vectors are")
    path = os.path.join(OUT, "embedding.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()

    # the few pairs worth quoting in the notes
    for a in [".", "t", "a", "q", " "]:
        if a not in tok.stoi:
            continue
        sims = sorted(((cosine(table[tok.stoi[a]], table[j]), tok.itos[j])
                       for j in range(len(table)) if j != tok.stoi[a]), reverse=True)
        print("  %-3s closest: %s" % (label(a),
              ", ".join("%s %.2f" % (label(c), s) for s, c in sims[:4])))
    print("wrote %s" % path)


def positions_plot():
    """The sin/cos table. Every row is one position's fingerprint."""
    m, _ = load_model()
    plt.figure(figsize=(9, 5))
    plt.imshow(m.pos, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="value")
    plt.xlabel("channel (0 to 31)")
    plt.ylabel("position in the window")
    plt.title("positional encoding, fixed sin and cos, nothing learned")
    path = os.path.join(OUT, "positions.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print("wrote %s" % path)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "loss"):
        loss_plot()
    if what in ("all", "attn"):
        attention_plot()
    if what in ("all", "emb"):
        embedding_plot()
    if what in ("all", "pos"):
        positions_plot()
