"""Visual artifacts, without matplotlib. Everything is printed as text.

    python src/visualise.py

Writes three files into docs/05-visuals/:

    10-loss-curve.txt            training and validation loss, as ASCII
    08-attention-heads.txt       one grid per head, for a fixed sentence
    09-embedding-neighbours.md   nearest characters in embedding space

A picture on its own proves nothing, so each file carries a note on what it
shows. The interpretations are written up in docs/05-visuals/README.md.
"""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import Model
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "checkpoints")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "docs", "05-visuals")

SHADES = " .:-=+*#%@"          # low to high


def show_char(ch):
    return {"\n": "\\n", " ": "_", "\t": "\\t"}.get(ch, ch)


def loss_curve(width=70, height=20):
    """Read runs/train_log.txt and draw both losses on one grid."""
    steps, train, val = [], [], []
    with open(os.path.join(RUNS, "train_log.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("step"):
                continue
            a, b, c, _ = line.split("\t")
            steps.append(int(a))
            train.append(float(b))
            val.append(float(c))

    lo = min(min(train), min(val))
    hi = max(max(train), max(val))
    last_step = steps[-1]

    grid = [[" "] * width for _ in range(height)]

    def plot(xs, ys, mark):
        for s, v in zip(xs, ys):
            col = int((s - steps[0]) / max(1, last_step - steps[0]) * (width - 1))
            row = int((hi - v) / (hi - lo) * (height - 1))
            grid[row][col] = mark

    plot(steps, val, "v")
    plot(steps, train, "t")        # train drawn last so it wins any overlap

    lines = []
    lines.append("training loss (t) and validation loss (v)")
    lines.append("")
    for r, row in enumerate(grid):
        value = hi - (hi - lo) * r / (height - 1)
        lines.append("%6.2f | %s" % (value, "".join(row)))
    lines.append("       +" + "-" * width)
    lines.append("        %-*s%s" % (width - len(str(last_step)), steps[0], last_step))
    lines.append("        step")
    lines.append("")
    lines.append("start  train %.4f   val %.4f" % (train[0], val[0]))
    lines.append("end    train %.4f   val %.4f" % (train[-1], val[-1]))
    lines.append("ln(vocab) = %.4f is where an untrained model sits" % math.log(65))
    lines.append("")
    lines.append("the train line is noisy because every step is a fresh batch of only 4")
    lines.append("sequences, so each point is one small sample, not the whole corpus.")
    lines.append("val is measured on held out text the optimizer never touched.")

    path = os.path.join(OUT, "10-loss-curve.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % path)
    print("\n".join(lines))


def load_model():
    tok = Tokenizer.load(os.path.join(CKPT, "tokenizer.json"))
    with open(os.path.join(CKPT, "model.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    m = Model(cfg["vocab_size"], cfg["block_size"], cfg["d_model"],
              cfg["n_heads"], cfg["n_layers"], random.Random(0))
    m.load(os.path.join(CKPT, "model.json"))
    return m, tok


def attention_maps(sentence="First Citizen:\nBefore we go"):
    """Run one sentence and draw every head's attention grid.

    Cell (i, j) is how much position i attended to position j. Only the lower
    triangle can be non zero, because of the causal mask.

    Each row is shaded against its own maximum, not against 1.0. A row sums to 1
    and is spread over i+1 cells, so later rows are diluted and would all look
    blank on an absolute scale. This is a relative scale within each row.
    """
    m, tok = load_model()
    ids = tok.encode(sentence)[:m.block_size]
    m.forward(ids)
    labels = [show_char(tok.itos[i]) for i in ids]

    lines = ["attention grids for: %r" % sentence,
             "",
             "cell (row i, col j) = how much position i attends to position j.",
             "shaded against each row's own maximum: '%s' is low, '%s' is high."
             % (SHADES[1], SHADES[-1]),
             "blank above the diagonal is the causal mask, exactly zero.",
             ""]

    for b_i, block in enumerate(m.blocks):
        for h_i, head in enumerate(block.attn.heads):
            lines.append("block %d, head %d" % (b_i, h_i))
            header = "        " + "".join("%-2s" % lab[:2] for lab in labels)
            lines.append(header)
            for i, row in enumerate(head.probs):
                mx = max(row[:i + 1])
                cells = []
                for j in range(len(row)):
                    if j > i:
                        cells.append("  ")
                    else:
                        level = int(row[j] / mx * (len(SHADES) - 1)) if mx > 0 else 0
                        cells.append(SHADES[level] * 2)
                lines.append("%-6s |%s" % (labels[i][:5], "".join(cells)))
            lines.append("")

    path = os.path.join(OUT, "08-attention-heads.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % path)
    print("\n".join(lines))


def cosine(u, v):
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def embedding_neighbours(probes=" etaoTqz.\n", k=6):
    """Nearest characters in embedding space, by cosine similarity.

    Nothing told the model which characters are related. If similar characters
    end up close together, that came purely from them being useful in the same
    places, which is what gradient descent does to the embedding table.
    """
    m, tok = load_model()
    table = m.tok_emb.weight.data

    lines = ["# Embedding neighbours after training",
             "",
             "Cosine similarity between rows of the embedding table. 1.0 means the same",
             "direction, 0.0 unrelated, -1.0 opposite. Nothing in the code groups characters,",
             "so any structure here was learned from the text.",
             ""]

    for ch in probes:
        if ch not in tok.stoi:
            continue
        i = tok.stoi[ch]
        sims = [(cosine(table[i], table[j]), tok.itos[j])
                for j in range(len(table)) if j != i]
        sims.sort(reverse=True)
        top = ", ".join("%s %.2f" % (show_char(c), s) for s, c in sims[:k])
        lines.append("- `%s` -> %s" % (show_char(ch), top))

    path = os.path.join(OUT, "09-embedding-neighbours.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s" % path)
    print("\n".join(lines))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "loss"):
        loss_curve()
        print()
    if what in ("all", "attn"):
        attention_maps()
        print()
    if what in ("all", "emb"):
        embedding_neighbours()
