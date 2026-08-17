"""The whole model, assembled.

    ids (T,)
      -> embedding                (T, C)   each character becomes C numbers
      -> + positional encoding    (T, C)   now they know where they are
      -> block 1 .. block N       (T, C)   the residual stream, width never changes
      -> final layernorm          (T, C)
      -> output head              (T, V)   V logits per position
"""

import json

import matrix as M
from block import Block
from layers import Embedding, LayerNorm, Linear, positional_encoding


class Model:
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers, rng):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.tok_emb = Embedding(vocab_size, d_model, rng)
        self.pos = positional_encoding(block_size, d_model)   # fixed, not learned
        self.blocks = [Block(d_model, n_heads, block_size, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size, rng)

    def parameters(self):
        ps = self.tok_emb.parameters()
        for b in self.blocks:
            ps += b.parameters()
        return ps + self.ln_f.parameters() + self.head.parameters()

    def num_params(self):
        return sum(len(p.data) * len(p.data[0]) for p in self.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, ids):
        """ids is a list of length T (T <= block_size). Returns logits (T, V)."""
        t = len(ids)
        if t > self.block_size:
            raise ValueError("sequence of %d is longer than block_size %d" % (t, self.block_size))
        x = self.tok_emb.forward(ids)
        x = M.add(x, [self.pos[i] for i in range(t)])
        for b in self.blocks:
            x = b.forward(x)
        x = self.ln_f.forward(x)
        return self.head.forward(x)

    def backward(self, dlogits):
        """dlogits is (T, V), the gradient of the loss w.r.t. the logits."""
        d = self.head.backward(dlogits)
        d = self.ln_f.backward(d)
        for b in reversed(self.blocks):          # reverse order: last layer first
            d = b.backward(d)
        # the positional table is fixed, so its branch of the add needs no update
        self.tok_emb.backward(d)

    def config(self):
        return {"vocab_size": self.vocab_size, "block_size": self.block_size,
                "d_model": self.d_model, "n_heads": self.n_heads,
                "n_layers": self.n_layers}

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"config": self.config(),
                       "params": [p.data for p in self.parameters()]}, f)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        saved = blob["params"]
        mine = self.parameters()
        if len(saved) != len(mine):
            raise ValueError("checkpoint has %d parameter matrices, model has %d"
                             % (len(saved), len(mine)))
        for p, data in zip(mine, saved):
            if len(data) != len(p.data) or len(data[0]) != len(p.data[0]):
                raise ValueError("checkpoint shape does not match the model")
            p.data = data
