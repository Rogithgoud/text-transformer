"""Causal self attention, forward and backward by hand.

This is the only place where positions look at each other. Everywhere else each
position is processed on its own.

Five steps in the forward pass:

    1. Q, K, V           three different projections of the same input
    2. scores = Q @ K^T  every query dotted with every key -> (T, T)
    3. / sqrt(head_size) keeps the scores small enough for softmax to stay soft
    4. mask              future cells set to -inf, so exp() makes them exactly 0
    5. softmax, then @ V each position walks away with a weighted blend of the
                         value vectors it was allowed to see

Why three projections and not one: the question a position asks (Q) is not the
same as the label it advertises (K), and neither is the same as the content it
hands over if picked (V).
"""

import math

import matrix as M
from layers import Linear, softmax_rows

NEG_INF = float("-inf")


class Head:
    """One attention head. (T, C) -> (T, head_size)."""

    def __init__(self, c, head_size, block_size, rng):
        self.q = Linear(c, head_size, rng)
        self.k = Linear(c, head_size, rng)
        self.v = Linear(c, head_size, rng)
        self.head_size = head_size
        self.block_size = block_size
        # 1/sqrt(head_size). Without it, a dot product over head_size channels
        # grows with head_size, softmax saturates to one-hot, and the gradient
        # through it goes to almost zero, so the layer stops learning.
        self.inv_sqrt = 1.0 / math.sqrt(head_size)
        self.probs = None
        self.Q = None
        self.K = None
        self.V = None

    def parameters(self):
        return self.q.parameters() + self.k.parameters() + self.v.parameters()

    def forward(self, x):
        t = len(x)
        Q = self.q.forward(x)
        K = self.k.forward(x)
        V = self.v.forward(x)

        # (T, hs) @ (hs, T) -> (T, T). scores[i][j] is "how much does position i
        # want position j".
        scores = M.matmul(Q, M.transpose(K))

        for i in range(t):
            row = scores[i]
            for j in range(t):
                if j > i:
                    row[j] = NEG_INF          # the future, blocked
                else:
                    row[j] = row[j] * self.inv_sqrt

        # softmax per row, so each row becomes weights that sum to 1.
        # exp(-inf) is 0, so masked positions get exactly zero weight. This is
        # why the mask has to be applied before the softmax and not after: after
        # it, the rows would no longer sum to 1.
        probs = softmax_rows(scores)

        self.Q, self.K, self.V, self.probs = Q, K, V, probs
        return M.matmul(probs, V)             # (T, T) @ (T, hs) -> (T, hs)

    def backward(self, dout):
        t = len(dout)

        # out = probs @ V
        dprobs = M.matmul(dout, M.transpose(self.V))       # (T, T)
        dV = M.matmul(M.transpose(self.probs), dout)        # (T, hs)

        # softmax backward, one row at a time:
        #   ds_j = p_j * (g_j - sum_k g_k p_k)
        # masked cells have p_j == 0, so their gradient is 0 automatically and
        # nothing leaks back through the future.
        dscores = M.zeros(t, t)
        for i in range(t):
            p = self.probs[i]
            g = dprobs[i]
            total = 0.0
            for j in range(t):
                total += g[j] * p[j]
            for j in range(i + 1):                          # only the past
                dscores[i][j] = p[j] * (g[j] - total) * self.inv_sqrt

        # scores = Q @ K^T
        dQ = M.matmul(dscores, self.K)                      # (T, hs)
        dK = M.matmul(M.transpose(dscores), self.Q)         # (T, hs)

        # x fed all three projections, so its gradient is the sum of all three
        dx = M.add(M.add(self.q.backward(dQ), self.k.backward(dK)), self.v.backward(dV))
        return dx


class MultiHeadAttention:
    """n_heads heads side by side, their outputs concatenated and projected back
    to C. (T, C) -> (T, C).

    d_model is split between the heads, so the total width is unchanged. The
    point is that each head gets its own Q/K/V and can therefore look for a
    different kind of relationship (previous character, spaces, start of line),
    which one single head of the same total width cannot do, because it only
    produces one attention pattern.
    """

    def __init__(self, c, n_heads, block_size, rng):
        if c % n_heads != 0:
            raise ValueError("d_model %d must divide by n_heads %d" % (c, n_heads))
        self.head_size = c // n_heads
        self.heads = [Head(c, self.head_size, block_size, rng) for _ in range(n_heads)]
        self.proj = Linear(c, c, rng)

    def parameters(self):
        ps = []
        for h in self.heads:
            ps += h.parameters()
        return ps + self.proj.parameters()

    def forward(self, x):
        outs = [h.forward(x) for h in self.heads]
        cat = [sum((o[i] for o in outs), []) for i in range(len(x))]   # (T, C)
        return self.proj.forward(cat)

    def backward(self, dout):
        dcat = self.proj.backward(dout)                                # (T, C)
        hs = self.head_size
        dx = None
        for n, h in enumerate(self.heads):
            piece = [row[n * hs:(n + 1) * hs] for row in dcat]         # this head's slice
            dxh = h.backward(piece)
            dx = dxh if dx is None else M.add(dx, dxh)
        return dx
