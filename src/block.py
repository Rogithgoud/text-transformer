"""One transformer block: attention, then a feed-forward layer, each wrapped in a
layernorm and a residual connection.

    h = x + attention(norm(x))          positions talk to each other
    y = h + feedforward(norm(h))        each position thinks on its own

This is pre-norm (the norm sits inside the branch, before the sublayer) rather
than post-norm as in the original paper. Pre-norm trains more stably, because the
residual path from input to output stays clean, with nothing rescaling it.

Why the residual at all:
  - the block only has to learn a change to x, not rebuild x from scratch
  - on the way back, d(x + f(x))/dx = 1 + df/dx, so that 1 sends the gradient
    straight through to the input. That is the highway that makes depth trainable.
"""

import matrix as M
from attention import MultiHeadAttention
from layers import LayerNorm, Linear, relu, relu_backward


class FeedForward:
    """C -> 4C -> relu -> C, applied to each position separately.

    The relu in the middle is what stops the two linears collapsing into one
    (B(Ax) = (BA)x, which would be a single linear layer). The 4x widening gives
    room to compute something before squashing back down to C.
    """

    def __init__(self, c, rng, mult=4):
        self.fc1 = Linear(c, mult * c, rng)
        self.fc2 = Linear(mult * c, c, rng)
        self.pre_relu = None

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def forward(self, x):
        self.pre_relu = self.fc1.forward(x)
        return self.fc2.forward(relu(self.pre_relu))

    def backward(self, dout):
        d = self.fc2.backward(dout)
        d = relu_backward(d, self.pre_relu)
        return self.fc1.backward(d)


class Block:
    def __init__(self, c, n_heads, block_size, rng):
        self.ln1 = LayerNorm(c)
        self.attn = MultiHeadAttention(c, n_heads, block_size, rng)
        self.ln2 = LayerNorm(c)
        self.ff = FeedForward(c, rng)

    def parameters(self):
        return (self.ln1.parameters() + self.attn.parameters()
                + self.ln2.parameters() + self.ff.parameters())

    def forward(self, x):
        h = M.add(x, self.attn.forward(self.ln1.forward(x)))
        y = M.add(h, self.ff.forward(self.ln2.forward(h)))
        return y

    def backward(self, dy):
        # y = h + ff(ln2(h)): the gradient reaches h down both paths, so add them
        dh = M.add(dy, self.ln2.backward(self.ff.backward(dy)))
        # h = x + attn(ln1(x)): same again
        dx = M.add(dh, self.ln1.backward(self.attn.backward(dh)))
        return dx
