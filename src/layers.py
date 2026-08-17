"""The basic layers, each with its forward and its backward written by hand.

There is no autograd engine here. Every layer stores what it needs from the
forward pass, and its backward(dout) does two things:

  1. adds to the gradients of its own parameters
  2. returns the gradient with respect to its input, so the layer before it can
     carry on

That is the chain rule, done locally. A layer never needs to know anything about
the rest of the network, only how to convert an incoming gradient into gradients
for the things it owns.

Shapes: x is (T, C_in) and the output is (T, C_out). T is the number of
positions. Batches are a loop in train.py, so there is no batch dimension.
"""

import math

import matrix as M


class Param:
    """One learnable matrix plus the gradient collected for it.

    grad is accumulated (+=) rather than overwritten, because a parameter can be
    used more than once in a pass, and because a batch here is several sequences
    run one after another. train.py zeroes the grads at the start of every step,
    which is what zero_grad means in a framework.
    """

    def __init__(self, data):
        self.data = data
        self.grad = M.zeros(len(data), len(data[0]))

    def zero_grad(self):
        for row in self.grad:
            for j in range(len(row)):
                row[j] = 0.0

    def shape(self):
        return (len(self.data), len(self.data[0]))


def softmax_rows(x):
    """Softmax along each row. (T, C) -> (T, C), every row summing to 1.

    The max is subtracted first. exp(700) already overflows a float, and
    attention scores get large, so without this the whole thing dies with
    "math range error". Subtracting a constant from a row does not change the
    result, because exp(a-m)/sum(exp(a-m)) cancels the exp(-m) top and bottom.
    """
    out = []
    for row in x:
        m = max(row)
        exps = [math.exp(v - m) for v in row]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out


def relu(x):
    """max(0, v) elementwise. The nonlinearity in the feed-forward layer.

    Without something like this, two stacked linear layers collapse into one
    single linear layer, and the extra depth buys nothing.
    """
    return [[v if v > 0.0 else 0.0 for v in row] for row in x]


def relu_backward(dout, x):
    """relu passes the gradient through where the input was positive, and blocks
    it where the input was negative (that side of the function is flat, slope 0).
    """
    return [[dout[i][j] if x[i][j] > 0.0 else 0.0 for j in range(len(x[0]))]
            for i in range(len(x))]


class Linear:
    """out = x @ W + b

    W is (C_in, C_out), b is a vector of length C_out.

    Backward, derived by writing one output cell as a sum and differentiating:
        dx = dout @ W^T          (T, C_in)
        dW = x^T @ dout          (C_in, C_out)
        db = sum of dout down the columns
    A quick way to remember it: the shapes only fit one way round.
    """

    def __init__(self, c_in, c_out, rng, scale=None):
        # 1/sqrt(fan_in) keeps the size of the output roughly the size of the
        # input, instead of growing with C_in
        if scale is None:
            scale = 1.0 / math.sqrt(c_in)
        self.W = Param(M.randn(c_in, c_out, scale, rng))
        self.b = Param([[0.0] * c_out])
        self.x = None

    def parameters(self):
        return [self.W, self.b]

    def forward(self, x):
        self.x = x
        return M.add_row(M.matmul(x, self.W.data), self.b.data[0])

    def backward(self, dout):
        dW = M.matmul(M.transpose(self.x), dout)
        for i in range(len(dW)):
            for j in range(len(dW[0])):
                self.W.grad[i][j] += dW[i][j]

        db = M.col_sum(dout)
        for j in range(len(db)):
            self.b.grad[0][j] += db[j]

        return M.matmul(dout, M.transpose(self.W.data))


class LayerNorm:
    """Normalise each position's own C numbers to mean 0, spread 1, then apply a
    learnable scale (gamma) and shift (beta).

    Note what it normalises over: one position's channels, not across positions.
    That is why it works fine with any sequence length and does not mix
    information between positions.

        mu   = mean(x)
        var  = mean((x - mu)^2)
        xhat = (x - mu) / sqrt(var + eps)
        out  = gamma * xhat + beta

    Backward for one row (the messy one, because mu and var both depend on every
    element of the row):

        dxhat = dout * gamma
        dx    = (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat)) / sqrt(var+eps)

    The two mean terms are the parts that flow back through mu and var.
    """

    def __init__(self, c, eps=1e-5):
        self.gamma = Param([[1.0] * c])
        self.beta = Param([[0.0] * c])
        self.eps = eps
        self.xhat = None
        self.inv_std = None

    def parameters(self):
        return [self.gamma, self.beta]

    def forward(self, x):
        c = len(x[0])
        xhat = []
        inv_std = []
        out = []
        g = self.gamma.data[0]
        b = self.beta.data[0]
        for row in x:
            mu = sum(row) / c
            var = sum((v - mu) ** 2 for v in row) / c
            inv = 1.0 / math.sqrt(var + self.eps)
            xh = [(v - mu) * inv for v in row]
            xhat.append(xh)
            inv_std.append(inv)
            out.append([g[j] * xh[j] + b[j] for j in range(c)])
        self.xhat = xhat
        self.inv_std = inv_std
        return out

    def backward(self, dout):
        c = len(dout[0])
        g = self.gamma.data[0]
        dx = []
        for i in range(len(dout)):
            xh = self.xhat[i]
            drow = dout[i]

            for j in range(c):
                self.gamma.grad[0][j] += drow[j] * xh[j]
                self.beta.grad[0][j] += drow[j]

            dxhat = [drow[j] * g[j] for j in range(c)]
            mean_dxhat = sum(dxhat) / c
            mean_dxhat_xhat = sum(dxhat[j] * xh[j] for j in range(c)) / c
            inv = self.inv_std[i]
            dx.append([(dxhat[j] - mean_dxhat - xh[j] * mean_dxhat_xhat) * inv
                       for j in range(c)])
        return dx


class Embedding:
    """A lookup table of shape (vocab, C). ids -> (T, C).

    Forward is just picking rows, so backward is just putting the gradient back
    on the rows that were picked. If the same character appears twice in the
    window, its row gets both gradients added, which is correct: that row was
    used twice.
    """

    def __init__(self, vocab, c, rng, scale=0.02):
        self.weight = Param(M.randn(vocab, c, scale, rng))
        self.ids = None

    def parameters(self):
        return [self.weight]

    def forward(self, ids):
        self.ids = ids
        return [list(self.weight.data[i]) for i in ids]

    def backward(self, dout):
        for pos, i in enumerate(self.ids):
            row = self.weight.grad[i]
            drow = dout[pos]
            for j in range(len(drow)):
                row[j] += drow[j]
        return None  # ids are integers, there is nothing before this to update


def positional_encoding(block_size, c):
    """Fixed sin/cos table, shape (block_size, C). No parameters to learn.

    Position p, channel pair (2i, 2i+1):
        even channel: sin(p / 10000^(2i/C))
        odd  channel: cos(p / 10000^(2i/C))

    Each channel is a wave, and the waves get slower as the channel index goes
    up: the fast ones tell positions apart locally, the slow ones give a rough
    sense of where you are in the whole window. Every position ends up with its
    own fingerprint, and since sin and cos of a shifted angle can be written
    from the unshifted ones, relative distance is recoverable from these too.

    Without this the model would see a bag of characters with no order, because
    attention only looks at similarity between pairs of vectors.
    """
    table = M.zeros(block_size, c)
    for p in range(block_size):
        for i in range(0, c, 2):
            freq = 1.0 / (10000.0 ** (i / c))
            table[p][i] = math.sin(p * freq)
            if i + 1 < c:
                table[p][i + 1] = math.cos(p * freq)
    return table
