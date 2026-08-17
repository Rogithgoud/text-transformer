"""SGD and Adam, both by hand.

SGD is the whole idea in one line: the gradient says which way the loss goes up,
so step the other way.

    w = w - lr * grad

Adam adds two running averages per weight, and it is worth knowing why:

    m = b1*m + (1-b1)*grad          average of the gradient (momentum)
    v = b2*v + (1-b2)*grad^2        average of the gradient squared

  - m smooths the direction, so a noisy gradient does not throw the step around.
  - v tracks how big this weight's gradients usually are. Dividing by sqrt(v)
    gives every weight its own effective step size: weights with consistently
    huge gradients take small steps, weights with tiny gradients take bigger
    ones. That is why Adam works with almost no tuning, which matters here
    because I only get a handful of runs at this speed.
  - eps stops a division by zero for weights whose gradient is always ~0.

Bias correction: m and v both start at 0, so early on they are biased towards 0
and the first steps would be far too small. Dividing by (1 - b1^t) and
(1 - b2^t) cancels exactly that, and both factors go to 1 as t grows.
"""

import math


class SGD:
    def __init__(self, params, lr=0.1):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            for i in range(len(p.data)):
                row = p.data[i]
                grow = p.grad[i]
                for j in range(len(row)):
                    row[j] -= self.lr * grow[j]


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m = [[[0.0] * len(p.data[0]) for _ in range(len(p.data))] for p in params]
        self.v = [[[0.0] * len(p.data[0]) for _ in range(len(p.data))] for p in params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t          # bias correction for m
        bc2 = 1.0 - self.b2 ** self.t          # bias correction for v
        for n, p in enumerate(self.params):
            m = self.m[n]
            v = self.v[n]
            for i in range(len(p.data)):
                row = p.data[i]
                grow = p.grad[i]
                mrow = m[i]
                vrow = v[i]
                for j in range(len(row)):
                    g = grow[j]
                    mrow[j] = self.b1 * mrow[j] + (1.0 - self.b1) * g
                    vrow[j] = self.b2 * vrow[j] + (1.0 - self.b2) * g * g
                    mhat = mrow[j] / bc1
                    vhat = vrow[j] / bc2
                    row[j] -= self.lr * mhat / (math.sqrt(vhat) + self.eps)
