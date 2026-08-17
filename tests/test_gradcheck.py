"""Gradient check. The one test that decides whether the hand-written backward
passes are actually right.

Idea: the gradient is by definition the slope of the loss with respect to one
weight. So nudge that weight by a tiny h, measure how much the loss moved, and
compare with what backward() claimed.

    numerical = (loss(w + h) - loss(w - h)) / (2h)

The two-sided version is used because its error shrinks like h^2 instead of h.

h is 1e-5, which is a compromise. Too large and the slope measurement is wrong
because the function curves. Too small and the two losses are so close that
subtracting them loses most of the significant digits (catastrophic
cancellation), so the answer is dominated by float noise.

Relative error is what gets compared, not the absolute difference, because the
gradients vary in size by orders of magnitude across the model.

One catch, see docs/03-errors/001: when both gradients are essentially zero the
relative error is meaningless, because dividing float noise by float noise gives
anything. Those cells are checked on absolute difference instead.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loss import cross_entropy
from model import Model

H = 1e-5
TOL = 1e-4
# below this magnitude a gradient is float noise, so compare absolutely, not
# relatively
TINY = 1e-6


def build():
    rng = random.Random(0)
    m = Model(vocab_size=7, block_size=6, d_model=8, n_heads=2, n_layers=2, rng=rng)
    ids = [1, 3, 0, 6, 2, 4]
    targets = [3, 0, 6, 2, 4, 1]
    return m, ids, targets


def loss_of(m, ids, targets):
    logits = m.forward(ids)
    loss, _ = cross_entropy(logits, targets)
    return loss


def test_gradcheck():
    m, ids, targets = build()

    # analytic gradients
    m.zero_grad()
    logits = m.forward(ids)
    loss, dlogits = cross_entropy(logits, targets)
    m.backward(dlogits)

    params = m.parameters()
    print("model has %d parameter matrices, %d numbers" % (len(params), m.num_params()))
    print("loss = %.6f" % loss)

    rng = random.Random(7)
    worst = 0.0
    worst_where = None
    checked = 0
    skipped = 0

    # checking every weight would mean two forward passes each, so a random
    # sample of cells from every matrix is used instead
    for n, p in enumerate(params):
        rows = len(p.data)
        cols = len(p.data[0])
        for _ in range(3):
            i = rng.randrange(rows)
            j = rng.randrange(cols)

            original = p.data[i][j]
            p.data[i][j] = original + H
            plus = loss_of(m, ids, targets)
            p.data[i][j] = original - H
            minus = loss_of(m, ids, targets)
            p.data[i][j] = original

            numerical = (plus - minus) / (2.0 * H)
            analytic = p.grad[i][j]

            if max(abs(numerical), abs(analytic)) < TINY:
                # both are zero for real (an unused embedding row, say). the
                # relative error here would just be noise over noise
                assert abs(numerical - analytic) < TINY, (n, i, j, numerical, analytic)
                skipped += 1
                continue

            rel = abs(numerical - analytic) / (abs(numerical) + abs(analytic))
            checked += 1
            if rel > worst:
                worst = rel
                worst_where = (n, i, j, numerical, analytic)

    print("checked %d cells (%d skipped as zero), worst relative error %.3e"
          % (checked, skipped, worst))
    if worst_where:
        n, i, j, num, ana = worst_where
        print("  worst was param %d cell (%d,%d): numerical %.8f vs analytic %.8f"
              % (n, i, j, num, ana))
    assert worst < TOL, "gradient check failed, worst relative error %.3e" % worst
    print("gradient check passed")


def test_initial_loss_is_about_log_vocab():
    import math
    rng = random.Random(3)
    m = Model(vocab_size=65, block_size=16, d_model=16, n_heads=2, n_layers=2, rng=rng)
    ids = [rng.randrange(65) for _ in range(16)]
    targets = [rng.randrange(65) for _ in range(16)]
    loss, _ = cross_entropy(m.forward(ids), targets)
    expected = math.log(65)
    print("initial loss %.4f, expected about %.4f" % (loss, expected))
    assert abs(loss - expected) < 0.5, (loss, expected)
    print("untrained model is uniformly clueless, as it should be")


if __name__ == "__main__":
    test_gradcheck()
    print()
    test_initial_loss_is_about_log_vocab()
    print("\nall gradcheck tests passed")
