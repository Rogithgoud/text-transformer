"""Cross entropy loss, and the one gradient in this project that comes out clean.

For one position: softmax the V logits, then take the negative log of the
probability the model gave to the correct character.

    p = softmax(logits)
    loss = -log(p[target])

Read it as surprise. If the model gave the right character probability 1, the
loss is -log(1) = 0, no surprise. If it gave it 0.01, the loss is 4.6. The log is
what makes being confidently wrong expensive.

Sanity check used all over this project: an untrained model spreads probability
evenly, so p[target] is about 1/V and the loss is about log(V). For V = 65 that
is about 4.17. If the first loss is not near that, something is broken before
training even starts.

The gradient, if you push the derivative of softmax through the derivative of the
log, collapses to:

    dlogits = p - onehot(target)

That is all. "Predicted minus actual". Every wrong character gets pushed down by
exactly how much probability it claimed, and the right one gets pushed up by how
much it was missing. Derived by hand in docs/01-math/.
"""

import math

from layers import softmax_rows


def cross_entropy(logits, targets):
    """logits (T, V), targets a list of T ids.

    Returns (mean loss over the T positions, dlogits (T, V)).
    The mean, not the sum, so the number does not change meaning when T changes.
    """
    t = len(logits)
    probs = softmax_rows(logits)

    total = 0.0
    for i in range(t):
        p = probs[i][targets[i]]
        # clamp: log(0) is -inf, and a probability can underflow to 0.0
        total += -math.log(p if p > 1e-12 else 1e-12)

    dlogits = []
    for i in range(t):
        row = list(probs[i])
        row[targets[i]] -= 1.0                 # p - onehot
        dlogits.append([v / t for v in row])   # /t because the loss was a mean

    return total / t, dlogits
