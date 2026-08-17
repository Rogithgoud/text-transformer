"""Loading the corpus and cutting it into training windows.

A training example is a window of ids plus the same window shifted left by one:

    text     h  e  l  l  o
    input  [ h  e  l  l ]
    target    [ e  l  l  o ]        target[i] == input[i+1]

So one window of length T contains T separate prediction problems, one per
position, and the causal mask is what stops position i peeking at i+1.

The off-by-one: the target needs one character past the end of the input, so a
start index can only go up to len(data) - block_size - 1. Forgetting the -1
gives an IndexError, or worse, quietly wrong data.
"""


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_train_val(ids, val_fraction=0.1):
    """Last 10% is held out. Split by position, not at random, so the
    validation text is genuinely unseen rather than shuffled in."""
    n = int(len(ids) * (1.0 - val_fraction))
    return ids[:n], ids[n:]


def max_start(ids, block_size):
    """Largest legal start index for a window."""
    return len(ids) - block_size - 1


def get_window(ids, start, block_size):
    """One (input, target) pair of ids, each of length block_size."""
    if start < 0 or start > max_start(ids, block_size):
        raise IndexError("start %d out of range for block_size %d" % (start, block_size))
    x = ids[start:start + block_size]
    y = ids[start + 1:start + block_size + 1]
    return x, y


def get_batch(ids, block_size, batch_size, rng):
    """A list of `batch_size` (input, target) pairs at random positions.

    Returned as a plain list because the batch is handled by looping in
    train.py: run each sequence, add up the gradients, then average.
    """
    limit = max_start(ids, block_size)
    if limit < 0:
        raise ValueError("text is shorter than block_size + 1")
    return [get_window(ids, rng.randint(0, limit), block_size) for _ in range(batch_size)]
