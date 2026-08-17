"""The most important test in the project.

A causal model must not let information flow backwards in time. If it does, the
model is reading the answer, and the loss looks suspiciously good while the whole
thing is meaningless. It does not crash and it does not look wrong, which is
exactly why it needs a test.

The check: run a sequence, then change a token at a LATER position. Every output
at or before the changed position must be bit for bit identical.
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from model import Model


def test_future_cannot_affect_the_past():
    rng = random.Random(11)
    m = Model(vocab_size=9, block_size=8, d_model=8, n_heads=2, n_layers=2, rng=rng)

    ids = [1, 4, 2, 7, 0, 3, 5, 6]
    base = m.forward(ids)

    for change_at in range(1, len(ids)):
        other = list(ids)
        other[change_at] = (other[change_at] + 1) % 9    # different character
        new = m.forward(other)

        for pos in range(change_at):                     # strictly before it
            assert base[pos] == new[pos], (
                "changing position %d changed the output at position %d, "
                "the future is leaking into the past" % (change_at, pos))

        # and the sanity half: the changed position itself MUST differ, otherwise
        # the test would pass on a model that ignores its input entirely
        assert base[change_at] != new[change_at], (
            "changing position %d did not change its own output" % change_at)

    print("no position can see the future, and every position sees itself")


def test_attention_rows_sum_to_one():
    rng = random.Random(5)
    m = Model(vocab_size=9, block_size=8, d_model=8, n_heads=2, n_layers=2, rng=rng)
    m.forward([1, 4, 2, 7, 0, 3, 5, 6])

    for b_i, b in enumerate(m.blocks):
        for h_i, h in enumerate(b.attn.heads):
            for i, row in enumerate(h.probs):
                total = sum(row)
                assert abs(total - 1.0) < 1e-9, (b_i, h_i, i, total)
                # everything past the diagonal must be exactly zero
                for j in range(i + 1, len(row)):
                    assert row[j] == 0.0, (b_i, h_i, i, j, row[j])
    print("every attention row sums to 1 and the upper triangle is exactly 0")


if __name__ == "__main__":
    test_future_cannot_affect_the_past()
    test_attention_rows_sum_to_one()
    print("all causal mask tests passed")
