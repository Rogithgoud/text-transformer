"""The off-by-one test. target[i] must be exactly input[i+1], and a window must
never be allowed to run off the end of the text."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import dataset as D
from tokenizer import Tokenizer


def test_hello_window():
    t = Tokenizer.from_text("hello")
    ids = t.encode("hello")          # [1, 0, 2, 2, 3]
    x, y = D.get_window(ids, 0, 4)
    assert x == [1, 0, 2, 2], x      # "hell"
    assert y == [0, 2, 2, 3], y      # "ello"
    print("hello window matches the table in the research notes")


def test_target_is_input_shifted():
    ids = list(range(50))
    for start in range(0, D.max_start(ids, 8) + 1):
        x, y = D.get_window(ids, start, 8)
        for i in range(len(x) - 1):
            assert y[i] == x[i + 1], (start, i, x, y)
        assert y[-1] == ids[start + 8]
    print("target[i] == input[i+1] at every legal start")


def test_last_legal_start():
    # 5 characters, block_size 4 -> only start 0 is legal
    ids = [1, 0, 2, 2, 3]
    assert D.max_start(ids, 4) == 0, D.max_start(ids, 4)
    D.get_window(ids, 0, 4)
    try:
        D.get_window(ids, 1, 4)
    except IndexError:
        print("start past the limit raises instead of returning short data")
        return
    raise AssertionError("expected an IndexError")


def test_batch_shapes():
    rng = random.Random(1234)
    ids = list(range(200))
    batch = D.get_batch(ids, block_size=16, batch_size=4, rng=rng)
    assert len(batch) == 4
    for x, y in batch:
        assert len(x) == 16 and len(y) == 16
    print("batch is 4 pairs of length 16")


def test_split():
    tr, va = D.split_train_val(list(range(100)), 0.1)
    assert len(tr) == 90 and len(va) == 10
    assert va[0] == 90  # split by position, not shuffled
    print("train/val split is by position")


if __name__ == "__main__":
    test_hello_window()
    test_target_is_input_shifted()
    test_last_legal_start()
    test_batch_shapes()
    test_split()
    print("all dataset tests passed")
