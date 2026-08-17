"""The tokenizer only has to satisfy one thing: decode(encode(s)) == s."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tokenizer import Tokenizer


def test_hello_vocab():
    t = Tokenizer.from_text("hello")
    # sorted unique chars of "hello" are e, h, l, o
    assert t.chars == ["e", "h", "l", "o"], t.chars
    assert t.vocab_size == 4
    assert t.encode("hello") == [1, 0, 2, 2, 3], t.encode("hello")
    print("hello vocab and ids match what I worked out by hand")


def test_round_trip():
    text = "To be, or not to be:\nthat is the question.\n"
    t = Tokenizer.from_text(text)
    assert t.decode(t.encode(text)) == text
    print("round trip holds on a longer string")


def test_ids_are_stable():
    # sorted() not set(), so two tokenizers built from the same text agree
    text = "abcabc\n xyz"
    assert Tokenizer.from_text(text).stoi == Tokenizer.from_text(text).stoi
    print("ids are reproducible between runs")


if __name__ == "__main__":
    test_hello_vocab()
    test_round_trip()
    test_ids_are_stable()
    print("all tokenizer tests passed")
