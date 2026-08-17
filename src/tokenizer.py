"""Character level tokenizer.

One token = one character. No BPE, no library. The vocabulary is just every
unique character in the training text, sorted so the ids are reproducible
between runs (a set has no fixed order, sorted() does).

Why character level: the vocab stays around 65 entries, so the embedding table
and the output head stay tiny, which matters a lot at pure-Python speed. The
cost is that the model has to learn spelling from scratch, and 32 characters of
context is only a few words.
"""

import json


class Tokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char -> id
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # id -> char

    @property
    def vocab_size(self):
        return len(self.chars)

    @classmethod
    def from_text(cls, text):
        """Build the vocabulary from the training text."""
        return cls(sorted(set(text)))

    def encode(self, text):
        """str -> list of ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """list of ids -> str."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"chars": self.chars}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f)["chars"])
