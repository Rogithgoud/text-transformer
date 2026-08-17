# data

The corpus itself is not committed (see `.gitignore`), only where it came from and its
statistics, so the experiments stay reproducible.

## Corpus

| Field | Value |
|---|---|
| Name | Tiny Shakespeare (a concatenation of Shakespeare's plays) |
| Source URL | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt |
| Licence | Shakespeare's text is public domain (author died 1616) |
| Downloaded on | 2026-08-18 |
| File size | 1,115,394 bytes |
| Character count | 1,115,394 |
| Unique characters (vocab size) | 65 |
| Train / validation split | last 10% held out by position, so 1,003,854 / 111,540 |

`ln(65) = 4.1744`. That is the loss an untrained model should start at, because a model that
knows nothing spreads its probability evenly over all 65 characters and so gives the correct one
about 1/65. Every run is checked against this number before anything else.

## Why this corpus

- Plain UTF-8 text, no markup.
- Public domain, and it's the standard choice for character level models, so my loss numbers can
  be compared against what other people get.
- Consistent style and a lot of structure a tiny model can actually pick up: names in capitals
  followed by a colon, line breaks, short speeches.
- Size is not the constraint here. At pure Python speed I'll only get through a small fraction of
  it anyway, so a bigger corpus would buy nothing.

## Character frequency

The top of the table, from `collections.Counter`:

| Char | Count | Frequency |
|---|---|---|
| (space) | 169,892 | 15.23% |
| e | 94,611 | 8.48% |
| t | 67,009 | 6.01% |
| o | 65,798 | 5.90% |
| a | 55,507 | 4.98% |
| h | 51,310 | 4.60% |
| s | 49,696 | 4.46% |
| r | 48,889 | 4.38% |
| n | 48,529 | 4.35% |
| i | 45,537 | 4.08% |
| (newline) | 40,000 | 3.59% |
| l | 33,339 | 2.99% |

Full vocabulary (65 characters):

```
\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

This table explains the first part of the loss curve. The very first thing the model learns is
not spelling, it's these base frequencies: guess a space 15% of the time and an 'e' 8% of the
time and the loss already drops a long way below `ln(65)`, without understanding anything.

Worth knowing so I don't read the early drop as the model being clever.

## Preprocessing decisions

- Lowercasing? **No.** The capital letters carry real structure here (`FIRST CITIZEN:`), and
  dropping them would throw away one of the few patterns short enough for a 32 character context
  to learn.
- Strip or keep newlines? **Keep.** `\n` is 3.59% of the text and it marks the line structure. It
  is just another character in the vocabulary.
- Collapse repeated whitespace? **No.** Leaving it alone means the raw file and my ids match one
  to one, so nothing needs undoing when I decode.
- Any characters removed? **No.** All 65 stay, which keeps `decode(encode(text)) == text` exactly
  true, and that's the tokenizer's test.
