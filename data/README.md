# data

The text file itself is not committed, only where it came from.

Tiny Shakespeare, from
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Downloaded 2026-08-18. Shakespeare is public domain.

1,115,394 characters, 65 unique ones:

```
\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
```

Last 10% held back for validation, split by position rather than shuffled so the held out text is
genuinely unseen. That gives 1,003,854 for training and 111,540 for validation.

`ln(65) = 4.1744`, which is the loss a model gets for knowing nothing and spreading its guess evenly
over all 65. First thing I check on any run.

## The frequencies matter

Space is 15.23% of the whole file, then `e` at 8.48%, `t` 6.01%, `o` 5.90%, `a` 4.98%, and newline
is 3.59%.

This explains the first part of the loss curve. The very first thing the model learns is not
spelling, it is these base rates. Guess a space 15% of the time and an `e` 8% of the time and the
loss already drops a long way below 4.17 without understanding anything. Worth knowing so I don't
read that early drop as the model being clever.

## What I did not do to it

No lowercasing, because the capitals carry real structure here (`FIRST CITIZEN:`) and that is one of
the few patterns short enough for a 32 character context to pick up. Kept the newlines, they are
just another character. Did not collapse whitespace and did not drop anything, so the file and my
ids match one to one and `decode(encode(text))` comes back exactly.
