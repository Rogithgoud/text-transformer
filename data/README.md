# data

The corpus itself is **not committed** (see `.gitignore`) — only its provenance and statistics
are recorded here, so the experiments stay reproducible.

## Corpus

| Field | Value |
|---|---|
| Name |  |
| Source URL |  |
| Licence / public-domain status |  |
| Downloaded on |  |
| File size |  |
| Character count |  |
| Unique characters (vocab size) |  |
| Train / validation split |  |

## Requirements for the corpus

- Plain text, UTF-8, no markup.
- Public domain or explicitly permissive licence — record which, and where it says so.
- Roughly 100 KB–1 MB. Larger buys nothing here: pure Python training speed, not data, is the
  binding constraint.
- Consistent style, so a tiny model has a chance of learning something visible.

Tiny Shakespeare is the conventional choice for exactly this reason.

## Character-frequency table

Fill in during Phase 4 — it explains a lot about the model's early behaviour, since the first
thing any language model learns is the base frequency of each character.

| Char | Count | Frequency | Notes |
|---|---|---|---|
|  |  |  |  |

## Preprocessing decisions (record each, with the reason)

- [ ] Lowercasing? yes / no — reason:
- [ ] Strip or keep newlines? — reason:
- [ ] Collapse repeated whitespace? — reason:
- [ ] Any characters removed from the vocabulary? — reason:

Each decision changes the vocabulary size, which changes the expected initial loss
(`ln(vocab_size)`). Note the resulting number here so the Phase 6 sanity check has something to
compare against.
