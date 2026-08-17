# Research Notes

Everything here is in my own words. No copy-paste from papers or blogs. If I can't restate
something myself then it doesn't go here, it goes in [00-open-questions.md](00-open-questions.md)
until I can.

---

## Project rules

Writing these down first so the whole project has one spec and I don't drift:

- Standard library only: `math`, `random`, `json`, `time`, `os`. Nothing else.
- No NumPy, no PyTorch, no TensorFlow, no JAX, no SciPy. I write matmul, softmax, backprop and
  the optimizer myself.
- Writing my own `exp` and `sqrt` (Taylor series / Newton's method) is a bonus in Phase 2. It is
  not a blocker. If it slows me down I use `math` and note it.
- No GPU. Not a choice, it's forced. A GPU is only reachable through CUDA, and CUDA is only
  reachable through the libraries I'm not allowed to use. So single CPU core. The model has to be
  tiny: character level, 2 layers, around 40k parameters, ~100 KB of text.
- Understanding is the goal here, not sample quality. Noting it so I don't waste weeks chasing
  better output.

---

## Reading list and status

| # | Source | Status | Notes file / section below |
|---|---|---|---|
| 1 | Language modelling as next-token prediction | done 2026-08-17 | §1 |
| 2 | *Attention Is All You Need* (Vaswani et al., 2017) — pass 1 (shape) | not read | §2 |
| 2 | pass 2 (equations) | not read | §2 |
| 2 | pass 3 (details I skipped) | not read | §2 |
| 3 | Karpathy — "Let's build GPT" / nanoGPT / micrograd (concept only) | not read | §3 |
| 4 | Alammar — *The Illustrated Transformer* | not read | §4 |
| 5 | Backprop as reverse-mode autodiff | not read | §5 |

---

## §1 — What a language model actually is

### The task, stated as probability

The model estimates exactly one thing:

```
P( next character | all the characters that came before it )
```

The `|` means "given". So it's the probability of the next character given the past.

At first I didn't see how that one small question covers all of language. It does, because of the
chain rule of probability. If I want the probability of a whole string like "hello", I can split
it into a product of next-character questions:

```
P("hello") = P(h)
           x P(e | h)
           x P(l | h,e)
           x P(l | h,e,l)
           x P(o | h,e,l,l)
```

Every line is the same type of question. So one model that answers "what comes next" can score
any text at all. That's the whole trick.

"Given all previous" also means each position is only allowed to look backwards:

```
position:   0    1    2    3    4
char:       h    e    l    l    o

to predict position 4 ('o') I may look at:  h  e  l  l
                                            ^^^^^^^^^^  past, allowed
                                                      o   <- not allowed
```

This is where the causal mask comes from later. It isn't a design choice someone made, it falls
straight out of the definition. If the model could see the future it would just copy the answer
and learn nothing, and then at generation time it would be useless because the future doesn't
exist yet.

### Why "next character" is enough to produce whole paragraphs

The model only ever predicts one step. Long text comes from a loop that feeds the model's own
output back in as input. That's what autoregressive means: auto (self) + regress (feed back).

```
  start with prompt = "h"

     "h"      -> model -> probabilities -> pick 'e'
     "he"     -> model -> probabilities -> pick 'l'
     "hel"    -> model -> probabilities -> pick 'l'
     "hell"   -> model -> probabilities -> pick 'o'
     "hello"  -> model -> probabilities -> pick ' '
        ^                                    |
        |________ append it and repeat ______|

  run the loop 500 times = 500 characters of text
```

Three things follow from this loop, and I think all three will show up in my results:

1. The input keeps growing, which is why there's a maximum context (`block_size`). Once the text
   is longer than that I have to drop the oldest characters. My model will remember 32 characters
   and nothing before that. It literally cannot remember the start of a long paragraph.
2. Mistakes compound. A bad character becomes part of the input for every step after it. That's
   why a weak model turns into nonsense partway through instead of staying just slightly bad.
3. Training and generation are not the same situation. In training the model always gets real text
   as input. In generation it gets its own output. Nobody fixes that in a basic setup, so it goes
   in the README as a known limitation.

### What the model outputs at each position, and its shape

Vocabulary = every unique character in my corpus. Call the count `V` (around 65 for Shakespeare).

At each position the model outputs exactly `V` numbers, one score per possible next character.

```
input: "hell"        output at the last position is V numbers:

              char:    a     b     c    ...   o   ...   z
             score:  -1.2   0.4  -3.1   ...  5.8  ...  -0.7
                     |______________ V numbers ______________|
                                  these are LOGITS
```

Logits are raw scores. They can be negative, they can be large, and they don't add up to
anything. They are not probabilities yet. Softmax turns them into probabilities, because I need
two properties: every value between 0 and 1, and the whole set summing to exactly 1.

```
      LOGITS (raw)                        PROBABILITIES (after softmax)
      any real number                     all in [0,1], sum to 1.0

 o  |#################  5.8         o  |#######################  0.71
 l  |#######            2.1         l  |###                      0.09
 a  |                  -1.2         a  |                         0.01
 b  |#                   0.4        b  |.                        0.02
 c  |                  -3.1         c  |                         0.00
    +-----------------              ... everything else          0.17
    total is unconstrained             --------------------------------
                                                        total = 1.00
```

Softmax does two jobs at once:

- `exp` on every score. This forces everything positive, and it also blows up the gaps, so a lead
  of 2 in logits becomes a much bigger lead in probability. That's why the name is soft-*max*: it
  approximates picking the biggest one, but smoothly.
- divide by the sum. This forces the total to 1.

So the pipeline at one position is: logits -> softmax -> probability distribution over the next
character -> sample from it.

One more thing I didn't expect. During training I don't only get the output at the last position,
I get `V` numbers at *every* position from a single pass:

```
input:  h    e    l    l        ->    output shape: (4 positions x V scores)
        |    |    |    |
        v    v    v    v
      [V]  [V]  [V]  [V]              4 separate predictions, one pass
```

So a sequence of length 4 gives me 4 training signals, not 1. Length 32 gives me 32. That's a big
part of why these models train efficiently.

### What training data looks like (input vs target)

I did this by hand for "hello" with `block_size = 4`.

**Step A, build the vocab.** Unique characters of "hello", sorted, then numbered:

```
unique chars:  e, h, l, o          ids:  e=0  h=1  l=2  o=3      V = 4
```

**Step B, encode.** "hello" -> `[1, 0, 2, 2, 3]`

**Step C, take a window and shift it by one.**

```
              raw text:      h    e    l    l    o
              encoded:       1    0    2    2    3

 INPUT  = first 4:        +----+----+----+----+
                          | h  | e  | l  | l  |          [1, 0, 2, 2]
                          +----+----+----+----+
                             \    \    \    \
                              \    \    \    \    shifted left by one
                               v    v    v    v
 TARGET = last 4:           +----+----+----+----+
                            | e  | l  | l  | o  |        [0, 2, 2, 3]
                            +----+----+----+----+
```

The target is just the input moved one step earlier. The rule is `target[i] = input[i+1]`.

**Step D, the four problems hiding inside that one pair.** This is the part that made it click
for me:

```
 pos | what the model can see | must predict | input ids | target id
-----+------------------------+--------------+-----------+-----------
  0  | h                      | e            | [1]       | 0
  1  | h e                    | l            | [1,0]     | 2
  2  | h e l                  | l            | [1,0,2]   | 2
  3  | h e l l                | o            | [1,0,2,2] | 3
```

The "can see" column is a triangle, growing by one every row:

```
          can see ->    h    e    l    l
  pos 0            [    y    n    n    n  ]    sees only itself
  pos 1            [    y    y    n    n  ]
  pos 2            [    y    y    y    n  ]
  pos 3            [    y    y    y    y  ]    sees everything before it

                        y = allowed (past)
                        n = blocked (future)   <- this is the causal mask
```

That triangle *is* the causal mask. I derived it here from the definition of the task, before I
even read the paper, which means when I build the mask in Phase 5 I already know what it is:
keep the lower triangle including the diagonal, block the upper triangle.

It also explains how one pass gives 4 problems. Position 0 is solving a length-1 problem while
position 3 is solving a length-4 problem, in the same pass, and the mask is what stops them
cheating off each other.

**Step E, the off-by-one trap.** The window has to fit the input *and* the shifted target. With 5
characters and `block_size = 4`:

```
 valid start positions: only 0    (input = chars 0..3, target = chars 1..4)

 start = 1 -> input  = chars 1..4 = "ello"
              target = chars 2..5 = "llo?"   <- char 5 doesn't exist, crash
```

So a random start index has to come from `0 .. len(data) - block_size - 1`. If I forget that
`- 1` I get an IndexError, or worse, silently wrong data. Already listed as a predicted bug in
[03-errors/README.md](03-errors/README.md), and I'll write the test for it in Phase 4.

### Self check

I redid the whole input/target table for "banana" with `block_size = 3` from memory to make sure
I actually had it and wasn't just following along:

```
 unique chars: a, b, n   ->  a=0  b=1  n=2   V = 3
 "banana" encoded: [1, 0, 2, 0, 2, 0]

 window starting at 0:
   INPUT  = [1, 0, 2]      "ban"
   TARGET = [0, 2, 0]      "ana"

 pos | can see | must predict
 ----+---------+-------------
  0  | b       | a
  1  | b a     | n
  2  | b a n   | a

 valid starts with 6 chars and block_size 3: 0, 1, 2   (6 - 3 - 1 = 2)
```

## §2 — Attention Is All You Need

- The one-sentence idea of the paper:
- Encoder–decoder in the paper vs **decoder-only** for text generation (what I am building, and
  what I am dropping):
- The attention equation, restated in my own words:
- Why scaling by √d_k:
- Why multiple heads:
- Why the position-wise feed-forward network:
- What the paper does *not* explain that I had to find elsewhere:

## §3 — Concept notes from implementations (no code copied)

- The overall training loop shape:
- Character-level vs subword tokenisation:
- Things I noticed frameworks hide from you:

## §4 — Visual intuitions

- Embedding space, pictured:
- Attention as a lookup / soft dictionary:
- The N×N attention grid and what the causal mask does to it:
- Residual stream as a "highway" that each block reads from and writes back to:

## §5 — Backpropagation

- The chain rule as a **local** rule:
- What a computational graph is, and why a topological ordering is required:
- Forward pass vs backward pass — what is stored, and why memory grows with depth:
- What `zero_grad` corresponds to, and why gradients accumulate by default:

---

## Vocabulary I had to learn

| Term | My definition | First time I met it |
|---|---|---|
| logit | a raw unbounded score, one per vocab entry. not a probability until softmax | §1 |
| token | one unit of input. for me a token is a single character | §1 |
| embedding |  |  |
| d_model |  |  |
| head |  |  |
| context / block size | how many past characters the model is allowed to see. mine will be 32 | §1 |
| residual stream |  |  |
| temperature |  |  |
| autoregressive | feeding the model's own output back in as input, one step at a time | §1 |
| causal mask | blocking the future half of the attention grid, the upper triangle | §1 |

---

## Phase 0 understanding gate

Answer without notes, then move the answers into [VIVA.md](VIVA.md):

1. Why does text generation need a causal mask? — **answered in §1** (it falls out of
   `P(next | previous)`; without it the model reads the answer, and at generation time the future
   doesn't exist anyway)
2. Why divide attention scores by √d_k, and what goes wrong numerically without it? — not yet
3. Why is there a residual connection around every sublayer? — not yet
4. Why do Transformers need positional information when RNNs do not? — not yet
