# What the task actually is

Worked out before I wrote any code, because I wanted to know what I was building rather than copy a
diagram. Everything here is in my own words.

## The task as probability

The model estimates one thing:

```
P( next character | all the characters before it )
```

I did not see at first how one small question like that covers all of language. It does, because of
the chain rule of probability. The probability of a whole string splits into a product of next
character questions:

```
P("hello") = P(h) x P(e|h) x P(l|h,e) x P(l|h,e,l) x P(o|h,e,l,l)
```

Every factor is the same kind of question, so a model that answers "what comes next" can score any
text at all.

"Given everything before" also means a position may only look backwards:

```
position:   0    1    2    3    4
char:       h    e    l    l    o

to predict position 4 I may use:  h  e  l  l
                                  ^^^^^^^^^^  the past
                                            o   not this
```

That restriction is where the causal mask comes from. It is not a trick somebody invented, it falls
out of the definition. If the model could see the future it would copy the answer and learn nothing,
and at generation time the future does not exist anyway.

## Why one character at a time gives paragraphs

The model only ever predicts one step. Long text comes from feeding its own output back in.

```
  "h"      -> model -> probabilities -> pick 'e'
  "he"     -> model -> probabilities -> pick 'l'
  "hel"    -> model -> probabilities -> pick 'l'
  "hell"   -> model -> probabilities -> pick 'o'
     ^                                    |
     |________ append it and go again ____|
```

Three things follow, and all three showed up in my results.

The input keeps growing, so there has to be a maximum context. Mine is 32 characters, and past that
the oldest characters get dropped. The model has no memory of anything earlier, at all.

Mistakes compound, because a bad character becomes part of the input for every step after it. That
is why weak output degenerates into nonsense partway through instead of staying slightly wrong.

Training and generation are different situations. In training the model always gets real text. In
generation it gets its own output. Nothing in a basic setup fixes that, so it goes in the readme as
a limitation.

## What comes out at one position

The vocabulary is every unique character in the corpus, 65 of them for Shakespeare. At each position
the model outputs exactly 65 numbers, one score per possible next character.

```
              char:    a     b     c    ...   o   ...   z
             score:  -1.2   0.4  -3.1   ...  5.8  ...  -0.7
                     |______________ 65 numbers ____________|
                                these are logits
```

Logits are raw scores. They can be negative, they can be large, they do not add up to anything.
Softmax turns them into probabilities, because I need every value between 0 and 1 and the whole set
summing to 1:

```
      logits (raw)                     probabilities (after softmax)

 o  |#################  5.8      o  |#######################  0.71
 l  |#######            2.1      l  |###                      0.09
 a  |                  -1.2      a  |                         0.01
 b  |#                   0.4     b  |.                        0.02
 c  |                  -3.1      c  |                         0.00
    total unconstrained             everything else           0.17
                                    ------------------------------
                                                    total = 1.00
```

Softmax does two jobs. `exp` on every score forces them positive and exaggerates the gaps, so a
lead of 2 in logits becomes a much bigger lead in probability, which is where the name soft max
comes from. Then dividing by the sum forces the total to 1.

One thing I did not expect: during training the model gives 65 numbers at every position from a
single pass, not just at the last one.

```
input:  h    e    l    l      ->   output is (4 positions x 65 scores)
        |    |    |    |
        v    v    v    v
      [65] [65] [65] [65]          four predictions, one pass
```

So a window of 32 characters gives 32 training signals rather than 1.

## What a training example looks like

Did this by hand for "hello" with a window of 4.

Vocabulary, sorted unique characters, then numbered:

```
e=0  h=1  l=2  o=3        "hello" -> [1, 0, 2, 2, 3]
```

Take a window and shift it by one:

```
              text:          h    e    l    l    o
              ids:           1    0    2    2    3

 input  = first 4:        +----+----+----+----+
                          | h  | e  | l  | l  |         [1, 0, 2, 2]
                          +----+----+----+----+
                             \    \    \    \
                               v    v    v    v
 target = last 4:           +----+----+----+----+
                            | e  | l  | l  | o  |       [0, 2, 2, 3]
                            +----+----+----+----+
```

The target is the input moved one step earlier, so `target[i] = input[i+1]`.

Then the part that made it click. That one pair contains four separate problems:

```
 pos | can see | must predict | input ids | target
-----+---------+--------------+-----------+--------
  0  | h       | e            | [1]       | 0
  1  | h e     | l            | [1,0]     | 2
  2  | h e l   | l            | [1,0,2]   | 2
  3  | h e l l | o            | [1,0,2,2] | 3
```

The "can see" column is a triangle that grows by one each row:

```
          can see ->    h    e    l    l
  pos 0            [    y    n    n    n  ]
  pos 1            [    y    y    n    n  ]
  pos 2            [    y    y    y    n  ]
  pos 3            [    y    y    y    y  ]
```

That triangle is the causal mask. I got to it from the definition of the task rather than from the
paper, which means when I built the mask later I already knew what it was for: keep the lower
triangle including the diagonal, block everything above it.

It also explains how one pass gives four problems. Position 0 is solving a one character problem
while position 3 is solving a four character problem, in the same pass, and the mask is what stops
them cheating off each other.

## The off by one

The window has to fit the input and the shifted target. With 5 characters and a window of 4:

```
 legal start positions: only 0     input = chars 0..3, target = chars 1..4

 start = 1 -> input  = chars 1..4 = "ello"
              target = chars 2..5 = "llo?"   character 5 does not exist
```

So a random start index has to come from `0 .. len(data) - block_size - 1`. Forget the `- 1` and
you either get an IndexError or, worse, quietly wrong data. This is tested in
`tests/test_dataset.py`.

## Checking myself

I redid the whole thing for "banana" with a window of 3, from memory, to make sure I had it and was
not just following along:

```
 a=0  b=1  n=2        "banana" -> [1, 0, 2, 0, 2, 0]

 window at 0:  input = [1, 0, 2] "ban"    target = [0, 2, 0] "ana"

 pos | can see | must predict
 ----+---------+-------------
  0  | b       | a
  1  | b a     | n
  2  | b a n   | a

 legal starts with 6 characters and a window of 3: 0, 1, 2
```

## Words I had to learn

Logit, a raw unbounded score, one per vocabulary entry, not a probability until softmax.

Token, one unit of input. Here a token is a single character.

Context or block size, how many past characters the model may see. Mine is 32.

Autoregressive, feeding the model's own output back in as input, one step at a time.

Causal mask, blocking the future half of the attention grid.

Residual stream, the fixed width channel that runs from the embedding to the output head, that every
block reads from and adds back into.
