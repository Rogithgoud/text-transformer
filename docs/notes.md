# What the task actually is

I worked this out before writing any code, because I wanted to know what I was building.

## The one thing the model does

Guess the next character from the characters before it. That is all.

It looks too small to be useful, but the probability of a whole string breaks into a chain of
exactly these guesses:

```
P("hello") = P(h) x P(e|h) x P(l|h,e) x P(l|h,e,l) x P(o|h,e,l,l)
```

Every factor is the same question. So a model that answers "what comes next" can score any text.

And "from the characters before it" means a position can only look backwards. That is where the
causal mask comes from. It is not a trick someone invented, it is in the definition. If the model
could see ahead it would just copy the answer, and at generation time there is nothing ahead
anyway.

## Why one character at a time gives paragraphs

Run it in a loop and feed its own output back in.

```
"h"      -> guess -> e
"he"     -> guess -> l
"hel"    -> guess -> l
"hell"   -> guess -> o
```

Three things follow from this, and all three showed up in my results.

The input keeps growing, so there has to be a limit. Mine is 32 characters and past that the oldest
get dropped, so the model genuinely cannot remember the start of a long paragraph.

Mistakes compound, because a bad character becomes input for every step after it. That is why bad
output degenerates instead of staying slightly wrong.

Training and generating are different situations. Training always feeds it real text, generating
feeds it its own text. Nothing in a basic setup fixes that.

## What comes out at one position

65 numbers, one score per possible next character. These are called logits and they are raw, they
can be negative and they do not add up to anything.

Softmax turns them into probabilities. It does two jobs: `exp` makes everything positive and
exaggerates the gaps, then dividing by the sum forces the total to 1.

One thing I did not expect. During training the model gives me 65 numbers at *every* position from
one pass, not just the last one. So a window of 32 characters is 32 training signals, not 1.

## What a training example looks like

Did this by hand for "hello" with a window of 4.

Sorted unique characters get numbered: `e=0 h=1 l=2 o=3`, so "hello" is `[1,0,2,2,3]`.

Take 4, then take the same 4 shifted one to the right:

```
input  = [1, 0, 2, 2]     "hell"
target = [0, 2, 2, 3]     "ello"
```

So `target[i]` is `input[i+1]`. Nothing more than that.

The bit that made it click is that this one pair is four separate problems:

```
 pos | can see | must predict
-----+---------+-------------
  0  | h       | e
  1  | h e     | l
  2  | h e l   | l
  3  | h e l l | o
```

Look at the "can see" column. It grows by one each row, so it is a triangle:

```
          h  e  l  l
 pos 0 [  y  n  n  n ]
 pos 1 [  y  y  n  n ]
 pos 2 [  y  y  y  n ]
 pos 3 [  y  y  y  y ]
```

That triangle is the mask. I got to it from the definition of the problem, before reading anything
about attention, so when I built the mask later I already knew what it was for.

It also explains the 32 signals from one pass. Position 0 is solving a one character problem while
position 3 solves a four character one, at the same time, and the mask is what stops them cheating
off each other.

## The off by one

The window needs to fit the input and the shifted target. With 5 characters and a window of 4, the
only legal start is 0, because the target needs a character past the end of the input.

So the random start has to come from `0 .. len(data) - block_size - 1`. Miss that `- 1` and you
either crash or, worse, quietly train on wrong data. Tested in `tests/test_dataset.py`.

## Checking myself

Redid the whole thing for "banana" with a window of 3 from memory:

```
a=0 b=1 n=2      "banana" -> [1,0,2,0,2,0]
input  = [1,0,2]  "ban"
target = [0,2,0]  "ana"
legal starts: 0, 1, 2
```
