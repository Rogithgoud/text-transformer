# Log

## 2026-08-17

Set up the repo and decided how strict to be about the no libraries rule. Settled on standard
library only and wrote it down so it does not drift later.

Then the first real work: what a language model actually is. Wrote the task out as
`P(next | previous)`, worked through the generation loop, and did the input and target windows by
hand for "hello", then again for "banana" from memory to be sure I had it.

The thing that surprised me is that the target is just the input shifted by one, and that one pair
contains as many prediction problems as there are positions. Writing out what each position can see
gave me a triangle, and that triangle is the mask. I did not expect to get there on my own.

Also spotted the off by one in the window index before writing any code, which is the sort of bug
that would not crash and would just quietly train on wrong data.

Two things I could not explain and wrote down instead of pretending: why nobody fixes the mismatch
between training on real text and generating from your own output, and whether softmax exaggerating
the gaps makes an untrained model overconfident.

## 2026-08-18

Built the whole thing and trained it. 11 modules, 5 test files, then 2000 steps on Tiny Shakespeare,
70 minutes, loss 5.02 down to 2.19 on held out text.

Two decisions to get there in a day rather than a week. No autograd engine, each layer just has its
own backward. And batches are a loop over sequences instead of an extra dimension on every matrix,
so everything stays 2D. Pure Python is stuck in for loops anyway, so the dimension would have cost
me clarity for nothing.

What made backprop click: the backward pass is the forward pass in reverse, nearly line for line.
`out = probs @ V` becomes `dprobs = dout @ V.T` and `dV = probs.T @ dout`. The shapes only fit one
way, which is how I checked myself when I was not sure about a transpose, and it caught two
mistakes.

The three tests each caught a different kind of problem. The gradient check is the only reason I
trust the derivations at all, 3.7e-08 over 128 weights. Overfitting one batch, 2.51 down to 0.00096,
proved forward and backward and Adam work together before I spent an hour training. And the mask
test protects the result from being fake, since a leak makes the loss look better, not worse.

Two things went wrong. The gradient check failed the first time at 3.3e-03 and I spent twenty
minutes sure my layernorm derivative was broken. It was not, the test was comparing two numbers that
were both basically zero. Written up in errors/001.

And I predicted the run would start at `ln(65) = 4.17` and it started at 5.02, because my output
layer starts with weights big enough that the first guesses already have opinions. Being confidently
wrong costs more than knowing nothing. That is the change I want to make next.

Also caught myself writing the parameter count from memory as 41,761 when it is 29,697. Fixed it.
Numbers go in these files from the output now, not from what I think I remember.

## 2026-08-20

Cleaned up. The docs had grown a scaffolding of templates and checklists that I was filling in
instead of actually writing notes, so that is gone. Rewrote the readme in my own words too.

Switched the plots from ASCII to matplotlib, since the rule was about not letting a library do the
model's math and matplotlib only draws pictures. Worth it immediately: in the text version of the
attention grid I thought a stripe was on the `z` of "Citizen", and in the real image it is obviously
column 0 and the punctuation columns. Fixed that in the notes.

Next: read the paper now that I have built the thing it describes, and run the next experiment with
the smaller output head init.
