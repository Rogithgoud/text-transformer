# Things I still don't know

Written down when I hit them, even the ones that feel stupid.

## Still open

Training always feeds the model real text, generating feeds it its own output. Why does nobody fix
that mismatch? I assume there is a standard approach and it costs too much to bother with. It is
probably part of why my samples fall apart halfway through, so I would like to know if that is my
bug or just how it is.

How do you pick the context length? I picked 32 for speed, and I know a longer one costs time
because attention grows with the square of it. What I do not know is what else goes wrong if it is
too long.

Should space, newline and punctuation be ordinary vocabulary entries? I kept them as normal
characters and it worked, but I never found out whether there is a reason to treat them differently.

The paper is encoder decoder and I built decoder only. I know I dropped the encoder and cross
attention. What I do not know is whether anything in the paper stops making sense without them,
which matters when I actually read it.

In block 1, one head has a clear job and the other looks scattered. Is that a small model failing to
specialise both heads, or is it a real pattern?

## Answered

Does one sequence give one prediction problem or many? Many. One pass gives logits at every
position, so a window of 32 is 32 problems at once, and the mask stops them cheating off each other.

Where does the causal mask come from? Straight out of `P(next | previous)`. Write out what each
position can see and the triangle is right there.

Why sample instead of always taking the most likely character? Because with no randomness it loops,
"the the the". Confirmed by trying temperature 0 and watching it repeat almost immediately.

Does softmax exaggerating the gaps make an untrained model overconfident? Yes, and it cost me a
prediction. My run started at 5.02 instead of 4.17 because the output layer's starting weights were
big enough to give the first guesses opinions.

## Probably won't get to

Why is `z` at 0.80 with `v` in the embedding table? My guess is rare characters never get enough
gradient to move anywhere, so they keep their random start and any similarity between two of them is
noise. To test it I would count how often each character appears and check whether all the odd
pairings are rare ones.
