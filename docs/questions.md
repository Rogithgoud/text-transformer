# Things I still don't know

Anything I could not explain gets written down here when I hit it, even if it feels like a stupid
question. Some of these I expect to answer, some I probably won't get to.

## Open

**In training the model always sees real text, but when generating it sees its own output. Why does
nobody fix that mismatch?** I assume there is a standard approach and it is too expensive to be
worth it, but I don't know. It is probably part of why my samples fall apart partway through, so I
want to know whether that is my bug or a known property. Raised 2026-08-17.

**How should I pick the context length?** I chose 32 for speed. I know a longer context costs
time because attention grows with the square of it, but I don't know what else gets worse if it is
too long, or what specifically breaks when it is too short beyond the obvious. Raised 2026-08-17.

**Should newline, space and punctuation be ordinary vocabulary entries?** I kept them all as normal
characters, which seemed right and worked, but I never found out whether there is a reason to treat
them differently. Raised 2026-08-17.

**The paper is encoder-decoder and I built decoder only. What exactly did I drop?** I know I dropped
cross attention and the encoder stack. What I don't know is whether anything in the paper stops
making sense without them, which matters when I sit down to read it properly. Raised 2026-08-17.

**Why is my attention block 1 head 0 so noisy compared to head 1?** Looking at the grids, one head
per block ends up with a clear job and the other looks scattered. Is that just a small model not
having enough signal to specialise both, or is it a real pattern. Raised 2026-08-18.

## Answered

**Does one training sequence give one prediction problem or many?** Many. One forward pass produces
logits at every position, so a window of 32 gives 32 problems at once, and the causal mask is what
stops them cheating off each other. Answered 2026-08-17, written up in notes.md.

**Where does the causal mask actually come from?** It falls out of the definition
`P(next | previous)`. Writing out what each position is allowed to see gives the triangle directly.
Answered 2026-08-17, written up in notes.md.

**Why sample instead of always taking the most likely character?** Greedy has no randomness, so it
falls into loops, "the the the". Confirmed by trying it: temperature 0 output repeats almost
immediately. Answered 2026-08-18.

**Does softmax exaggerating the gaps make an untrained model overconfident?** Yes, and it showed up
as a real result. My first run started at loss 5.02 instead of `ln(65) = 4.17`, because the output
head init was large enough that the first logits already had spread. The model started confidently
wrong, and confident wrong answers cost more than being uniformly clueless. Answered 2026-08-18,
written up in experiments/001-shakespeare.md.

## Probably won't get to

**Why does `z` end up with cosine similarity 0.80 to `v` in the embedding table?** My guess is that
rare characters never receive enough gradient to be pushed anywhere meaningful, so they keep most of
their random initialisation and any similarity between two of them is noise. I have not tested that,
and the way to test it would be to count how often each character appears and check whether the
odd pairings are all rare ones.
