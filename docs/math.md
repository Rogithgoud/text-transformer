# The derivations

The ones I had to do on paper. Everything else in the code follows from these.

## Softmax with cross entropy

Forward, for one position with logits `z` and correct character `c`:

```
p_i  = exp(z_i) / sum of exp(z_j)
loss = -log(p_c)
```

The loss is basically surprise. Right answer given probability 1 means loss 0. Given 0.01 means
loss 4.6. The log is what makes being confidently wrong expensive rather than just wrong.

For the gradient there are two cases, because `z_c` sits in the top and the bottom of the fraction
while every other `z_i` only sits in the bottom. Working both out:

```
for the right character:   p_c - 1
for everything else:       p_i
```

Which is one line if the target is a one hot vector:

```
d(loss)/dz = p - onehot(c)
```

Predicted minus actual. Every wrong character gets pushed down by exactly the probability it
claimed, and the right one gets pushed up by exactly what it was missing. All the messy softmax
terms cancel against the log, and that only happens because these two are used together. In the
code it is one subtraction.

This also gives me the number I check every run against. If all the logits are equal then every
`p` is `1/65` and the loss is `log(65) = 4.17`. Note that is for *equal* logits, which only matches
an untrained model if the starting weights are small. Mine were not, which is why my first run
started at 5.02.

## Matmul backward

With `Y = A B`, write one output cell as a sum:

```
Y[i][j] = sum over p of A[i][p] * B[p][j]
```

`A[i][p]` only appears in row i of Y, multiplied by `B[p][j]`. Differentiate and collect:

```
dA = dY B^T
dB = A^T dY
```

Once you have it, the way to remember it is that the shapes only fit one way round. When I was not
sure if a transpose belonged somewhere I wrote the shapes out and there was only one arrangement
that worked. That caught two of my mistakes in the attention backward.

## Layernorm backward

Forward, on one position's 32 channels:

```
mu   = mean(x)
var  = mean((x - mu)^2)
xhat = (x - mu) / sqrt(var + eps)
out  = gamma * xhat + beta
```

Gamma and beta are easy since they are applied elementwise. The input gradient is the awkward one,
and the reason is that `mu` and `var` both depend on every element in the row. So changing one
number moves the normalised value of all the others. There are three paths from `x_i` to the output:
straight through, through `mu`, and through `var`. Adding them and simplifying:

```
dxhat = dout * gamma
dx    = ( dxhat - mean(dxhat) - xhat * mean(dxhat * xhat) ) / sqrt(var + eps)
```

The two mean terms are the `mu` path and the `var` path.

A check I like on this one: if every `dxhat` is the same constant, then `mean(dxhat)` equals it and
`mean(dxhat * xhat)` is 0 because `xhat` has mean 0, so `dx` comes out 0. Which is right, because
shifting a whole row by the same amount is exactly what layernorm throws away, so it should have no
gradient.

## Softmax backward on its own

Needed inside attention, where softmax is not followed by cross entropy so nothing cancels. For one
row, with `p` the output and `g` the incoming gradient:

```
ds_j = p_j * ( g_j - sum over k of g_k * p_k )
```

That subtracted sum is what keeps the row adding to 1. Push one entry up and the others have to come
down.

Useful side effect in attention: masked cells have `p_j = 0`, so their gradient is 0 automatically.
I do not have to mask again on the way back, and nothing can leak in from the future.

## Do I trust any of this

Not because it looks right. Every one is checked against the definition of a derivative:

```
numerical = ( loss(w + h) - loss(w - h) ) / 2h
```

with `h = 1e-5`, in `tests/test_gradcheck.py`. Worst disagreement over 128 sampled weights was
3.7e-08.

Two sided rather than one sided because the error shrinks faster. And `h` is a compromise both ways:
too big and the measurement is off because the loss curves, too small and the two losses are so
close that subtracting them destroys the digits and you are left with float noise. That second
failure is what my first error log entry is about.
