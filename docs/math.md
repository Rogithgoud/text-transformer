# The derivations

The three I actually had to work out on paper. Everything else in the code is either a definition or
follows from these.

## Softmax and cross entropy together

The forward direction. For one position with logits `z` and the correct character `c`:

```
p_i  = exp(z_i) / sum_j exp(z_j)
loss = -log(p_c)
```

Read the loss as surprise. If the model gave the right character probability 1, the loss is
`-log(1) = 0`. If it gave it 0.01, the loss is 4.6. The log is what makes being confidently wrong
expensive, rather than just wrong.

Now the gradient. Two cases, because `z_c` appears in the numerator and in the denominator, and
every other `z_i` appears only in the denominator.

For the correct character:

```
d(loss)/d(z_c) = p_c - 1
```

For everything else:

```
d(loss)/d(z_i) = p_i
```

Which is one line if you write the target as a one hot vector:

```
d(loss)/d(z) = p - onehot(c)
```

Predicted minus actual. Every wrong character gets pushed down by exactly the probability it
claimed, and the right one gets pushed up by exactly what it was missing. All the messy softmax
derivative terms cancel against the derivative of the log, and that only happens because these two
functions are used together. This is the cleanest thing in the project and it's in `src/loss.py` as
a single subtraction.

The other reason it matters: `ln(V)` is the loss when all the logits are equal, because then every
`p_i` is `1/V` and the loss is `-log(1/V) = log(V)`. For 65 characters that is 4.1744, and it is the
first number I check on any run. Note that this is the loss for *equal* logits, which is only the
same as an untrained model when the initialisation is small enough. My first run started at 5.02
because the output head init was too big and the model started out confidently wrong.

## Matmul backward

Take `Y = A B` with `A` of shape `(n, k)` and `B` of shape `(k, m)`. Write one output element as an
explicit sum:

```
Y[i][j] = sum over p of A[i][p] * B[p][j]
```

Differentiate that with respect to one element of `A`. `A[i][p]` only appears in row `i` of `Y`, and
in that row it multiplies `B[p][j]`:

```
d(loss)/d(A[i][p]) = sum over j of dY[i][j] * B[p][j]
```

which is exactly row `i` of `dY` times row `p` of `B`, so:

```
dA = dY B^T          shape (n, k)
dB = A^T dY          shape (k, m)
```

The quick way to remember it, once derived: the shapes only fit one way round. If I am unsure
whether a transpose belongs, I write the shapes down and there is only one arrangement that works.
That trick caught two mistakes while I was writing the attention backward.

## Layernorm backward

The forward pass, on one position's `C` channels:

```
mu   = mean(x)
var  = mean((x - mu)^2)
xhat = (x - mu) / sqrt(var + eps)
out  = gamma * xhat + beta
```

The parameter gradients are easy, since `gamma` and `beta` are applied elementwise:

```
d(gamma) = sum over positions of dout * xhat
d(beta)  = sum over positions of dout
```

The input gradient is the awkward one, and the reason is worth stating: `mu` and `var` each depend
on every element of the row, so changing one `x_i` moves the normalised value of every other
channel too. So there are three paths from `x_i` to the output: directly through `xhat_i`, through
`mu`, and through `var`. Adding those three up and simplifying gives:

```
dxhat = dout * gamma
dx    = ( dxhat - mean(dxhat) - xhat * mean(dxhat * xhat) ) / sqrt(var + eps)
```

The two mean terms are the `mu` path and the `var` path. A useful sanity check on the result: if
every `dxhat` is the same constant, then `mean(dxhat)` equals it and `mean(dxhat * xhat)` is 0
because `xhat` has mean 0, so `dx` comes out as 0. That is correct, since shifting a whole row
equally is exactly what layernorm removes, and a change the layer discards should have no gradient.

## Softmax backward on its own

Needed for attention, where softmax is not followed by cross entropy and so nothing cancels. For one
row, with `p` the softmax output and `g` the incoming gradient:

```
d(loss)/d(s_j) = p_j * ( g_j - sum over k of g_k * p_k )
```

The subtracted term is what keeps the row summing to 1. If a gradient pushes one entry up, the
others have to come down, and that sum is how much comes off everything.

In the attention code this has a useful side effect: masked positions have `p_j = 0`, so their
gradient is 0 automatically. Nothing has to be masked again on the backward pass, and nothing can
leak back through the future.

## How I know these are right

None of the above is trusted because it looks right. Every one is checked against the definition of
a derivative:

```
numerical = ( loss(w + h) - loss(w - h) ) / 2h
```

with `h = 1e-5`, in `tests/test_gradcheck.py`. Worst relative error over 128 sampled weights was
3.7e-08.

Two sided rather than one sided, because its error shrinks like `h^2` instead of `h`. And `h` is a
compromise in both directions: too large and the measurement is wrong because the loss curves, too
small and the two losses are so close that subtracting them destroys the significant digits and the
answer is float noise. That second failure is what error 001 in this repo is about.
