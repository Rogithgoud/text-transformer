# Gradient check failed on weights whose gradient is genuinely zero

2026-08-18. Found in `tests/test_gradcheck.py`. Not a real bug in the end, but it blocked the run
until I understood it.

## What I saw

The check failed even though the two numbers printed looked identical:

```
checked 147 cells, worst relative error 3.331e-03
  worst was param 12 cell (0,3): numerical 0.00000000 vs analytic -0.00000000
AssertionError: gradient check failed, worst relative error 3.331e-03
```

Every other cell was fine. The one that failed printed as 0.0 against -0.0.

## What I thought it was

That one of my backward passes was wrong, probably layernorm, since that has the messiest derivation
of the lot, and that param 12 would tell me where.

## How I found out

Printed the actual magnitudes instead of eight decimal places. Both numbers were about 1e-11, so not
zero but tiny. Then checked what param 12 actually is: a bias, or an embedding row for a character
that never appears in the test window. Either way a weight the loss genuinely does not depend on.

## What was actually wrong

Nothing was wrong with the gradients. The test was wrong.

## The reason, mathematically

The check compares relative error:

```
rel = |numerical - analytic| / (|numerical| + |analytic|)
```

For real gradients that is the right thing to compare, because gradients across the model differ by
orders of magnitude and an absolute threshold would be meaningless.

But when the true gradient is 0, the analytic result is exactly 0.0 and the numerical estimate is
float noise. The numerical estimate is

```
(loss(w + h) - loss(w - h)) / 2h
```

and with `h = 1e-5` those two losses are identical to about 15 digits. Subtracting two nearly equal
floats destroys almost all the significant digits, which leaves whatever was in the last bits, and
then dividing by `2e-5` multiplies that by 50,000. So a rounding error of 1e-16 comes out as roughly
5e-12.

Then `rel` is noise divided by noise, which can be anything between 0 and 1. It is not measuring
error, it is measuring nothing.

## The fix

Cells where both gradients are below 1e-6 get compared absolutely instead of relatively, and counted
separately:

```python
if max(abs(numerical), abs(analytic)) < TINY:
    assert abs(numerical - analytic) < TINY
    skipped += 1
    continue
```

This is not loosening the test. A zero gradient still has to come out as zero. It just stops
dividing noise by noise.

After the fix:

```
checked 128 cells (19 skipped as zero), worst relative error 3.712e-08
gradient check passed
```

3.7e-08 across matmul, softmax, layernorm, attention, the residuals and the embedding backward.

## What stops it coming back

The rule is written in the docstring of the test: relative error for real gradients, absolute for
zero ones. Worth remembering for any finite difference check, not just this one.

## Time lost

About 20 minutes, most of it suspecting layernorm for no reason.
