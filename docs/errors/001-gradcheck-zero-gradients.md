# Gradient check failed on weights whose gradient is actually zero

2026-08-18, in `tests/test_gradcheck.py`. Turned out not to be a real bug, but it stopped me for a
while.

## What I saw

```
checked 147 cells, worst relative error 3.331e-03
  worst was param 12 cell (0,3): numerical 0.00000000 vs analytic -0.00000000
AssertionError: gradient check failed
```

Everything else was fine. The one that failed printed as 0.0 against -0.0, which made no sense.

## What I thought it was

That one of my backward passes was wrong, probably layernorm since that has the messiest derivation.

## What it actually was

I printed the real magnitudes instead of 8 decimals and both numbers were about 1e-11. So not zero,
but tiny. Then I checked what param 12 is: a bias, or an embedding row for a character that never
appears in the test window. Either way a weight the loss genuinely does not depend on.

So the gradients were right and the test was wrong.

## Why, mathematically

The test compares relative error, `|numerical - analytic| / (|numerical| + |analytic|)`. For real
gradients that is correct, because gradients across the model differ by orders of magnitude and a
fixed threshold would mean nothing.

But when the true gradient is 0, the analytic value is exactly 0.0 and the numerical one is noise.
The numerical estimate is `(loss(w+h) - loss(w-h)) / 2h`, and with `h = 1e-5` those two losses are
identical to about 15 digits. Subtracting two nearly equal floats throws away almost all the
significant digits, and then dividing by `2e-5` multiplies whatever is left by 50,000. A rounding
error of 1e-16 comes out around 5e-12.

Then the relative error is noise divided by noise, which can be anything from 0 to 1. It is not
measuring error, it is measuring nothing.

## Fix

Cells where both gradients are under 1e-6 get compared absolutely instead, and counted separately:

```python
if max(abs(numerical), abs(analytic)) < TINY:
    assert abs(numerical - analytic) < TINY
    skipped += 1
    continue
```

Not loosening the test. A zero gradient still has to come out zero. It just stops dividing noise by
noise.

After that: `checked 128 cells (19 skipped as zero), worst relative error 3.712e-08`.

The rule is in the test's docstring now so I do not repeat it: relative error for real gradients,
absolute for zero ones.

Lost about 20 minutes, most of it suspecting layernorm for no reason.
