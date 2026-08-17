# Error 001: gradient check failed on cells where the gradient is genuinely zero

**Date:** 2026-08-18
**Phase:** 3 (backward passes) / testing
**Where:** `tests/test_gradcheck.py`
**Severity:** false alarm, but it blocked the run

---

## 1. Symptom

The gradient check failed even though the numbers printed looked identical:

```
checked 147 cells, worst relative error 3.331e-03
  worst was param 12 cell (0,3): numerical 0.00000000 vs analytic -0.00000000
AssertionError: gradient check failed, worst relative error 3.331e-03
```

Every other cell was fine. The worst offender printed as 0.0 against -0.0.

## 2. What I first thought was wrong

That I had a real bug in one backward pass, probably layernorm since that one has
the messiest derivation, and that param 12 would point me at it.

## 3. How I investigated

Printed the actual magnitudes instead of 8 decimal places. Both numbers were
around 1e-11, not zero but tiny. Then looked at what param 12 was: a bias, or an
embedding row for a character that never appears in the test window. Either way a
weight the loss genuinely does not depend on.

## 4. What was actually wrong

Nothing was wrong with the gradients. The test was wrong.

## 5. The math reason

The check compares relative error:

```
rel = |numerical - analytic| / (|numerical| + |analytic|)
```

When both numbers are real gradients that division is exactly what I want, since
gradients across the model differ by orders of magnitude and an absolute
threshold would be meaningless.

But when the true gradient is 0, the analytic result is exactly 0.0 and the
numerical one is float noise. The numerical estimate is

```
(loss(w+h) - loss(w-h)) / (2h)
```

and with h = 1e-5, the two losses are identical to about 15 digits. Subtracting
two nearly equal floats destroys almost all the significant digits (catastrophic
cancellation), leaving whatever is left in the last bits, then that gets divided
by 2e-5, which magnifies it by 50,000. So a rounding error of 1e-16 comes out as
about 5e-12.

Then rel = noise / noise, which can be anything from 0 to 1. It is not measuring
error, it is measuring nothing.

## 6. The fix

Cells where both gradients are below 1e-6 are checked on absolute difference
instead of relative, and counted separately:

```
if max(abs(numerical), abs(analytic)) < TINY:
    assert abs(numerical - analytic) < TINY
    skipped += 1
    continue
```

This is not loosening the test. A zero gradient still has to come out as zero. It
just stops dividing noise by noise.

After the fix:

```
checked 128 cells (19 skipped as zero), worst relative error 3.712e-08
gradient check passed
```

3.7e-08 on 128 real gradient cells across matmul, softmax, layernorm, attention,
the residuals and the embedding. Every hand-derived backward pass is right.

## 7. How I now prevent it

The rule is in the docstring of `tests/test_gradcheck.py`: relative error for
real gradients, absolute for zero ones. Worth remembering for any future finite
difference check, not just this project.

## 8. Time lost

About 20 minutes, most of it suspecting layernorm for no reason.
