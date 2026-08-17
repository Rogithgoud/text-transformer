# Component: <NAME>

> Copy this file into `docs/02-components/NN-name.md`. Do not delete a section — if a section
> is genuinely empty, write *why* it is empty. Empty sections are where understanding hides.

**Status:** not started / implemented / tested / documented
**Source file:** `src/....py`
**Test file:** `tests/....py`
**Date started:** YYYY-MM-DD · **Date finished:** YYYY-MM-DD

---

## 1. What it is

One paragraph, plain English, no jargon. Explain it as if to someone who knows programming
but not machine learning.

## 2. Why it exists

What breaks or degrades if this component is removed entirely? Be concrete. If you cannot say
what breaks, you do not yet know why it exists.

## 3. The math

**Formula:**

```
<write the formula here>
```

**Every symbol defined:**

| Symbol | Meaning | Shape / range |
|---|---|---|
|  |  |  |

**Worked example with tiny numbers.** Do it by hand on paper first (2×2 or 3-vectors),
photograph or transcribe it here, then confirm the code reproduces the same numbers.

```
input:
by-hand output:
code output:
match? yes / no
```

## 4. What it does visually / geometrically

The spatial intuition. Examples of the right kind of answer: "this projects a vector onto a
lower-dimensional space", "this squashes any vector onto the probability simplex so the bars
sum to 1", "this blacks out the upper triangle so information cannot flow backwards in time".

Include an ASCII sketch, a drawing, or a link into `docs/05-visuals/`.

## 5. Shapes in / shapes out

| | Shape | Notes |
|---|---|---|
| Input |  |  |
| Parameters |  | count = ... |
| Output |  |  |

## 6. Derivative (backward pass)

The gradient of the output with respect to each input **and** each parameter.

**Derivation** — show the steps, not just the result:

```
```

**Gradient check:** analytic vs numerical `(f(x+h) − f(x−h)) / 2h`, with `h = ...`
Max relative error observed: `...`

## 7. How I tested it

- [ ] Shapes verified
- [ ] Hand-computed example matches code
- [ ] Gradient check passes
- [ ] Component-specific invariant (e.g. attention rows sum to 1; mask blocks the future)

Commands run and their output:

```
```

## 8. What confused me here

The honest bit. What took longest, what I got wrong first, what finally made it click.
Link any related entries in `docs/03-errors/`.
