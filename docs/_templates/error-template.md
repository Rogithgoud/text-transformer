# Error NNN: <short name>

> Copy into `docs/03-errors/NNN-short-name.md`.
> **Fill sections 1–4 BEFORE you fix it.** Writing down the wrong hypothesis is the point —
> it is the evidence of how the debugging actually went.

**Date:** YYYY-MM-DD
**Phase:** which roadmap phase
**Where:** `src/....py` : line
**Severity:** crash / silently wrong numbers / wrong output / slow

---

## 1. Symptom

What exactly happened. Paste the real traceback or the wrong numbers — never paraphrase them.

```
```

## 2. What I first thought was wrong

My initial hypothesis, honestly recorded, even if it was completely off.

## 3. How I investigated

The steps: what I printed, what I isolated, what I compared against a hand calculation, what
I ruled out and how.

## 4. What was actually wrong

The real root cause.

## 5. The math reason

Why this was wrong *mathematically*, not just mechanically. Examples: "`exp` of a large score
overflows a 64-bit float, so softmax needs the max subtracted first — which is valid because
softmax is invariant to a constant shift in its inputs"; "I transposed the wrong axis, so I was
computing similarity between the wrong pair of vectors."

If the cause was purely mechanical (a typo, a wrong variable name), say so plainly — but check
whether a missing invariant test would have caught it.

## 6. The fix

What changed, and why that is correct.

```
```

## 7. How I now prevent it

The test, assertion, or invariant added so this class of bug cannot return silently.
`tests/....py`

## 8. Time lost

Roughly how long. Useful later for judging where the real difficulty of this project sat.
