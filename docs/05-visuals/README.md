# 05 — Visual Artifacts

Your senior asked specifically for *visual* understanding. These are produced deliberately, not
as an afterthought. No matplotlib — ASCII, hand-built SVG/HTML, or photographs of paper drawings.

| # | Artifact | File | Made in phase | Done | Written interpretation |
|---|---|---|---|---|---|
| 01 | Architecture diagram, every arrow labelled with shapes | `01-architecture.*` | 0 | ☐ | ☐ |
| 02 | Shape-flow, one page: input ids → logits | `02-shape-flow.md` | 6 | ☐ | ☐ |
| 03 | Computational graph, one tiny forward + backward, values and grads on every edge | `03-computational-graph.*` | 3 | ☐ | ☐ |
| 04 | Dot product as projection | `04-dot-product.*` | 2 | ☐ | ☐ |
| 05 | Softmax before/after, as bars | `05-softmax-bars.*` | 2 | ☐ | ☐ |
| 06 | Positional encoding as a heatmap of waves | `06-positional-heatmap.txt` | 5 | ☐ | ☐ |
| 07 | Causal mask: the blacked-out upper triangle | `07-causal-mask.txt` | 5 | ☐ | ☐ |
| 08 | Attention heatmap per head, for one fixed sentence | `08-attention-heads.*` | 9 | ☐ | ☐ |
| 09 | Embedding nearest neighbours, before vs after training | `09-embedding-neighbours.md` | 9 | ☐ | ☐ |
| 10 | Loss curve, annotated with what changed and when | `10-loss-curve.*` | 9 | ☐ | ☐ |
| 11 | Temperature: the same distribution at T = 0.5, 1.0, 2.0 | `11-temperature.*` | 8 | ☐ | ☐ |

## The interpretation rule

A picture without a written interpretation proves nothing. Every artifact above needs a
paragraph answering: **what am I looking at, and what does it tell me the model has learned?**

For the attention heatmaps specifically, name what each head appears to do — attends to the
previous character, attends to spaces or line breaks, spreads attention evenly, focuses on the
start of the sequence. Being wrong here is fine; not looking is not.

## How to render a heatmap without a plotting library

Map each value to a character by magnitude, e.g. `" .:-=+*#%@"`, and print row by row with the
axes labelled. For SVG, emit `<rect>` elements with a computed fill — plain string formatting,
no dependencies. Document whichever scale you choose, and whether it is linear or logarithmic;
an undocumented colour scale makes a heatmap unreadable.
