# Embedding neighbours after training

Cosine similarity between rows of the embedding table. 1.0 means the same
direction, 0.0 unrelated, -1.0 opposite. Nothing in the code groups characters,
so any structure here was learned from the text.

- `_` -> - 0.75, & 0.52, \n 0.46, X 0.45, A 0.34, i 0.29
- `e` -> u 0.52, x 0.44, h 0.24, d 0.21, w 0.20, m 0.19
- `t` -> T 0.66, w 0.35, W 0.35, s 0.33, g 0.33, y 0.32
- `a` -> i 0.59, A 0.55, u 0.51, O 0.50, o 0.44, b 0.38
- `o` -> O 0.50, q 0.47, i 0.45, a 0.44, I 0.34, j 0.29
- `T` -> t 0.66, W 0.58, K 0.52, c 0.41, S 0.36, V 0.36
- `q` -> Q 0.55, B 0.53, J 0.53, o 0.47, g 0.43, X 0.41
- `z` -> v 0.80, H 0.67, Z 0.65, M 0.56, m 0.53, k 0.52
- `.` -> ? 0.96, ! 0.92, ; 0.73, 3 0.67, : 0.64, , 0.62
- `\n` -> - 0.57, & 0.54, _ 0.46, U 0.44, E 0.39, ' 0.31
