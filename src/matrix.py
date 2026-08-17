"""Matrix helpers, pure Python. No numpy.

A matrix is a list of rows, each row a list of floats. Shape (r, c) means
len(m) == r and len(m[0]) == c. A vector is just a flat list.

Everything in this project is 2D. Batches are handled by looping over
sequences in train.py, not by adding a third dimension, so the shapes stay
easy to follow: (T, C) means T positions, C numbers per position.
"""

import math
import random


def zeros(rows, cols):
    """Shape (rows, cols), all 0.0.

    Written as a comprehension on purpose. [[0.0] * cols] * rows would make
    `rows` references to ONE row, so writing to m[0][0] would change every row.
    """
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def randn(rows, cols, scale, rng):
    """Shape (rows, cols) of gaussian numbers, mean 0, std `scale`.

    scale matters. Too big and the values blow up as they pass through layers,
    too small and the signal dies out. Usual choice is 1/sqrt(fan_in), which
    keeps the size of a dot product roughly the same as the size of its inputs.
    """
    return [[rng.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]


def transpose(a):
    """(r, c) -> (c, r)."""
    rows = len(a)
    cols = len(a[0])
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matmul(a, b):
    """(n, k) x (k, m) -> (n, m).

    out[i][j] is the dot product of row i of a with column j of b. That is the
    whole definition of matrix multiply: every output cell is one dot product.

    Cost is n*k*m multiply-adds. This is where all the time in the project goes,
    so b is read column-by-column via a transpose first (row access on a list of
    lists is much cheaper than jumping down a column).
    """
    n = len(a)
    k = len(a[0])
    m = len(b[0])
    if len(b) != k:
        raise ValueError("shape mismatch: (%d,%d) x (%d,%d)" % (n, k, len(b), m))

    bt = transpose(b)  # (m, k), so each column of b is now a contiguous row
    out = zeros(n, m)
    for i in range(n):
        arow = a[i]
        orow = out[i]
        for j in range(m):
            bcol = bt[j]
            total = 0.0
            for p in range(k):
                total += arow[p] * bcol[p]
            orow[j] = total
    return out


def add(a, b):
    """Elementwise a + b, same shape in and out."""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def add_row(a, row):
    """Add a vector of length c to every row of (r, c). Used for biases."""
    return [[a[i][j] + row[j] for j in range(len(row))] for i in range(len(a))]


def scale(a, s):
    """Multiply every element by the scalar s."""
    return [[v * s for v in row] for row in a]


def mul(a, b):
    """Elementwise a * b (not matrix multiply). Needed in backward passes."""
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def dot(u, v):
    """Dot product of two flat vectors.

    Geometrically: how much u and v point the same way, times their lengths.
    Big and positive means aligned, near zero means unrelated. This one number
    is what attention scores are made of.
    """
    total = 0.0
    for i in range(len(u)):
        total += u[i] * v[i]
    return total


def col_sum(a):
    """(r, c) -> flat list of length c, summing down each column.

    This is the backward pass of add_row: a bias was added to every row, so its
    gradient is the sum of the gradients of all the rows.
    """
    cols = len(a[0])
    out = [0.0] * cols
    for row in a:
        for j in range(cols):
            out[j] += row[j]
    return out


def shape(a):
    return (len(a), len(a[0]))
