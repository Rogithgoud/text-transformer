"""Checks matmul against numbers I worked out on paper, plus a few identities."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matrix as M


def test_matmul_by_hand():
    # A is (2,3), B is (3,2), so the result must be (2,2).
    a = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0],
         [9.0, 10.0],
         [11.0, 12.0]]
    # out[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    # out[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    # out[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    # out[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    out = M.matmul(a, b)
    assert M.shape(out) == (2, 2), M.shape(out)
    assert out == [[58.0, 64.0], [139.0, 154.0]], out
    print("matmul matches the hand calculation")


def test_transpose_identity():
    # (AB)^T == B^T A^T
    a = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]      # (3,2)
    b = [[1.0, 0.0, 2.0], [-1.0, 3.0, 1.0]]       # (2,3)
    left = M.transpose(M.matmul(a, b))
    right = M.matmul(M.transpose(b), M.transpose(a))
    assert left == right, (left, right)
    print("(AB)^T == B^T A^T holds")


def test_shape_mismatch_raises():
    try:
        M.matmul([[1.0, 2.0]], [[1.0, 2.0]])
    except ValueError:
        print("shape mismatch raises, good")
        return
    raise AssertionError("expected a ValueError")


def test_zeros_rows_are_independent():
    # the [[0]*c]*r trap: writing one cell must not change the others
    z = M.zeros(3, 2)
    z[0][0] = 9.0
    assert z[1][0] == 0.0 and z[2][0] == 0.0, z
    print("zeros() rows are separate lists")


def test_col_sum_is_add_row_backward():
    a = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert M.col_sum(a) == [9.0, 12.0]
    print("col_sum sums down columns")


if __name__ == "__main__":
    test_matmul_by_hand()
    test_transpose_identity()
    test_shape_mismatch_raises()
    test_zeros_rows_are_independent()
    test_col_sum_is_add_row_backward()
    print("all matrix tests passed")
