
from typing import List, Optional


Matrix = List[List[Optional[int]]]


class InconsistentError(Exception):
    """Raised when the matrix cannot be completed under the constraints."""
    pass


def repair_matrix(mat: Matrix) -> Matrix:
    """
    Fill an N x 6 matrix of door destinations with None for unknowns so that:
      - For every i != j, the number of doors i -> j equals the number of doors j -> i.
      - Each row has exactly 6 destinations.
      - Self-links (i -> i) are allowed as single-door closures (no pairing required).

    Strategy:
      1) Enforce symmetry for all inter-room pairs (i != j) implied by known entries:
         if a[i][j] > a[j][i], fill 'need' blanks in row j with 'i', and vice versa.
      2) Fill all remaining blanks with self-links (i -> i).
      3) Validate the result.

    Time complexity: O(N^2 + N * 6) ~ O(N^2) for an N x 6 matrix.
    """
    if not mat:
        return []

    N = len(mat)
    if any(len(row) != 6 for row in mat):
        raise ValueError("Each row must have exactly 6 entries")

    # Work on a copy
    M: Matrix = [row[:] for row in mat]

    def none_positions(i: int):
        return [k for k, v in enumerate(M[i]) if v is None]

    def counts():
        """Return (a, rem) where a[i][j] = #doors i->j, rem[i] = #None in row i."""
        a = [[0] * N for _ in range(N)]
        rem = [0] * N
        for i in range(N):
            for v in M[i]:
                if v is None:
                    rem[i] += 1
                else:
                    if not (0 <= v < N):
                        raise ValueError(f"Room index out of range: {v}")
                    a[i][v] += 1
        return a, rem

    # --- Step 1: enforce symmetry for i != j ---
    a, rem = counts()
    for i in range(N):
        for j in range(i + 1, N):
            if a[i][j] > a[j][i]:
                need = a[i][j] - a[j][i]
                pos = none_positions(j)
                if len(pos) < need:
                    raise InconsistentError(
                        f"Room {j} lacks {need - len(pos)} blanks to mirror {i}->{j}"
                    )
                for t in range(need):
                    M[j][pos[t]] = i
            elif a[j][i] > a[i][j]:
                need = a[j][i] - a[i][j]
                pos = none_positions(i)
                if len(pos) < need:
                    raise InconsistentError(
                        f"Room {i} lacks {need - len(pos)} blanks to mirror {j}->{i}"
                    )
                for t in range(need):
                    M[i][pos[t]] = j

    # --- Step 2: fill remaining blanks with self-links (i -> i) ---
    for i in range(N):
        for p in none_positions(i):
            M[i][p] = i

    # --- Step 3: validate ---
    a, rem = counts()
    if any(r != 0 for r in rem):
        raise InconsistentError("Not all doors filled")
    for i in range(N):
        for j in range(N):
            if i != j and a[i][j] != a[j][i]:
                raise InconsistentError(f"Inter-room symmetry failed between {i} and {j}")

    return M


# -------------------------
# Example usage:
if __name__ == "__main__":
    example = [
        [2, 4, None, None, 1, 9],
        [0, 12, None, 9, 2, 12],
        [12, 16, 0, 3, 2, 1],
        [2, 3, 6, 17, None, 3],
        [16, 0, 7, 14, 8, None],
        [10, 13, None, 13, 9, 14],
        [12, 7, 3, 14, 17, 10],
        [11, 6, None, None, 4, None],
        [None, None, None, 4, None, 16],
        [11, 0, 1, None, 16, 5],
        [15, None, 6, None, 5, 10],
        [None, 9, 7, None, None, None],
        [None, 2, 6, None, 1, 1],
        [None, None, 17, 5, 5, 14],
        [13, 5, 6, 17, None, 4],
        [None, 17, 10, None, 16, 15],
        [2, 9, None, 8, 4, 15],
        [14, 6, 3, 13, None, 15],
    ]

    filled = repair_matrix(example)
    counts = [0] * len(filled)
    for row in filled:
        for v in row:
            counts[v] += 1
    print(counts)

    print(filled)
    for i, row in enumerate(filled):
        print(i, row)
