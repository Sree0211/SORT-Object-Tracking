"""Match tracks to detections with the Hungarian algorithm.

Two versions are available:
- hungarian(): written from scratch (for learning)
- scipy's linear_sum_assignment (the default, faster)
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian(cost_matrix):
    """Hungarian algorithm from scratch (O(n^3) potentials version).

    Returns (row_idx, col_idx) of the minimum cost assignment, like scipy does.
    A non-square matrix is padded to square with cost 1 (= IoU of 0).
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    rows, cols = cost_matrix.shape
    n = max(rows, cols)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    padded = np.full((n, n), 1.0)
    padded[:rows, :cols] = cost_matrix

    # Potentials for rows and columns (1-indexed, index 0 is a helper)
    row_pot = np.zeros(n + 1)
    col_pot = np.zeros(n + 1)
    col_row_map = np.zeros(n + 1, dtype=int)  # which row is assigned to each column
    way = np.zeros(n + 1, dtype=int)          # path used to update the matching

    for i in range(1, n + 1):
        col_row_map[0] = i
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)
        j0 = 0

        # Grow the path until a free column is found
        while True:
            used[j0] = True
            i0 = col_row_map[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = padded[i0 - 1, j - 1] - row_pot[i0] - col_pot[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    row_pot[col_row_map[j]] += delta
                    col_pot[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if col_row_map[j0] == 0:
                break

        # Flip the matching along the path
        while True:
            j1 = way[j0]
            col_row_map[j0] = col_row_map[j1]
            j0 = j1
            if j0 == 0:
                break

    # Keep only real (not padded) pairs
    row_idx, col_idx = [], []
    for j in range(1, n + 1):
        r, c = col_row_map[j] - 1, j - 1
        if r < rows and c < cols:
            row_idx.append(r)
            col_idx.append(c)

    order = np.argsort(row_idx)
    return np.array(row_idx, dtype=int)[order], np.array(col_idx, dtype=int)[order]


def assign_tracks(cost_matrix, iou_threshold=0.3, use_scratch=False):
    """Once the cost matrix (1 - IoU) is created, match track IDs with detections.

    Returns (matches, unmatched_detections, unmatched_tracks).
    Pairs with IoU below the threshold are rejected and treated as unmatched.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    rows, cols = cost_matrix.shape

    if rows == 0 or cols == 0:
        return [], list(range(cols)), list(range(rows))

    if use_scratch:
        row_idx, col_idx = hungarian(cost_matrix)
    else:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(rows))
    unmatched_detections = set(range(cols))

    for r, c in zip(row_idx, col_idx):
        iou = 1 - cost_matrix[r, c]
        if iou >= iou_threshold:
            matches.append((int(r), int(c)))
            unmatched_tracks.discard(int(r))
            unmatched_detections.discard(int(c))

    return matches, sorted(unmatched_detections), sorted(unmatched_tracks)
