"""Small sanity checks. Run with: pytest"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from sort import Sort, hungarian, iou


def test_iou_same_box_is_one():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_no_overlap_is_zero():
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # Overlap 50, union 150
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 50 / 150) < 1e-9


def test_scratch_hungarian_matches_scipy():
    rng = np.random.default_rng(0)
    for shape in [(5, 5), (4, 7), (7, 4)]:
        cost = rng.random(shape)
        r1, c1 = hungarian(cost)
        r2, c2 = linear_sum_assignment(cost)
        assert np.isclose(cost[r1, c1].sum(), cost[r2, c2].sum())


def test_tracker_keeps_the_same_id():
    # One box moving right by 5 pixels every frame should keep one ID
    tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.2)
    ids = set()
    for f in range(20):
        out = tracker.update([[100 + 5 * f, 100, 150 + 5 * f, 200, 1.0]])
        ids.update(t["track_id"] for t in out)
    assert ids == {1}
