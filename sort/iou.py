"""IoU between boxes and the IoU matrix used as the matching cost."""

import numpy as np

def iou(box_1, box_2):
    """IoU of two boxes given as [x_min, y_min, x_max, y_max]."""
    # Intersection rectangle
    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])

    w_int = max(0.0, x_2 - x_1)
    h_int = max(0.0, y_2 - y_1)
    area_int = w_int * h_int

    area_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
    area_2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])
    area_union = area_1 + area_2 - area_int

    if area_union <= 0:
        return 0.0
    return area_int / area_union


def iou_matrix(boxes_1, boxes_2):
    """IoU for every pair: rows = boxes_1 (tracks), cols = boxes_2 (detections)."""
    result = np.zeros((len(boxes_1), len(boxes_2)))
    for i, box1 in enumerate(boxes_1):
        for j, box2 in enumerate(boxes_2):
            result[i, j] = iou(box1, box2)
    return result
