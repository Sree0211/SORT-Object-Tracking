"""CLEAR MOT metrics: MOTA, MOTP and ID switches.

MOTP = sum(IoU of all matches) / total matches
MOTA = 1 - (FN + FP + ID switches) / total ground truth boxes
"""

import numpy as np

from .assignment import assign_tracks
from .iou import iou_matrix


def evaluate(tracking_results, gt_dict, iou_threshold=0.5):
    """Compare predicted tracks with ground truth, frame by frame."""
    total_tp = total_fp = total_fn = total_ids = 0
    total_iou = 0.0
    total_gt = 0
    last_match = {}   # gt_id -> pred_id it was last matched to
    frame_stats = []

    frames = sorted(set(tracking_results) | set(gt_dict))
    for frame in frames:
        preds = tracking_results.get(frame, [])
        gts = gt_dict.get(frame, [])

        pred_ids = [p["track_id"] for p in preds]
        pred_boxes = [p["bbox"] for p in preds]
        gt_ids = [g[0] for g in gts]
        gt_boxes = [g[1:5] for g in gts]

        # Frame matching with prediction and ground truth
        ious = iou_matrix(pred_boxes, gt_boxes)
        matches, unmatched_gt, unmatched_pred = assign_tracks(1 - ious, iou_threshold)

        ids = 0
        for p, g in matches:
            gt_id, pred_id = gt_ids[g], pred_ids[p]
            total_iou += ious[p, g]
            # ID switch: this gt object was followed by a different track before
            if gt_id in last_match and last_match[gt_id] != pred_id:
                ids += 1
            last_match[gt_id] = pred_id

        tp, fp, fn = len(matches), len(unmatched_pred), len(unmatched_gt)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_ids += ids
        total_gt += len(gts)
        frame_stats.append({"frame": frame, "tp": tp, "fp": fp, "fn": fn, "ids": ids})

    motp = total_iou / total_tp if total_tp else 0.0
    mota = 1 - (total_fn + total_fp + total_ids) / total_gt if total_gt else 0.0

    summary = {
        "MOTA": mota,
        "MOTP": motp,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "ID switches": total_ids,
        "GT boxes": total_gt,
    }
    return summary, frame_stats


def print_summary(summary):
    print("-" * 30)
    for key, value in summary.items():
        if isinstance(value, (float, np.floating)):
            print(f"{key:<12} {value:.4f}")
        else:
            print(f"{key:<12} {value}")
    print("-" * 30)
