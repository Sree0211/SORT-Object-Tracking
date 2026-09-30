"""Load MOT17 detections and ground truth into per-frame dictionaries."""

import os

def load_detections(det_path):
    """Read det.txt and return {frame_id: [[x_min, y_min, x_max, y_max, conf], ...]}.

    MOT format per line: frame, id, x, y, w, h, conf, ...
    SORT works with corner boxes, so (x, y, w, h) is converted to (x_min, y_min, x_max, y_max).
    """
    frame_dict = {}
    with open(det_path, "r") as f:
        for line in f:
            val = line.strip().split(",")
            if len(val) < 7:
                continue

            frame_id = int(float(val[0]))
            x_min = float(val[2])
            y_min = float(val[3])
            w = float(val[4])
            h = float(val[5])
            conf = float(val[6])

            # For SORT
            x_max = x_min + w
            y_max = y_min + h

            frame_dict.setdefault(frame_id, []).append([x_min, y_min, x_max, y_max, conf])
    return frame_dict


def load_ground_truth(gt_path):
    """Read gt.txt and return {frame_id: [[gt_id, x_min, y_min, x_max, y_max], ...]}.

    MOT17 gt format: frame, id, x, y, w, h, consider_flag, class, visibility
    Only pedestrians (class 1) that are marked to be considered (flag 1) are kept.
    """
    gt_dict = {}
    with open(gt_path, "r") as f:
        for line in f:
            vals = line.strip().split(",")
            if len(vals) < 8:
                continue

            frame = int(float(vals[0]))
            gt_id = int(float(vals[1]))

            x = float(vals[2])
            y = float(vals[3])
            w = float(vals[4])
            h = float(vals[5])

            consider = int(float(vals[6]))
            cls = int(float(vals[7]))

            if consider != 1 or cls != 1:
                continue

            gt_dict.setdefault(frame, []).append([gt_id, x, y, x + w, y + h])
    return gt_dict


def list_images(img_folder):
    """Return the sorted list of image paths (000001.jpg, 000002.jpg, ...)."""
    names = sorted(n for n in os.listdir(img_folder) if n.lower().endswith((".jpg", ".png")))
    return [os.path.join(img_folder, n) for n in names]
