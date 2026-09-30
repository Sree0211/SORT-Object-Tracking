"""Drawing helpers: boxes with track IDs, and an FP / FN plot."""

import cv2


def draw_boxes(img, boxes, color=(0, 255, 0), track_ids=None, label_prefix=""):
    """Draw boxes (and optional track IDs) on a copy of the image. Colors are BGR."""
    img = img.copy()
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        if track_ids is not None:
            text = f"{label_prefix}{track_ids[i]}"
            cv2.putText(img, text, (x_min, max(y_min - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def plot_errors(frame_stats, out_path):
    """Save a plot of false positives and false negatives per frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [s["frame"] for s in frame_stats]
    plt.figure(figsize=(10, 4))
    plt.plot(frames, [s["fp"] for s in frame_stats], label="False positives")
    plt.plot(frames, [s["fn"] for s in frame_stats], label="False negatives")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
