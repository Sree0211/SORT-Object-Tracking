"""Run SORT on a MOT17 sequence, evaluate it against ground truth and save the results.

Example:
    python run_sort.py --seq data/MOT17/train/MOT17-02-FRCNN --video
"""

import argparse
import os

import cv2

from sort import Sort
from sort.data import list_images, load_detections, load_ground_truth
from sort.metrics import evaluate, print_summary
from sort.visualize import draw_boxes, plot_errors


def parse_args():
    parser = argparse.ArgumentParser(description="SORT tracker on MOT17")
    parser.add_argument("--seq", required=True, help="MOT17 sequence folder (has img1/, det/, gt/)")
    parser.add_argument("--out", default="output", help="folder for results")
    parser.add_argument("--max-frames", type=int, default=None, help="only track the first N frames")
    parser.add_argument("--max-age", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--iou-threshold", type=float, default=0.2)
    parser.add_argument("--scratch-hungarian", action="store_true",
                        help="use the from-scratch Hungarian algorithm instead of scipy")
    parser.add_argument("--video", action="store_true", help="save a video with the tracked boxes")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    seq_name = os.path.basename(os.path.normpath(args.seq))

    # Load the detections (and ground truth if the sequence has it)
    detections = load_detections(os.path.join(args.seq, "det", "det.txt"))
    gt_path = os.path.join(args.seq, "gt", "gt.txt")
    gt_dict = load_ground_truth(gt_path) if os.path.exists(gt_path) else None

    last_frame = max(detections) if detections else 0
    if args.max_frames:
        last_frame = min(last_frame, args.max_frames)

    tracker = Sort(args.max_age, args.min_hits, args.iou_threshold, args.scratch_hungarian)

    # Track every frame
    tracking_results = {}
    for frame_id in range(1, last_frame + 1):
        tracking_results[frame_id] = tracker.update(detections.get(frame_id, []))

    # Save results in MOT format: frame, id, x, y, w, h, conf, -1, -1, -1
    result_file = os.path.join(args.out, f"{seq_name}.txt")
    with open(result_file, "w") as f:
        for frame_id, tracks in tracking_results.items():
            for t in tracks:
                x1, y1, x2, y2 = t["bbox"]
                f.write(f"{frame_id},{t['track_id']},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1\n")
    print(f"Tracked {last_frame} frames, results saved to {result_file}")

    # Evaluate against ground truth
    if gt_dict:
        gt_dict = {f: v for f, v in gt_dict.items() if f <= last_frame}
        summary, frame_stats = evaluate(tracking_results, gt_dict)
        print_summary(summary)
        plot_path = os.path.join(args.out, f"{seq_name}_errors.png")
        plot_errors(frame_stats, plot_path)
        print(f"FP / FN plot saved to {plot_path}")

    # Plot the predicted boxes and track IDs on each frame
    if args.video:
        images = list_images(os.path.join(args.seq, "img1"))[:last_frame]
        first = cv2.imread(images[0])
        h, w = first.shape[:2]
        video_path = os.path.join(args.out, f"{seq_name}.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

        for frame_id, img_path in enumerate(images, start=1):
            img = cv2.imread(img_path)
            tracks = tracking_results.get(frame_id, [])
            img = draw_boxes(img, [t["bbox"] for t in tracks],
                             track_ids=[t["track_id"] for t in tracks], label_prefix="ID:")
            writer.write(img)
        writer.release()
        print(f"Video saved to {video_path}")


if __name__ == "__main__":
    main()
