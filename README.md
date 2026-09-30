# SORT – Simple Online and Realtime Tracking (from scratch)

A from-scratch Python implementation of the SORT multi-object tracker from the paper
[Simple Online and Realtime Tracking (Bewley et al., 2016)](https://arxiv.org/abs/1602.00763),
tested on the [MOT17](https://motchallenge.net/data/MOT17/) dataset.

I built this as a learning project to understand how the paper works step by step:
IoU matching, the Hungarian algorithm and the Kalman filter.

## How SORT works

For every frame:

1. **Predict**: each track's Kalman filter predicts where its box will be in this frame.
2. **Cost**: build a cost matrix of `1 - IoU` between predicted boxes and new detections.
3. **Match**: the Hungarian algorithm finds the best track ↔ detection pairs.
   Pairs with IoU below the threshold are rejected.
4. **Update**:
   - matched tracks are corrected with their detection (Kalman update)
   - unmatched detections start new tracks
   - tracks missing for more than `max_age` frames are removed

The Kalman state for each object is `x = [u, v, s, r, du, dv, ds]`
(box centre, area, aspect ratio and their velocities), with a constant velocity model.

## Project structure

```
sort/
  data.py         # load MOT17 detections and ground truth
  iou.py          # IoU and IoU matrix
  assignment.py   # Hungarian algorithm (from scratch + scipy)
  track.py        # one track with its Kalman filter
  tracker.py      # the SORT loop
  metrics.py      # MOTA, MOTP, ID switches
  visualize.py    # draw boxes and plots
run_sort.py       # run tracker + evaluation on a sequence
tests/            # small sanity checks
```

## Setup

```bash
pip install -r requirements.txt
# or install as a package
pip install -e .
```

Download MOT17 from [motchallenge.net](https://motchallenge.net/data/MOT17/) and place it under `data/`:

```
data/MOT17/train/MOT17-02-FRCNN/
  img1/000001.jpg ...
  det/det.txt
  gt/gt.txt
```

## Usage

```bash
python run_sort.py --seq data/MOT17/train/MOT17-02-FRCNN --video
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--max-age` | 3 | frames a track can be missing before it is deleted |
| `--min-hits` | 3 | detections needed before a track is reported |
| `--iou-threshold` | 0.2 | minimum IoU to accept a match |
| `--scratch-hungarian` | off | use my own Hungarian implementation instead of scipy |
| `--max-frames` | all | only track the first N frames |
| `--video` | off | save a video with boxes and track IDs |

Outputs in `output/`:
- `<sequence>.txt` - tracks in MOT format (works with the official evaluation tools)
- `<sequence>_errors.png` - false positives / false negatives per frame
- `<sequence>.mp4` - video with tracked boxes (with `--video`)

## Evaluation

The script prints CLEAR MOT metrics when ground truth is available:

- **MOTA** = 1 - (FN + FP + ID switches) / total ground truth boxes
- **MOTP** = average IoU of matched boxes (IoU >= 0.5)
- **ID switches** = times a ground truth object changes its predicted track ID

## Run tests

```bash
pytest
```

## What I learned

- Why IoU alone is not enough and a motion model (Kalman filter) is needed
- How the Hungarian algorithm solves the assignment problem, by writing it myself
- How the Kalman gain balances the prediction against the detection
- How tracking is evaluated with MOTA / MOTP
*Note: AI tools were used for better readability, comments and writing dummy test cases.*

## Results

Evaluated on MOT17 train - MOT17-02-FRCNN data.
Tracked 600 frames, results saved to output\MOT17-02-FRCNN.txt

| MOTA         |    0.2603
| MOTP         |   0.8801
| TP           |  6393
| FP           |  1443
| FN           |  12188
| ID switches  |  113
| GT boxes     | 18581

## Reference

Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016).
*Simple Online and Realtime Tracking*. ICIP 2016.
