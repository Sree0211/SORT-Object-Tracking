"""SORT: predict -> IoU cost -> Hungarian matching -> update / create / delete tracks."""

from .assignment import assign_tracks
from .iou import iou_matrix
from .track import Track


class Sort:
    def __init__(self, max_age=3, min_hits=3, iou_threshold=0.2, use_scratch_hungarian=False):
        self.max_age = max_age              # frames a track can go missing before removal
        self.min_hits = min_hits            # hits needed before a track is reported
        self.iou_threshold = iou_threshold  # minimum IoU to accept a match
        self.use_scratch_hungarian = use_scratch_hungarian
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0

    def update(self, detections):
        """Run one frame.

        detections: list of [x_min, y_min, x_max, y_max, (conf)]
        returns: list of {"track_id": id, "bbox": [x_min, y_min, x_max, y_max]}
        """
        self.frame_count += 1
        det_boxes = [d[:4] for d in detections]

        # 1. Kalman prediction for every existing track
        pred_boxes = [tr.predict() for tr in self.tracks]

        # 2. Cost calculation (1 - IoU)
        cost = 1 - iou_matrix(pred_boxes, det_boxes)

        # 3. Hungarian algorithm -> matches
        matches, unmatched_detections, unmatched_tracks = assign_tracks(
            cost, self.iou_threshold, self.use_scratch_hungarian
        )

        # 4. Track association
        for t, d in matches:
            self.tracks[t].update(det_boxes[d])

        # Unmatched tracks keep their prediction; time_since_update already grew in predict()

        # New track for every detection that did not match
        for d in unmatched_detections:
            self.tracks.append(Track(self.next_track_id, det_boxes[d]))
            self.next_track_id += 1

        # Remove tracks that have been missing for too long
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]

        # Report only tracks seen this frame and confirmed (or during the first frames)
        results = []
        for tr in self.tracks:
            if tr.time_since_update == 0 and (tr.hits >= self.min_hits or self.frame_count <= self.min_hits):
                results.append({"track_id": tr.id, "bbox": tr.get_bbox()})
        return results
