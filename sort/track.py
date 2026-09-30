"""A single tracked object with its own Kalman filter.

State vector (from the SORT paper):
    x = [u, v, s, r, du, dv, ds]
    u, v   -> centre of the box
    s      -> area (scale) of the box
    r      -> aspect ratio (w / h), assumed constant
    du, dv, ds -> velocities
"""

import numpy as np


def bbox_to_z(bbox):
    """[x_min, y_min, x_max, y_max] -> measurement [u, v, s, r]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    u = bbox[0] + w / 2
    v = bbox[1] + h / 2
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([u, v, s, r], dtype=float)


def x_to_bbox(x):
    """State [u, v, s, r, ...] -> [x_min, y_min, x_max, y_max]."""
    u, v, s, r = x[:4]
    s = max(s, 0.0)
    r = max(r, 1e-6)
    width = np.sqrt(s * r)
    height = s / width if width > 0 else 0.0
    return [u - width / 2, v - height / 2, u + width / 2, v + height / 2]


class Track:
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # Kalman - state
        z = bbox_to_z(bbox)
        self.x = np.array([z[0], z[1], z[2], z[3], 0.0, 0.0, 0.0])

        # Kalman - Covariance matrix (uncertainty of the state estimate)
        # Velocities are unknown at the start, so they get a high uncertainty
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0, 1000.0])

        # Constant velocity motion model
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # We only observe [u, v, s, r]
        self.H = np.eye(4, 7)

        # Process noise covariance - uncertainty in the motion model
        self.Q = np.diag([1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 0.1])

        # Measurement noise - expected noise in the detections [u, v, s, r]
        self.R = np.diag([10.0, 10.0, 50.0, 1.0])

        self.nis = 0.0  # Normalised innovation squared (how surprised the filter was)

    def predict(self):
        """Move the state one frame forward and return the predicted box."""
        # Area cannot go negative, so stop shrinking if it would
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, bbox):
        """Correct the prediction with the matched detection."""
        z = bbox_to_z(bbox)

        y = z - self.H @ self.x                       # Measurement residual - how wrong the prediction is
        S = self.H @ self.P @ self.H.T + self.R       # Denoted by S in the research paper
        K = self.P @ self.H.T @ np.linalg.inv(S)      # Kalman gain - how much to trust the detection

        self.x = self.x + K @ y
        A = np.eye(7) - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T  # Joseph form, keeps P stable

        self.nis = float(y.T @ np.linalg.inv(S) @ y)
        self.hits += 1
        self.time_since_update = 0

    def get_bbox(self):
        return x_to_bbox(self.x)
