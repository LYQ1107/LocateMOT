"""OC-SORT-style constant-velocity Kalman wrapper (clean reimplementation).

Reference: OC-SORT (MIT) commit 8462e7e7, trackers/ocsort_tracker/ocsort.py and
kalmanfilter.py. This is a compact reimplementation of the same 7-dim constant
velocity model and observation-centric update, with attribution; no code copied
verbatim from the MIT original beyond the public algorithm definition.
"""
from __future__ import annotations

import numpy as np


def _bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return np.array([bbox[0] + w / 2, bbox[1] + h / 2, w * h, w / (h + 1e-6)]).reshape(4, 1)


def _z_to_bbox(z, score=None):
    z = np.asarray(z).ravel()
    w = np.sqrt(max(float(z[2] * z[3]), 1e-12))
    h = z[2] / w if w > 0 else 0.0
    b = np.array([z[0] - w / 2, z[1] - h / 2, z[0] + w / 2, z[1] + h / 2])
    return np.concatenate([b, [score]]) if score is not None else b


class KalmanBoxTracker7:
    """7-dim [x,y,s,r,vx,vy,vs] constant velocity Kalman filter."""

    def __init__(self, bbox):
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], dtype=np.float64)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)
        self.x = np.zeros((7, 1))
        self.P = np.eye(7) * 10.0
        self.R = np.eye(4)
        self.Q = np.eye(7)
        self.R[2:, 2:] *= 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x[:4] = _bbox_to_z(bbox)
        self.time_since_update = 0
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.observations = {}
        self.age = 0
        self.velocity = None
        self.hit_streak = 0
        self.hits = 0

    def predict(self):
        if (self.x[6] + self.x[2]) <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_bbox(self.x[:4])

    def update(self, bbox):
        if bbox is None:
            self.x = self.x
            return
        if self.last_observation.sum() >= 0:
            prev = None
            for dt in (3, 2, 1):
                if self.age - dt in self.observations:
                    prev = self.observations[self.age - dt]
                    break
            if prev is None:
                prev = self.last_observation
            dx = (bbox[2] + bbox[0]) / 2 - (prev[0] + prev[2]) / 2
            dy = (bbox[3] + bbox[1]) / 2 - (prev[1] + prev[3]) / 2
            norm = np.sqrt(dx * dx + dy * dy) + 1e-6
            self.velocity = np.array([dy / norm, dx / norm])
        self.last_observation = bbox
        self.observations[self.age] = bbox
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = _bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def state(self):
        return _z_to_bbox(self.x[:4])
