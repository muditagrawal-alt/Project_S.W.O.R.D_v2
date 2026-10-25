"""
Per-track trajectory features (Phase 2): position, velocity, acceleration, heading, dwell time, zone transitions.

Distances follow the roadmap's rule: a CCTV frame gives pixels, not metres. Each track is located by its
footpoint (bottom-centre of the person box). If a ground-plane homography (image -> metres) is supplied, the
footpoint is mapped through it and velocities are reported in m/s; otherwise only pixel units and a
scale-free version (body-heights per second, using the box height) are reported.

Velocity/acceleration are least-squares fits over a short time window rather than frame differences, which
removes most box jitter.
"""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np


def footpoint(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def apply_homography(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def fit_motion(t: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares velocity (linear fit) and acceleration (quadratic fit, needs >= 4 samples)."""
    tt = t - t[-1]
    vel = np.polyfit(tt, xy, 1)[0] if len(t) >= 2 else np.zeros(2)
    acc = 2.0 * np.polyfit(tt, xy, 2)[0] if len(t) >= 4 else np.zeros(2)
    return np.atleast_1d(vel), np.atleast_1d(acc)


class TrajectoryAnalyzer:
    """
    Args:
        zones: [{"name": str, "polygon": [[x, y], ...]}] in image pixels (footpoint is tested).
        homography: 3x3 image -> ground-plane (metres) matrix, or None.
        window_s: time window for the velocity fit; accel uses 2 × window_s.
        stale_s: forget tracks unseen for longer than this.
    """

    def __init__(self, zones: list[dict] | None = None, homography=None, window_s: float = 0.5, stale_s: float = 3.0):
        self.zones = [(z["name"], np.asarray(z["polygon"], dtype=np.float32)) for z in (zones or [])]
        self.H = None if homography is None else np.asarray(homography, dtype=np.float64)
        self.window_s, self.stale_s = window_s, stale_s
        self.hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=512))
        self.first_seen: dict[int, float] = {}
        self.zone_of: dict[int, str | None] = {}
        self.zone_since: dict[int, float] = {}
        self.transitions: dict[int, int] = defaultdict(int)

    def _zone(self, x: float, y: float) -> str | None:
        for name, poly in self.zones:
            if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                return name
        return None

    def update(self, t: float, tracks: list[dict]) -> tuple[dict[int, dict], list[dict]]:
        feats, events = {}, []
        for tr in tracks:
            tid = tr["track_id"]
            fx, fy = footpoint(tr["bbox"])
            h = max(1.0, tr["bbox"][3] - tr["bbox"][1])
            gx, gy = apply_homography(self.H, fx, fy) if self.H is not None else (np.nan, np.nan)
            self.hist[tid].append((t, fx, fy, h, gx, gy))
            self.first_seen.setdefault(tid, t)
            arr = np.asarray(self.hist[tid])
            win = arr[arr[:, 0] >= t - self.window_s]
            win2 = arr[arr[:, 0] >= t - 2 * self.window_s]
            v_px, _ = fit_motion(win[:, 0], win[:, 1:3])
            _, a_px = fit_motion(win2[:, 0], win2[:, 1:3])
            speed = float(np.hypot(*v_px))
            f = {"footpoint_px": [round(fx, 1), round(fy, 1)],
                 "velocity_px_s": [round(float(v_px[0]), 2), round(float(v_px[1]), 2)],
                 "speed_px_s": round(speed, 2),
                 "speed_heights_s": round(speed / float(np.median(win[:, 3])), 3),
                 "acceleration_px_s2": [round(float(a_px[0]), 2), round(float(a_px[1]), 2)],
                 "heading_deg": round(float(np.degrees(np.arctan2(v_px[1], v_px[0]))), 1) if speed > 1e-6 else None,
                 "dwell_s": round(t - self.first_seen[tid], 3),
                 "n_obs": int(len(arr))}
            if self.H is not None:
                v_m, _ = fit_motion(win[:, 0], win[:, 4:6])
                _, a_m = fit_motion(win2[:, 0], win2[:, 4:6])
                f.update(ground_m=[round(gx, 3), round(gy, 3)], speed_m_s=round(float(np.hypot(*v_m)), 3),
                         acceleration_m_s2=[round(float(a_m[0]), 3), round(float(a_m[1]), 3)])
            z = self._zone(fx, fy)
            prev = self.zone_of.get(tid, "__new__")
            if prev != z:
                if prev not in ("__new__", None):
                    events.append({"type": "zone_exit", "track_id": tid, "zone": prev, "timestamp": t})
                if z is not None:
                    events.append({"type": "zone_enter", "track_id": tid, "zone": z, "timestamp": t})
                if prev != "__new__":
                    self.transitions[tid] += 1
                self.zone_of[tid], self.zone_since[tid] = z, t
            f.update(zone=z, time_in_zone_s=round(t - self.zone_since[tid], 3), zone_transitions=self.transitions[tid])
            feats[tid] = f
        for tid in [k for k, d in self.hist.items() if d and t - d[-1][0] > self.stale_s]:
            for store in (self.hist, self.first_seen, self.zone_of, self.zone_since, self.transitions):
                store.pop(tid, None)
        return feats, events
