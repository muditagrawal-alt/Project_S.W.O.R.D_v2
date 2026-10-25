"""
Weapon ↔ person association (links Phase 1 detections to Phase 2/3 person tracks).

With pose: the weapon is attached to the person whose wrist is closest to the weapon box, measured in units of
that person's box height (scale-free). Without pose: the person box that contains the largest share of the
weapon box. Unassociated weapons (on a table, on the floor, or false alarms) keep person_id = None — Phase 5 can
use "unattended weapon" vs "weapon in hand" as different evidence.
"""

from __future__ import annotations

import numpy as np

L_WRIST, R_WRIST = 9, 10


def _point_box_distance(px: float, py: float, box) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return float(np.hypot(dx, dy))


def _containment(inner, outer) -> float:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def associate(weapons: list[dict], persons: list[dict], kpt_thr: float = 0.3, max_wrist_dist: float = 0.35,
              min_containment: float = 0.5) -> list[dict]:
    """Return one dict per weapon: {person_id, hand, assoc_dist_norm, method}."""
    out = []
    for w in weapons:
        candidates = []  # (priority, cost, person_id, hand, dist) — wrist evidence (0) beats box overlap (1)
        for p in persons:
            h = max(1.0, p["bbox"][3] - p["bbox"][1])
            kp, sc = p.get("keypoints"), p.get("scores")
            wrists = [(i, hand) for i, hand in ((L_WRIST, "left"), (R_WRIST, "right"))
                      if kp is not None and sc is not None and sc[i] >= kpt_thr]
            if wrists:
                d, hand = min((_point_box_distance(kp[i][0], kp[i][1], w["bbox"]) / h, hand) for i, hand in wrists)
                if d <= max_wrist_dist:
                    candidates.append((0, d, p["track_id"], hand, round(d, 3)))
            else:
                c = _containment(w["bbox"], p["bbox"])
                if c >= min_containment:
                    candidates.append((1, -c, p["track_id"], None, None))
        if candidates:
            prio, _, pid, hand, d = min(candidates)
            out.append({"person_id": pid, "hand": hand, "assoc_dist_norm": d, "method": "wrist" if prio == 0 else "box"})
        else:
            out.append({"person_id": None, "hand": None, "assoc_dist_norm": None, "method": None})
    return out
