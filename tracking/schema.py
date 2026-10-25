"""
Canonical per-frame record shared by every S.W.O.R.D. phase.

It is a superset of the Phase 4 "tracking & pose" schema (video, fps, width, height, frames[].persons[] with
track_id / bbox / keypoints[17] / scores[17]), so Phase 4's clip builder reads it unchanged. Phase 1 adds
`weapons`, Phase 2 adds `trajectory` and zone `events`, Phase 3 adds `kinematics`; Phase 5 consumes all of it.

{
  "schema": "sword.frames.v1",
  "video": "cam01.mp4", "fps": 25.0, "width": 1280, "height": 720,
  "sources": {"person_detector": "...", "weapon_detector": "...", "tracker": "...", "pose": "..."},
  "zones": [{"name": "entrance", "polygon": [[x, y], ...]}],
  "homography": null | 3x3 image->ground (metres),
  "frames": [{
      "frame_idx": 0, "timestamp": 0.0,
      "persons": [{"track_id": 3, "bbox": [x1, y1, x2, y2], "score": 0.91,
                   "keypoints": [[x, y] x 17], "scores": [s x 17],     # COCO-17; zeros if pose disabled
                   "trajectory": {...}, "kinematics": {...} | null}],
      "weapons": [{"track_id": 7 | null, "bbox": [...], "cls": 0, "cls_name": "Pistol", "score": 0.84,   # every detection
                   "person_id": 3 | null, "hand": "right" | "left" | null, "assoc_dist_norm": 0.21 | null}],
      "events": [{"type": "zone_enter" | "zone_exit", "track_id": 3, "zone": "entrance"}]
  }]
}
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "sword.frames.v1"
N_KEYPOINTS = 17


class SchemaError(ValueError):
    pass


def empty_pose() -> tuple[list, list]:
    return [[0.0, 0.0] for _ in range(N_KEYPOINTS)], [0.0] * N_KEYPOINTS


def validate(doc: dict[str, Any]) -> None:
    """Raise SchemaError if `doc` is not a valid sword.frames.v1 document (also enforces the Phase 4 rules)."""
    def need(cond, msg):
        if not cond:
            raise SchemaError(msg)

    need(isinstance(doc, dict), "root must be an object")
    for k in ("video", "fps", "width", "height", "frames"):
        need(k in doc, f"missing top-level key '{k}'")
    need(isinstance(doc["fps"], (int, float)) and doc["fps"] > 0, "fps must be > 0")
    need(isinstance(doc["width"], int) and doc["width"] > 0 and isinstance(doc["height"], int) and doc["height"] > 0,
         "width/height must be positive integers")
    last = -1
    for f in doc["frames"]:
        need(isinstance(f.get("frame_idx"), int) and f["frame_idx"] > last, "frame_idx must be increasing integers")
        last = f["frame_idx"]
        need(isinstance(f.get("persons"), list), f"frame {last}: persons must be a list")
        for p in f["persons"]:
            need(isinstance(p.get("track_id"), int), f"frame {last}: person track_id must be int")
            need(isinstance(p.get("bbox"), list) and len(p["bbox"]) == 4, f"frame {last}: bbox must have 4 numbers")
            need(isinstance(p.get("keypoints"), list) and len(p["keypoints"]) == N_KEYPOINTS
                 and all(isinstance(k, list) and len(k) == 2 for k in p["keypoints"]),
                 f"frame {last}: keypoints must be 17 [x, y] pairs")
            need(isinstance(p.get("scores"), list) and len(p["scores"]) == N_KEYPOINTS, f"frame {last}: scores must have 17 values")
        for w in f.get("weapons", []):
            need((w.get("track_id") is None or isinstance(w.get("track_id"), int)) and isinstance(w.get("cls"), int),
                 f"frame {last}: weapon track_id must be int or null, cls int")
            need(isinstance(w.get("bbox"), list) and len(w["bbox"]) == 4, f"frame {last}: weapon bbox must have 4 numbers")
