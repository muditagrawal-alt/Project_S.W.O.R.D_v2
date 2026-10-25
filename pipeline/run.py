"""
Run the full S.W.O.R.D. pipeline (Phases 1, 2-slot, 3, 4, 5) on a video and store research outputs.

    python -m pipeline.run --video clip.mp4 --out-dir research/runs/<id> [--gt clip_gt.json] [--save-video]

Outputs in --out-dir:
    frames.json          sword.frames.v1 records + Phase 4 actions + Phase 5 threat state per frame
    fusion_events.jsonl  level changes (track and scene)
    threat_timeline.csv  per frame: time, scene level, people, weapons, max raw weapon score, GT flag
    annotated.mp4        (--save-video) boxes, skeleton joints, actions, threat level colours
    summary.json         sources, config, timings per stage, level occupancy, counts
    evaluation.json      (--gt) baseline ladder B0..B4: alarms, event recall, detection delay, false alarms/min
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from fusion.evaluate import evaluate_alarms, ladder_details, segments, verified_metrics
from fusion.rules import LEVELS, FusionConfig
from pipeline.system import SwordSystem
from tracking.schema import SCHEMA_VERSION

LEVEL_BGR = {"NORMAL": (60, 180, 60), "ELEVATED": (0, 215, 255), "SUSPICIOUS_ACTIVITY": (0, 140, 255),
             "WEAPON_RELATED_EVENT": (0, 0, 255)}


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def draw(frame: np.ndarray, rec: dict) -> np.ndarray:
    out = frame.copy()
    th = rec["threat"]
    for p in rec["persons"]:
        lv = th["tracks"].get(p["track_id"], {}).get("level", "NORMAL")
        col = LEVEL_BGR[lv]
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        act = p.get("action") or {}
        label = f"P{p['track_id']} {lv.split('_')[0]}"
        if act:
            label += f" | {act['top1_name'][:22]} {act['top1_prob']:.2f}"
        cv2.putText(out, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        for (x, y), s in zip(p["keypoints"], p["scores"]):
            if s >= 0.35:
                cv2.circle(out, (int(x), int(y)), 2, (255, 255, 0), -1)
    for w in rec["weapons"]:
        x1, y1, x2, y2 = map(int, w["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        owner = f"->P{w['person_id']}" if w["person_id"] is not None else ""
        cv2.putText(out, f"{w['cls_name']} {w['score']:.2f}{owner}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 255), 1, cv2.LINE_AA)
    lv = th["scene_level"]
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), LEVEL_BGR[lv], -1)
    cv2.putText(out, f"S.W.O.R.D. | t={rec['timestamp']:.1f}s | scene: {lv}", (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if lv != "ELEVATED" else (0, 0, 0), 2, cv2.LINE_AA)
    return out


def load_gt(path: Optional[str], n_frames: int, merge_gap_s: float):
    if not path:
        return None, None
    gt = json.loads(Path(path).read_text())
    by_idx = {f["frame_idx"]: f for f in gt["frames"]}
    times = [by_idx.get(i, {}).get("timestamp", i / gt["fps"]) for i in range(n_frames)]
    flags = [bool(by_idx.get(i, {}).get("boxes")) for i in range(n_frames)]
    return flags, segments(times, flags, merge_gap_s)


class ImageSequence:
    """Image-sequence source with explicit timestamps: JSON {"fps", "frames": [{"image", "timestamp", ...}]}.

    Used for frame datasets (e.g. the 2-fps mock-attack CCTV set): original JPEGs are fed without re-encoding and
    gaps in the frame numbering keep their real duration. The same file can serve as --gt (per-frame "boxes").
    """

    def __init__(self, path: str):
        doc = json.loads(Path(path).read_text())
        base = Path(path).resolve().parent
        self.items = [(str(Path(f["image"]) if Path(f["image"]).is_absolute() else base / f["image"]), float(f["timestamp"]))
                      for f in doc["frames"]]
        self.fps = float(doc["fps"])
        self.i, self.last_t = 0, None
        first = cv2.imread(self.items[0][0]) if self.items else None
        self.h, self.w = first.shape[:2] if first is not None else (0, 0)

    def isOpened(self) -> bool:
        return bool(self.items) and self.w > 0

    def read(self):
        if self.i >= len(self.items):
            return False, None
        path, self.last_t = self.items[self.i]
        self.i += 1
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"cannot read {path}")
        return True, img

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: self.w, cv2.CAP_PROP_FRAME_HEIGHT: self.h,
                cv2.CAP_PROP_FRAME_COUNT: len(self.items)}.get(prop, 0)

    def release(self) -> None:
        pass


def evaluate_frames(frames: list, frame_shape, fps: float, gt: Optional[str], cfg: FusionConfig, conf: float = 0.5) -> dict:
    """Baseline ladder + alarm metrics for one recorded run. Pure function of the recording, so stored runs can be
    re-scored with the current code (research/tools/integration/run_full_pipeline.py --reevaluate)."""
    merge_gap_s = max(1.0, 2.5 / fps)  # same merging for ground truth and alarms (1.25 s at 2 fps)
    gt_flags, gt_segs = load_gt(gt, len(frames), merge_gap_s=merge_gap_s)
    details = ladder_details(frames, frame_shape, conf=conf, cfg=cfg)
    times = [r["timestamp"] for r in frames]
    evaluation = {name: evaluate_alarms(times, d["flags"], gt_segs, gt_flags, merge_gap_s=merge_gap_s)
                  for name, d in details.items()}
    if gt_segs is not None:
        by_idx = {f["frame_idx"]: f for f in json.loads(Path(gt).read_text())["frames"]}
        gt_boxes = [[b["bbox"] for b in by_idx.get(i, {}).get("boxes", [])] for i in range(len(frames))]
        for name, v in verified_metrics(frames, details, gt_boxes, gt_segs, conf=conf, merge_gap_s=merge_gap_s).items():
            evaluation[name].update(v)
        evaluation["_ground_truth"] = {"events": len(gt_segs), "weapon_frames": int(sum(gt_flags)),
                                       "segments_s": [[round(a, 2), round(b, 2)] for a, b in gt_segs]}
    return evaluation


def open_source(source: str):
    """A video file, an image-sequence JSON, a camera index ("0") or a stream URL (rtsp://, http://)."""
    if str(source).endswith(".json"):
        return ImageSequence(str(source))
    return cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))


def run_video(video: str, out_dir: Path, system: Optional[SwordSystem] = None, gt: Optional[str] = None,
              save_video: bool = False, max_frames: Optional[int] = None, fps_override: Optional[float] = None,
              on_frame: Optional[Callable[[dict, np.ndarray], None]] = None, system_kwargs: Optional[dict] = None,
              conf: float = 0.5, live: bool = False, keep_frames: bool = True,
              should_stop: Optional[Callable[[], bool]] = None,
              make_system: Optional[Callable[[float], SwordSystem]] = None, frame_stride: int = 1,
              video_id: Optional[str] = None) -> dict:
    """live=True: timestamps from the wall clock (cameras/streams) instead of frame_idx / fps.
    keep_frames=False: do not hold full per-frame records (long live runs); frames.json and the baseline ladder are
    then skipped, the timeline and summary are still written.
    make_system(fps): builds the system once the source fps is known (used by the Phase 6 service).
    frame_stride=k: process every k-th frame of a video (lossless temporal subsampling; fps becomes fps / k).
    video_id: identifier stored as frames.json "video" (default: the file name; needed when names repeat)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = open_source(video)
    if not cap.isOpened():
        raise FileNotFoundError(video)
    src_fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = src_fps / max(1, frame_stride)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    system = system or (make_system(fps) if make_system else SwordSystem(fps=fps, **(system_kwargs or {})))
    system.reset()
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if save_video else None
    frames, rows, events, t0, idx = [], [], [], time.time(), 0
    t_live0 = time.monotonic()
    src_idx = -1
    while True:
        if should_stop is not None and should_stop():
            break
        ok, frame = cap.read()
        src_idx += 1
        if not ok or (max_frames and idx >= max_frames):
            break
        if frame_stride > 1 and src_idx % frame_stride:
            continue
        if live:
            t = time.monotonic() - t_live0
        elif isinstance(cap, ImageSequence):
            t = cap.last_t
        else:
            t = src_idx / src_fps
        rec = system.process(frame, idx, t)
        if keep_frames:
            frames.append(rec)
        rows.append((rec["frame_idx"], rec["timestamp"], rec["threat"]["scene_level"], len(rec["persons"]),
                     len(rec["weapons"]), max([d["score"] for d in rec["weapon_detections"]] + [0.0])))
        events.extend(rec["fusion_events"])
        if writer is not None:
            writer.write(draw(frame, rec))
        if on_frame is not None:
            on_frame(rec, frame)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    wall = time.time() - t0
    if keep_frames:
        doc = {"schema": SCHEMA_VERSION, "video": video_id or Path(str(video)).name, "fps": float(fps), "width": w, "height": h,
               "sources": system.sources, "frames": frames}
        (out_dir / "frames.json").write_text(json.dumps(doc, default=_json_default))
    with open(out_dir / "fusion_events.jsonl", "w") as fp:
        for e in events:
            fp.write(json.dumps(e, default=_json_default) + "\n")
    merge_gap_s = max(1.0, 2.5 / fps)  # same merging for ground truth and alarms (1.25 s at 2 fps)
    gt_flags, gt_segs = load_gt(gt, len(rows), merge_gap_s=merge_gap_s)
    with open(out_dir / "threat_timeline.csv", "w", newline="") as fp:
        wr = csv.writer(fp)
        wr.writerow(["frame_idx", "t", "scene_level", "n_persons", "n_weapon_tracks", "max_raw_weapon_score", "gt_weapon"])
        for i, r in enumerate(rows):
            wr.writerow(list(r) + ["" if gt_flags is None else int(gt_flags[i])])
    levels = Counter(r[2] for r in rows)
    summary = {"video": str(video), "frames": len(rows), "fps": fps, "live": live, "wall_seconds": round(wall, 1),
               "sources": system.sources, "fusion_config": asdict(system.p5.cfg), "timing_ms": system.timing_summary(),
               "scene_level_share": {lv: round(levels.get(lv, 0) / max(len(rows), 1), 4) for lv in LEVELS},
               "person_tracks": len({p["track_id"] for r in frames for p in r["persons"]}) if keep_frames else None,
               "action_updates": sum(len(r["actions"]) for r in frames) if keep_frames else None,
               "fusion_events": len(events)}
    evaluation = None
    if keep_frames and frames:
        evaluation = evaluate_frames(frames, (h, w), fps, gt, system.p5.cfg, conf)
        (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        summary["evaluation_file"] = "evaluation.json"
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    return {"summary": summary, "evaluation": evaluation}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file, image-sequence JSON, camera index or stream URL")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--fps", type=float, help="override the container fps (e.g. 2 for image sequences sampled at 2 fps)")
    ap.add_argument("--weapon-model", default=None)
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--weapon-imgsz", type=int, default=640, help="weapon detector input size (640 = training size)")
    ap.add_argument("--stride", type=int, default=1, help="process every k-th frame (temporal subsampling)")
    ap.add_argument("--no-pose", action="store_true")
    ap.add_argument("--no-action", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    kw = {"use_ensemble": args.ensemble, "use_pose": not args.no_pose, "use_action": not args.no_action, "device": args.device,
          "weapon_imgsz": args.weapon_imgsz}
    if args.weapon_model:
        kw["weapon_model"] = args.weapon_model
    res = run_video(args.video, Path(args.out_dir), gt=args.gt, save_video=args.save_video, max_frames=args.max_frames,
                    fps_override=args.fps, system_kwargs=kw, frame_stride=args.stride)
    s = res["summary"]
    print(json.dumps({k: s[k] for k in ("frames", "wall_seconds", "person_tracks", "action_updates", "fusion_events",
                                        "scene_level_share")}, indent=1))
    print("throughput fps:", s["timing_ms"].get("throughput_fps"))


if __name__ == "__main__":
    main()
