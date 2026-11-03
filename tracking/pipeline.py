"""
S.W.O.R.D. connected pipeline: Phase 1 (weapons) → Phase 2 (tracking, trajectories, zones) → Phase 3 (pose)
→ shared per-frame record (tracking/schema.py), ready for Phase 4 clips and Phase 5 fusion.

Per frame:
    1. people:  COCO person detector (default yolo26n.pt) → ByteTrack → persistent track IDs
    2. pose:    RTMPose-m top-down on the TRACKED boxes (pose_pipeline.RTMPosePipeline.process_tracked)
    3. weapons: S.W.O.R.D. detector (one model live; --ensemble for the WBF ensemble) → ByteTrack
    4. trajectory features + zone events, weapon↔person association
    5. append to the JSON document (+ optional annotated video)

Usage:
    python -m tracking.pipeline --video clip.mp4 --out clip_frames.json
    python -m tracking.pipeline --video clip.mp4 --out clip.json --pose --zones zones.json --save-video clip_annotated.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.classes import CLASS_NAMES  # noqa: E402
from tracking.association import associate  # noqa: E402
from tracking.schema import SCHEMA_VERSION, empty_pose, validate  # noqa: E402
from tracking.tracker import ByteTrackWrapper  # noqa: E402

from tracking.interface import DEFAULT_PERSON_MODEL  # noqa: E402  (COCO-pretrained YOLO26n)
DEFAULT_WEAPON_MODEL = PROJECT_ROOT / "weights" / "yolo26s_best.pt"


def weapons_from_detections(weapon_trk: ByteTrackWrapper, wb, ws, wc, persons: list, src=None,
                            extra: list | None = None) -> tuple[list, list]:
    """Raw detections -> (raw records, weapons linked to persons). Shared by the live pipeline and offline
    replays/re-evaluation so both use the same tracking and association.

    Every detection is passed on (with its person link); `track_id` is the ByteTrack ID when a confirmed weapon
    track owns the detection, else None. Weapons are NOT filtered by track confirmation: at low frame rates a
    hand-held weapon moves too far between frames for IoU matching (2-fps CCTV: median IoU 0.08 between
    consecutive frames), and confirmation silently dropped ~90% of perfect detections. Temporal persistence is
    Phase 5's job, where it is explicit and measured.
    """
    wb, ws, wc = np.asarray(wb, dtype=float).reshape(-1, 4), np.asarray(ws, dtype=float).reshape(-1), np.asarray(wc).reshape(-1)
    raw = [{"bbox": [round(float(v), 1) for v in b], "cls": int(c), "score": round(float(s), 3)}
           for b, s, c in zip(wb, ws, wc)]
    if src is not None:   # detection origin: "full" frame or "zoom" crop (tracking/zoom.py)
        for r, o in zip(raw, src):
            r["src"] = o
    if extra is not None:  # e.g. det_score / verify from the second-stage verifier (tracking/verifier.py)
        for r, e in zip(raw, extra):
            r.update(e)
    tid_by_det = {t["det_idx"]: t["track_id"] for t in weapon_trk.update(wb, ws, wc)}
    dets = [{"bbox": r["bbox"], "det_idx": i} for i, r in enumerate(raw)]
    links = associate(dets, persons)
    weapons = [{"track_id": tid_by_det.get(i), "bbox": r["bbox"], "cls": r["cls"],
                "cls_name": CLASS_NAMES[r["cls"]] if 0 <= r["cls"] < len(CLASS_NAMES) else "?",
                "score": r["score"], "person_id": a["person_id"], "hand": a["hand"],
                "assoc_dist_norm": a["assoc_dist_norm"], "assoc_method": a["method"], **({"src": r["src"]} if "src" in r else {})}
               for i, (r, a) in enumerate(zip(raw, links))]
    return raw, weapons


def rederive_weapons(frames: list, fps: float, track_seconds: float = 1.5, zoom_min_score: float = 0.0,
                     max_area_frac: float = 1.0, zoom_min_window: int = 0, frame_shape=None,
                     zoom_modes: tuple = ("hands",), verify_mode: str = "none", verify_min: float = 0.0) -> None:
    """Recompute frames[i]["weapons"] from the recorded raw `weapon_detections` and persons with the CURRENT code
    (fresh weapon tracker, same settings as SwordPipeline). Lets stored runs be re-scored after a code change.
    The live pipeline's detection filters are applied to the recorded detections: zoom_min_score (zoom-crop
    detections below it), max_area_frac (boxes larger than this share of the frame; needs frame_shape) and
    zoom_min_window (zoom detections from windows smaller than this; needs frame_shape and the recorded zoom_modes).
    Filtered detections are removed from `weapon_detections` too: the live pipeline never records them, and the
    frame-level baselines (B0) read that list. verify_mode / verify_min: fuse the recorded detector score with the
    second-stage verifier's p(weapon) ("verify", written by research/tools/phase1/apply_weapon_verifier.py) and drop
    detections whose p(weapon) is below verify_min, as SwordPipeline does with a verifier."""
    from tracking.zoom import ZoomConfig, from_allowed_window

    trk = ByteTrackWrapper(track_buffer=max(5, int(round(fps * track_seconds))))
    zcfg = ZoomConfig(modes=tuple(zoom_modes), min_window=zoom_min_window)
    for f in frames:
        d = [x for x in f["weapon_detections"] if x.get("src") != "zoom" or x["score"] >= zoom_min_score]
        if verify_mode != "none" or verify_min > 0:
            from tracking.verifier import fuse_scores

            fused = []
            for x in d:
                if "verify" not in x:
                    fused.append(x)
                    continue
                if x["verify"] < verify_min:
                    continue
                base = x.get("det_score", x["score"])
                fused.append({**x, "det_score": base, "score": round(float(fuse_scores([base], [x["verify"]], verify_mode)[0]), 3)})
            d = fused
        if max_area_frac < 1.0 or zoom_min_window:
            assert frame_shape is not None, "max_area_frac / zoom_min_window need frame_shape"
            H, W = frame_shape[:2]
            d = [x for x in d if (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]) <= max_area_frac * H * W
                 and (not zoom_min_window or x.get("src") != "zoom" or from_allowed_window(x["bbox"], f["persons"], (H, W), zcfg))]
        keys = ("det_score", "verify")
        f["weapon_detections"], f["weapons"] = weapons_from_detections(
            trk, [x["bbox"] for x in d], [x["score"] for x in d], [x["cls"] for x in d], f["persons"],
            src=[x.get("src", "full") for x in d] if any("src" in x for x in d) else None,
            extra=[{k: x[k] for k in keys if k in x} for x in d] if any("verify" in x for x in d) else None)


def pick_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class SwordPipeline:
    """Per-frame Phases 1-3: weapons (P1) + people from a Phase 2 module + top-down pose on tracked people (P3).

    `phase2` is any object satisfying tracking.interface.Phase2Tracker; by default a temporary stand-in is used
    until the team's Phase 2 module is ready.
    """

    def __init__(self, fps: float, device: str, person_model: Path = DEFAULT_PERSON_MODEL,
                 weapon_model: Path = DEFAULT_WEAPON_MODEL, use_ensemble: bool = False, use_pose: bool = True,
                 zones: list | None = None, homography=None, det_conf: float = 0.1, track_seconds: float = 1.5,
                 phase2=None, pose_device: str = "cpu", weapon_imgsz: int = 640, weapon_zoom: tuple = (),
                 zoom_imgsz: int = 640, zoom_min_score: float = 0.0, zoom_min_window: int = 0, weapon_max_area: float = 1.0,
                 weapon_verifier: str | None = None, verify_mode: str = "geo", verify_min: float = 0.0,
                 weapon_tta_scales: tuple = (), weapon_tta_flip: bool = False, zoom_flip: bool = False):
        from ultralytics import YOLO

        from tracking.interface import StandInPhase2

        self.device = device
        self.phase2 = phase2 or StandInPhase2(fps, device, person_model, det_conf, track_seconds, zones, homography)
        if use_ensemble:
            from ensemble.ensemble_model import SWORDEnsembleDetector

            self.weapon_det = SWORDEnsembleDetector(str(PROJECT_ROOT / "weights" / "yolo26s_best.pt"),
                                                    str(PROJECT_ROOT / "weights" / "yolo26m_best.pt"),
                                                    str(PROJECT_ROOT / "weights" / "rtdetr_l_best.pt"),
                                                    device=device, conf_thr=det_conf)
        else:
            self.weapon_det = YOLO(str(weapon_model))
        self.use_ensemble = use_ensemble
        self.weapon_imgsz = weapon_imgsz  # 640 = training size; CCTV weapons are tiny, larger sizes trade speed for recall
        # test-time augmentation: the full frame is also detected at these sizes and, with weapon_tta_flip, flipped;
        # zoom_flip also detects the flipped zoom crops. Everything is merged by NMS (tracking/zoom.py)
        self.weapon_tta_scales = tuple(weapon_tta_scales or ())
        self.weapon_tta_flip, self.zoom_flip = weapon_tta_flip, zoom_flip
        # person-zoom (tracking/zoom.py): also detect on enlarged crops around tracked people / hands
        self.weapon_zoom, self.zoom_imgsz, self.zoom_min_score = tuple(weapon_zoom or ()), zoom_imgsz, zoom_min_score
        self.zoom_min_window = zoom_min_window
        # boxes larger than this share of the frame are dropped: CCTV weapons are small (largest CCTV training box:
        # 12.3% of the frame), while web training images taught the detector frame-filling weapons
        self.weapon_max_area = weapon_max_area
        # second-stage verifier (tracking/verifier.py): re-checks every detection on a crop; scores are fused with it
        self.verifier, self.verify_mode, self.verify_min = None, verify_mode, verify_min
        if weapon_verifier:
            from tracking.verifier import WeaponVerifier

            self.verifier = WeaponVerifier(weapon_verifier, device=device)
        if (self.weapon_zoom or self.weapon_tta_scales or weapon_tta_flip) and use_ensemble:
            raise ValueError("person-zoom and test-time augmentation are implemented for single detectors, not the WBF ensemble")
        self.track_buffer = max(5, int(round(fps * track_seconds)))
        self.weapon_trk = ByteTrackWrapper(track_buffer=self.track_buffer)
        self.pose = None
        if use_pose:
            from pose_pipeline import RTMPosePipeline

            # wrist velocity: allow gaps up to ~2.5 frames (0.5 s at >= 5 fps, 1.25 s at 2 fps)
            self.pose = RTMPosePipeline(mode="balanced", device=pose_device, track_people=False, load_detector=False,
                                        max_gap_s=max(0.5, 2.5 / fps))
            if not self.pose.is_ready:
                print("[pipeline] RTMPose unavailable — continuing without pose")
                self.pose = None
        self.det_conf = det_conf
        self.sources = {
            "phase2": getattr(self.phase2, "name", type(self.phase2).__name__),
            "weapon_detector": ("WBF ensemble (YOLO26s+YOLO26m+RT-DETR-L)" if use_ensemble else Path(weapon_model).name)
                               + f" @ imgsz {weapon_imgsz}" + "".join(f"+{sz}" for sz in self.weapon_tta_scales)
                               + (" +flip" if weapon_tta_flip else "")
                               + (f" + zoom {'+'.join(self.weapon_zoom)} @ {zoom_imgsz}" if self.weapon_zoom else "")
                               + (" +flip" if self.weapon_zoom and zoom_flip else "")
                               + (f", boxes <= {weapon_max_area:.1%} of frame" if weapon_max_area < 1.0 else "")
                               + (f", verifier {Path(weapon_verifier).name} ({verify_mode})" if weapon_verifier else ""),
            "pose": "RTMPose-m body7 256x192 (rtmlib, top-down on tracked boxes)" if self.pose else None,
        }

    def reset(self) -> None:
        self.phase2.reset()
        self.weapon_trk = ByteTrackWrapper(track_buffer=self.track_buffer)
        if self.pose is not None:
            self.pose.reset_tracking()

    def _weapons(self, frame: np.ndarray):
        if self.use_ensemble:
            r = self.weapon_det.predict(frame, img_size=self.weapon_imgsz)[0]
            return r["boxes"], r["scores"], r["labels"]
        from tracking.zoom import detect_full_frame

        return detect_full_frame(self.weapon_det, frame, self.weapon_imgsz, self.weapon_tta_scales, self.weapon_tta_flip,
                                 conf=self.det_conf, device=self.device)

    def step(self, frame: np.ndarray, frame_idx: int, t: float) -> dict:
        t0 = time.perf_counter()
        people, events = self.phase2.update(frame, t)
        t1 = time.perf_counter()
        pose_by_id = {}
        if self.pose is not None and people:
            for tel in self.pose.process_tracked(frame, people, t):
                pose_by_id[tel["person_id"]] = tel
        t2 = time.perf_counter()
        persons = []
        for p in people:
            tel = pose_by_id.get(p["track_id"])
            if tel is not None:
                kp = [[round(float(x), 1), round(float(y), 1)] for x, y in tel["keypoints_raw"]]
                sc = [round(float(s), 3) for s in tel["scores_raw"]]
                kin = {"hand_to_waist": tel["hand_to_waist"], "arm_angles": tel["arm_angles"],
                       "wrist_velocity": tel["wrist_velocity"], "body_orientation": tel["body_orientation"]}
            else:
                (kp, sc), kin = empty_pose(), None
            persons.append({"track_id": int(p["track_id"]), "bbox": [round(float(v), 1) for v in p["bbox"]],
                            "score": round(float(p["score"]), 3), "keypoints": kp, "scores": sc,
                            "trajectory": p.get("trajectory"), "kinematics": kin})
        t3 = time.perf_counter()
        wb, ws, wc = self._weapons(frame)
        src = None
        if self.weapon_zoom:
            from tracking.zoom import ZoomConfig, detect_in_windows, merge_detections, zoom_windows

            windows = zoom_windows(persons, frame.shape, ZoomConfig(modes=self.weapon_zoom, min_window=self.zoom_min_window))
            zb, zs, zc = detect_in_windows(self.weapon_det, frame, windows, imgsz=self.zoom_imgsz,
                                           conf=max(self.det_conf, self.zoom_min_score), device=self.device,
                                           flip=self.zoom_flip)
            wb, ws, wc, origin = merge_detections([(wb, ws, wc), (zb, zs, zc)], return_source=True)
            src = ["full" if o == 0 else "zoom" for o in origin]
        if self.weapon_max_area < 1.0 and len(wb):
            wb = np.asarray(wb, dtype=float).reshape(-1, 4)
            keep = (wb[:, 2] - wb[:, 0]) * (wb[:, 3] - wb[:, 1]) <= self.weapon_max_area * frame.shape[0] * frame.shape[1]
            wb, ws, wc = wb[keep], np.asarray(ws)[keep], np.asarray(wc)[keep]
            src = [o for o, k in zip(src, keep) if k] if src is not None else None
        extra = None
        if self.verifier is not None and len(wb):
            from tracking.verifier import fuse_scores

            wb, ws, wc = np.asarray(wb, float).reshape(-1, 4), np.asarray(ws, float), np.asarray(wc)
            v = self.verifier.predict(frame, wb)
            keep = v >= self.verify_min
            extra = [{"det_score": round(float(s0), 3), "verify": round(float(p), 4)} for s0, p, k in zip(ws, v, keep) if k]
            wb, wc, src = wb[keep], wc[keep], [o for o, k in zip(src, keep) if k] if src is not None else None
            ws = fuse_scores(ws[keep], v[keep], self.verify_mode)
        t4 = time.perf_counter()
        raw, weapons = weapons_from_detections(self.weapon_trk, wb, ws, wc, persons, src=src, extra=extra)
        self.last_timing = {"phase2_ms": (t1 - t0) * 1e3, "phase3_pose_ms": (t2 - t1) * 1e3,
                            "phase1_weapons_ms": (t4 - t3) * 1e3, "assoc_ms": (time.perf_counter() - t4) * 1e3}
        return {"frame_idx": frame_idx, "timestamp": round(t, 4), "persons": persons, "weapons": weapons,
                "weapon_detections": raw, "events": events}


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
    for p in rec["persons"]:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        sp = p["trajectory"]["speed_heights_s"] if p.get("trajectory") else 0
        cv2.putText(out, f"P{p['track_id']} {sp:.2f}h/s", (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        for (x, y), s in zip(p["keypoints"], p["scores"]):
            if s >= 0.35:
                cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1)
    for w in rec["weapons"]:
        x1, y1, x2, y2 = map(int, w["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        owner = f"->P{w['person_id']}" if w["person_id"] is not None else ""
        tid = "" if w["track_id"] is None else f"W{w['track_id']} "
        cv2.putText(out, f"{tid}{w['cls_name']} {w['score']:.2f}{owner}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return out


def run(args) -> dict:
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zones = json.loads(Path(args.zones).read_text()) if args.zones else []
    H = json.loads(Path(args.homography).read_text()) if args.homography else None
    pipe = SwordPipeline(fps, pick_device(args.device), Path(args.person_model), Path(args.weapon_model),
                         args.ensemble, args.pose, zones, H, args.det_conf)
    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frames, t0, idx = [], time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break
        rec = pipe.step(frame, idx, idx / fps)
        frames.append(rec)
        if writer is not None:
            writer.write(draw(frame, rec))
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - t0
    doc = {"schema": SCHEMA_VERSION, "video": Path(args.video).name, "fps": float(fps), "width": w, "height": h,
           "sources": pipe.sources, "zones": zones, "homography": H, "frames": frames,
           "run": {"frames_processed": idx, "seconds": round(elapsed, 2), "pipeline_fps": round(idx / max(elapsed, 1e-9), 2)}}
    validate(doc)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, default=_json_default))
    n_tracks = len({p["track_id"] for f in frames for p in f["persons"]})
    print(f"[pipeline] {idx} frames in {elapsed:.1f}s ({idx / max(elapsed, 1e-9):.1f} fps); {n_tracks} person tracks; "
          f"{sum(len(f['weapons']) for f in frames)} weapon detections -> {args.out}")
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--person-model", default=str(DEFAULT_PERSON_MODEL))
    ap.add_argument("--weapon-model", default=str(DEFAULT_WEAPON_MODEL))
    ap.add_argument("--ensemble", action="store_true", help="use the 3-model WBF ensemble for weapons (slow)")
    ap.add_argument("--pose", action="store_true", help="run RTMPose-m on tracked people")
    ap.add_argument("--zones", help="JSON list of {name, polygon}")
    ap.add_argument("--homography", help="JSON 3x3 image->ground (metres) matrix")
    ap.add_argument("--det-conf", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save-video")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
