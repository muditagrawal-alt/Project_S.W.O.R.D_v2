"""
Phase 2 benchmark: person trackers on the standard MOTChallenge data, every tracker fed IDENTICAL detections.

Sequences (research/tools/data/fetch_mot.py):
  MOT17 validation half: the second half of each of the 7 MOT17 train sequences (the protocol of the ByteTrack,
    OC-SORT and BoT-SORT ablations), frames renumbered from 1, trackers start fresh at the first val frame
  PETS09-S2L1 (MOT15): the scene of the vtest.avi street clip used throughout this project

Detections:
  yolo26n   COCO YOLO26n person class, conf >= 0.1, 640 px (the detector of the S.W.O.R.D. Phase 2 slot)
  public    MOT17's public Faster R-CNN detections (MOT17 only; detector-independent comparison)

Trackers:
  aditya_as_submitted   persistent_video_tracking as submitted: detections cut at 0.35, sv.ByteTrack (activation 0.30,
                        buffer 30, matching 0.75) not updated on empty frames, buffer not adapted to the frame rate
  aditya_fixed          the same tracker after the fixes: detections >= 0.1, updated every frame, frame-rate aware
  standin_bytetrack     the system's previous Phase 2: Ultralytics ByteTrack, buffer 1.5 s
  botsort               Ultralytics BoT-SORT with camera-motion compensation (sparse optical flow), no ReID
  iou_baseline          greedy IoU matching to the previous frame (no motion model): lower bound

Metrics: TrackEval (HOTA, DetA, AssA, MOTA, IDF1, ID switches, MT/ML, official MOTChallenge preprocessing) and, as a
cross-check on PETS09-S2L1, py-motmetrics (MOTA, IDF1). Tracker time is the update time per frame (detection excluded).

    python research/tools/tracking/mot_benchmark.py --data research/data/mot --out research/runs/<id>_mot-benchmark
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tracking.interface import DEFAULT_PERSON_MODEL  # noqa: E402
from tracking.tracker import ByteTrackWrapper, _Detections  # noqa: E402

MOT17_SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
TRACKERS = ["aditya_as_submitted", "aditya_fixed", "standin_bytetrack", "botsort", "iou_baseline"]


# ---------------------------------------------------------------- sequences and detections
def sequences(data: Path) -> dict:
    seqs = {}
    for s in MOT17_SEQS:
        d = data / "MOT17" / "train" / f"{s}-FRCNN"
        ini = configparser.ConfigParser()
        ini.read(d / "seqinfo.ini")
        n = int(ini["Sequence"]["seqLength"])
        seqs[s] = {"bench": "MOT17", "dir": d, "first": n // 2 + 1, "last": n, "fps": float(ini["Sequence"]["frameRate"]),
                   "w": int(ini["Sequence"]["imWidth"]), "h": int(ini["Sequence"]["imHeight"])}
    d = data / "MOT15" / "train" / "PETS09-S2L1"
    ini = configparser.ConfigParser()
    ini.read(d / "seqinfo.ini")
    seqs["PETS09-S2L1"] = {"bench": "MOT15", "dir": d, "first": 1, "last": int(ini["Sequence"]["seqLength"]),
                           "fps": float(ini["Sequence"]["frameRate"]), "w": int(ini["Sequence"]["imWidth"]),
                           "h": int(ini["Sequence"]["imHeight"])}
    return seqs


def frame_path(seq: dict, f: int) -> Path:
    return seq["dir"] / "img1" / f"{f:06d}.jpg"


def yolo_detections(seq: dict, cache: Path, device: str) -> dict:
    """{frame: (xyxy[N,4], score[N])}, cached as .npz."""
    if cache.exists():
        z = np.load(cache)
        return {int(f): (z["b"][z["f"] == f], z["s"][z["f"] == f]) for f in range(seq["first"], seq["last"] + 1)}
    from ultralytics import YOLO

    model = YOLO(str(DEFAULT_PERSON_MODEL))
    F, B, S, out = [], [], [], {}
    for f in range(seq["first"], seq["last"] + 1):
        r = model.predict(str(frame_path(seq, f)), conf=0.1, classes=[0], imgsz=640, device=device, verbose=False)[0]
        b, s = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
        out[f] = (b, s)
        F += [f] * len(s)
        B.append(b)
        S.append(s)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, f=np.array(F), b=np.concatenate(B) if B else np.zeros((0, 4)), s=np.concatenate(S) if S else np.zeros(0))
    return out


def public_detections(seq: dict) -> dict:
    a = np.loadtxt(seq["dir"] / "det" / "det.txt", delimiter=",", ndmin=2)
    out = {}
    for f in range(seq["first"], seq["last"] + 1):
        r = a[a[:, 0] == f]
        out[f] = (np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]], r[:, 6].astype(np.float32))
    return out


# ---------------------------------------------------------------- trackers (xyxy, scores, frame) -> [(id, xyxy, score)]
class Aditya:
    def __init__(self, fps: float, as_submitted: bool):
        import supervision as sv
        from persistent_video_tracking.tracking.dual_tracker import DualStreamTracker

        self.sv, self.min_score = sv, 0.35 if as_submitted else 0.1
        self.trk = DualStreamTracker(frame_rate=30.0 if as_submitted else fps, update_on_empty=not as_submitted)

    def __call__(self, b, s, img):
        keep = s >= self.min_score
        d = (self.sv.Detections(xyxy=b[keep].astype(np.float32), confidence=s[keep].astype(np.float32),
                                class_id=np.zeros(int(keep.sum()), int)) if keep.any() else self.sv.Detections.empty())
        obs, _, _ = self.trk.update(self.sv.Detections.empty(), d, "seq", 0, 0.0)
        return [(o.track_id, (o.x1, o.y1, o.x2, o.y2), o.confidence) for o in obs if o.object_type == "person"]


class StandIn:
    def __init__(self, fps: float):
        self.trk = ByteTrackWrapper(track_buffer=max(5, int(round(fps * 1.5))))

    def __call__(self, b, s, img):
        return [(t["track_id"], t["bbox"], t["score"]) for t in self.trk.update(b, s)]


class BoTSORT:
    def __init__(self, fps: float):
        from ultralytics.trackers.bot_sort import BOTSORT

        self.trk = BOTSORT(SimpleNamespace(tracker_type="botsort", track_high_thresh=0.25, track_low_thresh=0.1,
                                           new_track_thresh=0.25, track_buffer=max(5, int(round(fps * 1.5))),
                                           match_thresh=0.8, fuse_score=True, gmc_method="sparseOptFlow",
                                           proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model="auto"))

    def __call__(self, b, s, img):
        out = self.trk.update(_Detections(b, s.astype(np.float32), np.zeros_like(s, dtype=np.float32)), img)
        return [(int(r[4]), tuple(r[:4]), float(r[5])) for r in np.asarray(out).reshape(-1, 8)]


class IoUBaseline:
    def __init__(self, fps: float, min_score: float = 0.5, iou_thr: float = 0.3):
        self.prev, self.next_id, self.min_score, self.iou_thr = [], 1, min_score, iou_thr

    def __call__(self, b, s, img):
        keep = s >= self.min_score
        b, s = b[keep], s[keep]
        out, used = [], set()
        for i in np.argsort(-s):
            best, best_iou = None, self.iou_thr
            for tid, pb in self.prev:
                if tid in used:
                    continue
                ix = max(0, min(b[i][2], pb[2]) - max(b[i][0], pb[0]))
                iy = max(0, min(b[i][3], pb[3]) - max(b[i][1], pb[1]))
                inter = ix * iy
                u = (b[i][2] - b[i][0]) * (b[i][3] - b[i][1]) + (pb[2] - pb[0]) * (pb[3] - pb[1]) - inter
                if inter / max(u, 1e-9) > best_iou:
                    best, best_iou = tid, inter / u
            if best is None:
                best, self.next_id = self.next_id, self.next_id + 1
            used.add(best)
            out.append((best, tuple(b[i]), float(s[i])))
        self.prev = [(t, bb) for t, bb, _ in out]
        return out


def make_tracker(name: str, fps: float):
    return {"aditya_as_submitted": lambda: Aditya(fps, True), "aditya_fixed": lambda: Aditya(fps, False),
            "standin_bytetrack": lambda: StandIn(fps), "botsort": lambda: BoTSORT(fps),
            "iou_baseline": lambda: IoUBaseline(fps)}[name]()


# ---------------------------------------------------------------- evaluation
def write_gt(seq_name: str, seq: dict, gt_root: Path, split: str):
    d = gt_root / f"{seq['bench']}-{split}" / seq_name
    (d / "gt").mkdir(parents=True, exist_ok=True)
    a = np.loadtxt(seq["dir"] / "gt" / "gt.txt", delimiter=",", ndmin=2)
    a = a[(a[:, 0] >= seq["first"]) & (a[:, 0] <= seq["last"])]
    a[:, 0] -= seq["first"] - 1
    fmt = ["%d", "%d"] + ["%.2f"] * 4 + ["%g"] * (a.shape[1] - 6)
    np.savetxt(d / "gt" / "gt.txt", a, delimiter=",", fmt=fmt)
    n = seq["last"] - seq["first"] + 1
    (d / "seqinfo.ini").write_text(f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate={seq['fps']:g}\nseqLength={n}\n"
                                   f"imWidth={seq['w']}\nimHeight={seq['h']}\nimExt=.jpg\n")


def trackeval_run(gt_root: Path, trk_root: Path, bench: str, split: str, seqs: list, trackers: list, out: Path) -> dict:
    import trackeval

    seqmap = gt_root / "seqmaps" / f"{bench}-{split}.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(seqs) + "\n")
    ev_cfg = trackeval.Evaluator.get_default_eval_config()
    ev_cfg.update(USE_PARALLEL=False, PRINT_CONFIG=False, PRINT_RESULTS=False, PRINT_ONLY_COMBINED=True,
                  TIME_PROGRESS=False, DISPLAY_LESS_PROGRESS=True, OUTPUT_SUMMARY=True, OUTPUT_DETAILED=True,
                  PLOT_CURVES=False)
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update(GT_FOLDER=str(gt_root), TRACKERS_FOLDER=str(trk_root), OUTPUT_FOLDER=str(out), BENCHMARK=bench,
                  SPLIT_TO_EVAL=split, SEQMAP_FILE=str(seqmap), TRACKERS_TO_EVAL=trackers, TRACKER_SUB_FOLDER="data",
                  PRINT_CONFIG=False, DO_PREPROC=bench != "MOT15")
    res, _ = trackeval.Evaluator(ev_cfg).evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)],
                                                  [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
                                                   trackeval.metrics.Identity()])
    out_ = {}
    for t in trackers:
        per = {}
        for s, r in res["MotChallenge2DBox"][t].items():
            m = r["pedestrian"]
            per[s] = {"HOTA": round(100 * float(np.mean(m["HOTA"]["HOTA"])), 2),
                      "DetA": round(100 * float(np.mean(m["HOTA"]["DetA"])), 2),
                      "AssA": round(100 * float(np.mean(m["HOTA"]["AssA"])), 2),
                      "MOTA": round(100 * float(m["CLEAR"]["MOTA"]), 2), "IDF1": round(100 * float(m["Identity"]["IDF1"]), 2),
                      "IDSW": int(m["CLEAR"]["IDSW"]), "Frag": int(m["CLEAR"]["Frag"]), "MT": int(m["CLEAR"]["MT"]),
                      "ML": int(m["CLEAR"]["ML"]), "FP": int(m["CLEAR"]["CLR_FP"]), "FN": int(m["CLEAR"]["CLR_FN"])}
        out_[t] = per
    return out_


def motmetrics_check(gt_file: Path, trk_file: Path) -> dict:
    """Independent MOTA / IDF1 with py-motmetrics (IoU >= 0.5), with our own IoU matrix: motmetrics 1.4's helper calls
    np.asfarray, which NumPy 2 removed."""
    import motmetrics as mm

    def load(f):
        a = np.loadtxt(f, delimiter=",", ndmin=2)
        return a[a[:, 6] > 0] if a.size else a.reshape(0, 10)

    gt, tr = load(gt_file), load(trk_file)
    acc = mm.MOTAccumulator(auto_id=True)
    for f in range(1, int(gt[:, 0].max()) + 1):
        g, t = gt[gt[:, 0] == f], tr[tr[:, 0] == f] if len(tr) else tr
        gb, tb = g[:, 2:6], t[:, 2:6]
        d = np.full((len(g), len(t)), np.nan)
        if len(g) and len(t):
            ix = np.clip(np.minimum(gb[:, None, 0] + gb[:, None, 2], tb[None, :, 0] + tb[None, :, 2])
                         - np.maximum(gb[:, None, 0], tb[None, :, 0]), 0, None)
            iy = np.clip(np.minimum(gb[:, None, 1] + gb[:, None, 3], tb[None, :, 1] + tb[None, :, 3])
                         - np.maximum(gb[:, None, 1], tb[None, :, 1]), 0, None)
            inter = ix * iy
            iou = inter / (gb[:, None, 2] * gb[:, None, 3] + tb[None, :, 2] * tb[None, :, 3] - inter)
            d = np.where(iou >= 0.5, 1 - iou, np.nan)
        acc.update(g[:, 1].astype(int).tolist(), t[:, 1].astype(int).tolist(), d)
    s = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches"], name="x")
    return {"MOTA": round(100 * float(s["mota"].iloc[0]), 2), "IDF1": round(100 * float(s["idf1"].iloc[0]), 2),
            "IDSW": int(s["num_switches"].iloc[0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trackers", nargs="+", default=TRACKERS)
    ap.add_argument("--detectors", nargs="+", default=["yolo26n", "public"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = sequences(Path(args.data))
    gt_root = out / "gt"
    for name, seq in seqs.items():
        write_gt(name, seq, gt_root, "val" if seq["bench"] == "MOT17" else "train")
    timing, results, checks = {}, {}, {}
    for det in args.detectors:
        for name, seq in seqs.items():
            if det == "public" and seq["bench"] != "MOT17":
                continue
            dets = (yolo_detections(seq, out / "detections" / "yolo26n" / f"{name}.npz", args.device)
                    if det == "yolo26n" else public_detections(seq))
            n_det = sum(len(v[1]) for v in dets.values())
            print(f"[{det}] {name}: frames {seq['first']}-{seq['last']}, {n_det} detections", flush=True)
            for tname in args.trackers:
                trk, rows, t_upd = make_tracker(tname, seq["fps"]), [], 0.0
                for f in range(seq["first"], seq["last"] + 1):
                    img = cv2.imread(str(frame_path(seq, f))) if tname == "botsort" else None
                    b, s = dets[f]
                    t0 = time.perf_counter()
                    for tid, (x1, y1, x2, y2), sc in trk(np.asarray(b, float).reshape(-1, 4), np.asarray(s, float).reshape(-1), img):
                        rows.append(f"{f - seq['first'] + 1},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{sc:.4f},-1,-1,-1")
                    t_upd += time.perf_counter() - t0
                split = "val" if seq["bench"] == "MOT17" else "train"
                tdir = out / "trackers" / det / f"{seq['bench']}-{split}" / tname / "data"
                tdir.mkdir(parents=True, exist_ok=True)
                (tdir / f"{name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
                timing.setdefault(det, {}).setdefault(tname, []).append(1000 * t_upd / (seq["last"] - seq["first"] + 1))
        for bench, split in (("MOT17", "val"), ("MOT15", "train")):
            names = [n for n, s in seqs.items() if s["bench"] == bench]
            if det == "public" and bench == "MOT15":
                continue
            results.setdefault(det, {})[f"{bench}-{split}"] = trackeval_run(
                gt_root, out / "trackers" / det, bench, split, names, args.trackers, out / "trackeval" / det)
        if "PETS09-S2L1" in seqs and det == "yolo26n":
            gtf = gt_root / "MOT15-train" / "PETS09-S2L1" / "gt" / "gt.txt"
            checks = {t: motmetrics_check(gtf, out / "trackers" / det / "MOT15-train" / t / "data" / "PETS09-S2L1.txt")
                      for t in args.trackers}
    summary = {"created": datetime.now().isoformat(timespec="seconds"), "trackers": args.trackers,
               "detectors": args.detectors, "sequences": {n: {k: v for k, v in s.items() if k != "dir"} for n, s in seqs.items()},
               "results": results, "tracker_ms_per_frame": {d: {t: round(float(np.mean(v)), 3) for t, v in tt.items()}
                                                            for d, tt in timing.items()},
               "motmetrics_crosscheck_PETS09": checks}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    lines = ["| detector | data | tracker | HOTA | DetA | AssA | MOTA | IDF1 | ID sw. | MT | ML | ms/frame |", "|" + "---|" * 12]
    for det, bb in results.items():
        for b, rr in bb.items():
            for t, per in rr.items():
                c = per["COMBINED_SEQ"]
                lines.append(f"| {det} | {b} | {t} | {c['HOTA']} | {c['DetA']} | {c['AssA']} | {c['MOTA']} | {c['IDF1']} | "
                             f"{c['IDSW']} | {c['MT']} | {c['ML']} | {summary['tracker_ms_per_frame'][det][t]} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
