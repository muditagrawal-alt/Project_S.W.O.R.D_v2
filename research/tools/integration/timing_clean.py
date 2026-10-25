"""
Clean timing (nothing else on the GPU): per-stage latency of the connected pipeline and batch-1 detector latency.

The accuracy runs shared the GPU with other jobs, so their timings are not for reporting; accuracy is unaffected
(synchronous Phase 4, deterministic tracking). This measures, on the same machine with no concurrent jobs:
    pipeline   vtest.avi (768x576, 10 fps) and one 1080p mock-attack segment; Phase 4 sync and async;
               weapon detector configs given by --weapon (name:weights:imgsz)
    detectors  batch-1 latency (preloaded frames, warm-up excluded) for each model x input size on CCTV frames

    python research/tools/integration/timing_clean.py --out research/runs/<id>_timing-clean \\
        --weapon yolo26s-640:weights/yolo26s_best.pt:640 --weapon yolo26m-1280:weights/yolo26m_best.pt:1280
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.system import SwordSystem  # noqa: E402

SEGMENT = PROJECT_ROOT / "research/data/mock_attack/sequences/Cam5-From09-24-18To09-26-43-Guns_x264_Segment_0_x264.json"
VTEST = PROJECT_ROOT / "research/data/videos/vtest.avi"
MODELS = {"YOLO26s": ("yolo", "weights/yolo26s_best.pt"), "YOLO26m": ("yolo", "weights/yolo26m_best.pt"),
          "RT-DETR-L": ("rtdetr", "weights/rtdetr_l_best.pt")}


def load_frames(n_vtest: int, n_seg: int):
    cap = cv2.VideoCapture(str(VTEST))
    vt = []
    while len(vt) < n_vtest:
        ok, f = cap.read()
        if not ok:
            break
        vt.append(f)
    cap.release()
    doc = json.loads(SEGMENT.read_text())
    seg = [(cv2.imread(f["image"]), f["timestamp"]) for f in doc["frames"][:n_seg]]
    return vt, seg


def pipeline_timing(frames_t, fps, **kw) -> dict:
    s = SwordSystem(fps=fps, **kw)
    s.reset()
    for i, (f, t) in enumerate(frames_t[:5]):   # warm-up (model init, CoreML compile, MPS kernels)
        s.process(f, i, t)
    s.reset()
    t0 = time.perf_counter()
    for i, (f, t) in enumerate(frames_t):
        s.process(f, i, t)
    wall = time.perf_counter() - t0
    out = s.timing_summary()
    out["wall_fps"] = round(len(frames_t) / wall, 2)
    s.close()
    return out


def detector_latency(frames: list, imgsizes: list, device: str, n: int) -> dict:
    from ultralytics import RTDETR, YOLO

    res = {}
    for name, (kind, w) in MODELS.items():
        m = (RTDETR if kind == "rtdetr" else YOLO)(str(PROJECT_ROOT / w))
        for sz in imgsizes:
            for f in frames[:3]:
                m.predict(f, imgsz=sz, conf=0.001, device=device, verbose=False)
            lat = []
            for f in frames[:n]:
                t0 = time.perf_counter()
                m.predict(f, imgsz=sz, conf=0.001, device=device, verbose=False)
                lat.append((time.perf_counter() - t0) * 1e3)
            res[f"{name}@{sz}"] = {"mean_ms": round(float(np.mean(lat)), 1), "p50_ms": round(float(np.median(lat)), 1),
                                   "p95_ms": round(float(np.percentile(lat, 95)), 1), "n": len(lat)}
            print(f"  {name}@{sz}: {res[f'{name}@{sz}']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weapon", action="append", default=[], help="name:weights:imgsz (repeatable)")
    ap.add_argument("--n-vtest", type=int, default=300)
    ap.add_argument("--n-seg", type=int, default=121)
    ap.add_argument("--n-latency", type=int, default=60)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280, 1920])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vt, seg = load_frames(args.n_vtest, args.n_seg)
    weapons = [w.split(":") for w in (args.weapon or ["yolo26s-640:weights/yolo26s_best.pt:640"])]
    report = {"created": datetime.now().isoformat(timespec="seconds"), "pipeline": {}, "platform": platform.platform(),
              "python": platform.python_version(), "args": vars(args)}
    for name, w, sz in weapons:
        for async_action in (False, True):
            for clip, frames_t, fps in (("vtest_768x576_10fps", [(f, i / 10) for i, f in enumerate(vt)], 10.0),
                                        ("cctv_1080p_2fps", seg, 2.0)):
                key = f"{name}|{'async' if async_action else 'sync'}|{clip}"
                report["pipeline"][key] = pipeline_timing(frames_t, fps, weapon_model=str(PROJECT_ROOT / w),
                                                          weapon_imgsz=int(sz), async_action=async_action)
                t = report["pipeline"][key]
                print(key, "total mean", t["total_ms"]["mean"], "p95", t["total_ms"]["p95"], "| wall fps", t["wall_fps"], flush=True)
                (out / "timing.json").write_text(json.dumps(report, indent=2))
    report["detectors_cctv_1080p"] = detector_latency([f for f, _ in seg], args.imgsz, args.device, args.n_latency)
    (out / "timing.json").write_text(json.dumps(report, indent=2))
    stages = ["phase2_ms", "phase3_pose_ms", "phase1_weapons_ms", "assoc_ms", "phase4_action_ms", "phase5_fusion_ms", "total_ms"]
    lines = ["| config | " + " | ".join(stages) + " | wall fps |", "|---" * (len(stages) + 2) + "|"]
    for k, t in report["pipeline"].items():
        lines.append(f"| {k} | " + " | ".join(f"{t[s]['mean']} / {t[s]['p95']}" for s in stages) + f" | {t['wall_fps']} |")
    lines += ["", "mean / p95 ms per frame", "", "| detector | mean ms | p50 | p95 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} |" for k, v in report["detectors_cctv_1080p"].items()]
    (out / "timing_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
