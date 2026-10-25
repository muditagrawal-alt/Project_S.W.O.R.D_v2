"""
End-to-end smoke test of the connected pipeline on a public, weapon-free pedestrian clip (OpenCV's vtest.avi).

    venv/bin/python research/tools/integration/run_vtest_smoke.py --worktree ../SWORD_v2-phase4

Stages and outputs (research/runs/<timestamp>_pipeline-smoke-vtest/):
    1. Phases 1-3  tracking.pipeline (YOLO26n people + ByteTrack, YOLO26s weapons, RTMPose-m) -> vtest_frames.json,
                   vtest_annotated.mp4
    2. Phase 4     clip_builder (uniform 3 s windows, nearest-partner M=2) + fixed PoseC3D -> vtest_actions.csv/json
                   + the ORIGINAL branch model/preprocessing on the same clips, for comparison
    3. Phase 1     false alarms on weapon-free footage: every weapon detection is a false positive, so
                   detections per minute at several thresholds = false-alarm rate (per model and WBF ensemble)
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
VIDEO = ROOT / "research" / "data" / "videos" / "vtest.avi"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def false_alarms(out_dir: Path, device: str) -> dict:
    from ultralytics import RTDETR, YOLO
    from ensemble.wbf import weighted_boxes_fusion

    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    minutes = len(frames) / fps / 60
    models = {"YOLO26s": YOLO(str(ROOT / "weights" / "yolo26s_best.pt")), "YOLO26m": YOLO(str(ROOT / "weights" / "yolo26m_best.pt")),
              "RT-DETR-L": RTDETR(str(ROOT / "weights" / "rtdetr_l_best.pt"))}
    raw = {k: [] for k in models}
    for k, m in models.items():
        for s in range(0, len(frames), 16):
            for r in m.predict(frames[s:s + 16], imgsz=640, conf=0.05, device=device, verbose=False):
                raw[k].append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))
    h, w = frames[0].shape[:2]
    scale = np.array([w, h, w, h])
    raw["WBF(3-model)"] = []
    for i in range(len(frames)):
        b, s_, c = weighted_boxes_fusion([raw[k][i][0] / scale for k in models], [raw[k][i][1] for k in models],
                                         [raw[k][i][2] for k in models], weights=[1.0, 1.1, 1.1], iou_thr=0.55)
        raw["WBF(3-model)"].append((b * scale, s_, c))
    out = {"video": VIDEO.name, "frames": len(frames), "fps": fps, "minutes": round(minutes, 3), "per_model": {}}
    for k, dets in raw.items():
        row = {}
        for thr in (0.25, 0.5, 0.7):
            n_det = sum(int((s_ >= thr).sum()) for _, s_, _ in dets)
            n_frames = sum(int((s_ >= thr).any()) for _, s_, _ in dets)
            classes = Counter(int(c) for _, s_, cl in dets for c in cl[s_ >= thr])
            row[f"conf>={thr}"] = {"false_detections": n_det, "per_minute": round(n_det / minutes, 2),
                                   "frames_with_alarm_%": round(100 * n_frames / len(frames), 2), "classes": dict(classes)}
        out["per_model"][k] = row
    (out_dir / "false_alarms_vtest.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    run = ROOT / "research" / "runs" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_pipeline-smoke-vtest"
    run.mkdir(parents=True)
    log = []

    t0 = time.time()
    frames_json = run / "vtest_frames.json"
    subprocess.run([sys.executable, "-m", "tracking.pipeline", "--video", str(VIDEO), "--out", str(frames_json), "--pose",
                    "--save-video", str(run / "vtest_annotated.mp4"), "--device", args.device], cwd=ROOT, check=True)
    doc = json.loads(frames_json.read_text())
    tracks = Counter(p["track_id"] for f in doc["frames"] for p in f["persons"])
    with_pose = sum(1 for f in doc["frames"] for p in f["persons"] if max(p["scores"]) > 0)
    stage1 = {"seconds": round(time.time() - t0, 1), "pipeline_fps": doc["run"]["pipeline_fps"], "person_tracks": len(tracks),
              "track_length_frames": {"median": float(np.median(list(tracks.values()))), "max": int(max(tracks.values()))},
              "person_detections": int(sum(tracks.values())), "with_pose": with_pose,
              "weapon_detections_at_tracking_conf_0.1": sum(len(f["weapons"]) for f in doc["frames"])}
    log.append(("phases_1_3", stage1))

    cb = load_module(args.worktree / "clip_builder.py", "clip_builder_fixed")
    ia = load_module(args.worktree / "infer_actions.py", "infer_actions_fixed")
    clips, stats = cb.build_clips_from_intermediate_json(str(frames_json), clip_len=48, window_s=3.0, stride_s=1.0, pair_nearest=True)
    ckpt = args.worktree / "checkpoints" / "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth"
    res_fixed = ia.run_action_inference(clips, str(ckpt), ia.default_action_map_path(), device="cpu", flip_test=True)
    ia.save_results(res_fixed, str(run / "vtest"))
    orig_path = run / "infer_actions_original.py"
    orig_path.write_text(subprocess.run(["git", "-C", str(args.worktree), "show", "HEAD:infer_actions.py"],
                                        capture_output=True, text=True, check=True).stdout)
    orig = load_module(orig_path, "infer_actions_original")
    import torch

    m_orig = orig.load_slowonly_weights(orig.ResNet3dSlowOnly(), str(ckpt))
    amap = ia.load_action_map(ia.default_action_map_path())
    names = amap["all_ntu120_classes"]
    top_fixed, top_orig, agree = Counter(), Counter(), 0
    for c, rf in zip(clips, res_fixed):
        with torch.no_grad():
            p = torch.softmax(m_orig(orig.generate_pose_heatmap_volume(c["keypoint"][:1], c["keypoint_score"][:1], (56, 56), 0.6)), 1)[0]
        to = int(p.argmax())
        top_orig[names.get(to, to)] += 1
        top_fixed[rf["top_action_name"]] += 1
        agree += int(to == rf["top_action_id"])
    stage4 = {"clip_stats": stats, "n_clips": len(clips),
              "fixed_top1_distribution": dict(top_fixed.most_common(10)),
              "original_top1_distribution": dict(top_orig.most_common(10)),
              "top1_agreement_fixed_vs_original": round(agree / max(len(clips), 1), 3),
              "fixed_threat_groups": dict(Counter(r["threat_group"] for r in res_fixed)),
              "heuristic_aiming_fired": int(sum(r["heuristic_aiming_posture"] for r in res_fixed))}
    log.append(("phase_4", stage4))
    log.append(("false_alarms", false_alarms(run, args.device)))
    summary = dict(log)
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str)[:6000])
    print("run folder:", run)


if __name__ == "__main__":
    main()
