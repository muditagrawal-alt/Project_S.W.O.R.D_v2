"""
Derive interval labels (fusion/labels.py format) from dataset annotations. All rows are `source=derived:<rule>`
(unverified); see fusion/LABELING_GUIDELINES.md for the rules and the review workflow.

    python research/tools/labels/derive_intervals.py --out research/data/labels

Writes <dataset>_derived.csv + <dataset>_derived.manifest.json for: mendeley, ucf, usrt (test only).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from fusion.labels import Interval, LabelSetManifest, merge_adjacent, save_labels  # noqa: E402

DATA = PROJECT_ROOT / "research" / "data"
MENDELEY = DATA / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset"
UCF = DATA / "ucf_crime"
USRT = DATA / "mock_attack" / "sequences"
ELEVATED_CODES = {"N3": "running", "N4": "sneaking up with empty hands"}
UCF_LEVEL = {"Shooting": "WEAPON_RELATED_EVENT", "Robbery": "SUSPICIOUS_ACTIVITY", "Fighting": "SUSPICIOUS_ACTIVITY",
             "Assault": "SUSPICIOUS_ACTIVITY"}


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n


def spans(frames, fps: float, gap_s: float):
    """Sorted frame indices -> [(start_s, end_s)] merging gaps <= gap_s."""
    out = []
    for f in sorted(frames):
        t = f / fps
        if out and t - out[-1][1] <= gap_s + 1e-9:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [(a, b) for a, b in out]


def mendeley(frame_offset: int = 1):
    ivs, videos, exhaustive = [], [], {}
    missing = []
    for cls_dir in sorted(p for p in MENDELEY.iterdir() if p.is_dir()):
        for vdir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            code = vdir.name.split("_")[0]
            vid = f"mendeley/{cls_dir.name}/{vdir.name}/video.mp4"
            fps, n = video_info(vdir / "video.mp4")
            videos.append(vid)
            if cls_dir.name == "No_Gun":
                label = "ELEVATED" if code in ELEVATED_CODES else "NORMAL"
                ivs.append(Interval(vid, 0.0, (n - 1) / fps, label, source="derived:mendeley_action_code",
                                    notes=f"{code} {ELEVATED_CODES.get(code, '')}".strip()))
                exhaustive[vid] = True
            elif not (vdir / "label.json").exists():   # gun video shipped without annotations: cannot be labelled
                missing.append(vid)
                videos.remove(vid)
            else:
                lab = json.loads((vdir / "label.json").read_text())
                frames = [a["image_id"] - frame_offset for a in lab["annotations"]]
                if frames:
                    a, b = min(frames) / fps, max(frames) / fps
                    ivs.append(Interval(vid, a, b, "WEAPON_RELATED_EVENT", source="derived:mendeley_boxes",
                                        notes=f"{code} {cls_dir.name}; weapon boxed in {len(set(frames))} of {n} frames"))
                exhaustive[vid] = False
    return ivs, videos, exhaustive, missing


def ucf():
    rows = [l.split() for l in next((UCF / "annotations").rglob("Temporal_Anomaly_Annotation.txt")).read_text().splitlines() if l.strip()]
    vids_dir = UCF / "extracted" / "UCF_Crimes" / "Videos"
    local = {p.name: p for p in vids_dir.glob("*/*.mp4")}   # normals live in more than one archive folder
    ivs, videos, exhaustive, review = [], [], {}, []
    for r in rows:
        name, cat, s1, e1, s2, e2 = r[0], r[1], *map(int, r[2:6])
        path = local.get(name)
        if path is None:
            continue
        folder = path.parent.name
        if cat != "Normal" and cat not in UCF_LEVEL:
            continue
        fps, n = video_info(path)
        vid = f"ucf/{folder}/{name}"
        videos.append(vid)
        exhaustive[vid] = True
        dur = (n - 1) / fps
        ivs.append(Interval(vid, 0.0, dur, "NORMAL", source="derived:ucf_temporal", notes="outside annotated anomaly"))
        if cat != "Normal":
            for s, e in ((s1, e1), (s2, e2)):
                if s >= 0 and e >= 0:
                    note = f"{cat} (UCF-Crime temporal annotation, frames {s}-{e})"
                    if cat == "Robbery":
                        note += "; NEEDS REVIEW: upgrade to WEAPON_RELATED_EVENT if a weapon is visible"
                        review.append(vid)
                    ivs.append(Interval(vid, s / fps, min(e / fps, dur), UCF_LEVEL[cat], source="derived:ucf_temporal", notes=note))
    return ivs, videos, exhaustive, sorted(set(review))


def usrt():
    ivs, videos = [], []
    for f in sorted(USRT.glob("Cam*.json")):
        doc = json.loads(f.read_text())
        vid = f.name
        videos.append(vid)
        times = [fr["timestamp"] for fr in doc["frames"]]
        ivs.append(Interval(vid, 0.0, times[-1], "NORMAL", source="derived:usrt_boxes"))
        on = [fr["timestamp"] for fr in doc["frames"] if fr["boxes"]]
        cur = None
        for t in on:
            if cur and t - cur[1] <= max(1.0, 2.5 / doc["fps"]) + 1e-9:
                cur[1] = t
            else:
                if cur:
                    ivs.append(Interval(vid, cur[0], cur[1], "WEAPON_RELATED_EVENT", source="derived:usrt_boxes"))
                cur = [t, t]
        if cur:
            ivs.append(Interval(vid, cur[0], cur[1], "WEAPON_RELATED_EVENT", source="derived:usrt_boxes"))
    return ivs, videos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DATA / "labels"))
    ap.add_argument("--what", nargs="+", default=["mendeley", "ucf", "usrt"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    if "mendeley" in args.what and MENDELEY.exists():
        ivs, videos, exh, missing = mendeley()
        save_labels(merge_adjacent(ivs, gap_s=1.0), out / "mendeley_derived.csv")
        LabelSetManifest("mendeley_derived", videos, covers_video=False, sources=["derived:mendeley_boxes", "derived:mendeley_action_code"],
                         description="Gun videos: WEAPON_RELATED_EVENT over the boxed span (rest unknown). No_Gun: whole video, "
                                     "N3/N4 ELEVATED else NORMAL.", extra={"exhaustive_per_video": exh, "missing_annotations": missing,
                                                                          "created": stamp}).save(out / "mendeley_derived.manifest.json")
        print(f"mendeley: {len(ivs)} intervals, {len(videos)} videos, missing annotations: {missing}")
    if "ucf" in args.what and (UCF / "annotations").exists():
        ivs, videos, exh, review = ucf()
        save_labels(ivs, out / "ucf_derived.csv")
        LabelSetManifest("ucf_derived", videos, covers_video=True, sources=["derived:ucf_temporal"],
                         description="UCF-Crime official temporal annotations of test videos mapped to levels.",
                         extra={"exhaustive_per_video": exh, "needs_review": review, "created": stamp}).save(out / "ucf_derived.manifest.json")
        print(f"ucf: {len(ivs)} intervals, {len(videos)} videos, {len(review)} need review")
    if "usrt" in args.what and USRT.exists():
        ivs, videos = usrt()
        save_labels(ivs, out / "usrt_derived_TEST_ONLY.csv")
        LabelSetManifest("usrt_derived_TEST_ONLY", videos, covers_video=True, sources=["derived:usrt_boxes"],
                         description="Mock-attack CCTV: WEAPON_RELATED_EVENT where weapons are boxed. External TEST set, never train.",
                         extra={"created": stamp}).save(out / "usrt_derived_TEST_ONLY.manifest.json")
        print(f"usrt: {len(ivs)} intervals, {len(videos)} videos")


if __name__ == "__main__":
    main()
