"""
Run the connected pipeline (Phases 1, 2-stand-in, 3, 4, 5) over a collection of videos and store one recording per
video (frames.json, summary.json, ...). Resumable: videos with a summary.json are skipped. The recordings feed the
Phase 4 evaluation (phase4_video_eval.py) and the Temporal Transformer features (fusion.transformer).

    python research/tools/integration/run_video_set.py --set mendeley --stride 2 --out research/runs/<id>_pipeline-mendeley
    python research/tools/integration/run_video_set.py --set ucf --stride 2 --out research/runs/<id>_pipeline-ucf
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.run import run_video  # noqa: E402
from pipeline.system import SwordSystem  # noqa: E402

DATA = PROJECT_ROOT / "research" / "data"


def list_videos(which: str) -> list:
    """(video_id, path) pairs; video_id matches the derived label files."""
    if which == "mendeley":
        root = DATA / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset"
        return [(f"mendeley/{p.parent.parent.name}/{p.parent.name}/video.mp4", p) for p in sorted(root.glob("*/*/video.mp4"))]
    if which == "ucf":
        root = DATA / "ucf_crime" / "extracted" / "UCF_Crimes" / "Videos"
        return [(f"ucf/{p.parent.name}/{p.name}", p) for p in sorted(root.glob("*/*.mp4"))]
    raise ValueError(which)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, choices=["mendeley", "ucf"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2, help="process every k-th frame (25/30 fps -> 12.5/15 fps)")
    ap.add_argument("--weapon-model", default=str(PROJECT_ROOT / "weights/yolo26s_best.pt"))
    ap.add_argument("--weapon-imgsz", type=int, default=640)
    ap.add_argument("--only", nargs="*", help="substring filters on video_id")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = list_videos(args.set)
    if args.only:
        vids = [v for v in vids if any(s in v[0] for s in args.only)]
    if args.limit:
        vids = vids[:args.limit]
    log = open(out / "log.txt", "a")
    systems = {}
    t_all = time.time()
    done = 0
    for vid, path in vids:
        d = out / "videos" / vid.replace("/", "__")
        if (d / "summary.json").exists():
            continue
        cap = cv2.VideoCapture(str(path))
        fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / args.stride
        cap.release()
        key = round(fps, 3)
        if key not in systems:
            systems[key] = SwordSystem(fps=fps, weapon_model=args.weapon_model, weapon_imgsz=args.weapon_imgsz)
        t0 = time.time()
        res = run_video(str(path), d, system=systems[key], frame_stride=args.stride, video_id=vid)
        done += 1
        line = (f"[{datetime.now():%H:%M:%S}] {vid}: {res['summary']['frames']} frames in {time.time() - t0:.0f}s, "
                f"{res['summary']['action_updates']} action estimates")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
    git = lambda *c: subprocess.run(["git", *c], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args), "videos": [v for v, _ in vids],
                "processed_this_call": done, "wall_minutes_this_call": round((time.time() - t_all) / 60, 1),
                "git_head": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                "sources": next(iter(systems.values())).sources if systems else None, "platform": platform.platform()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for s in systems.values():
        s.close()


if __name__ == "__main__":
    main()
