"""
Fine-tune a S.W.O.R.D. detector on the CCTV set (build_cctv_detector_set.py), then evaluate it:
  1. in-domain test split (Mendeley P4 CCTV + P5 phone, UCF-Crime test videos) with Ultralytics val; also the CCTV
     validation split (used for best.pt selection) and val_web (original web domain: forgetting check)
  2. the EXTERNAL mock-attack CCTV set with the Phase 1 protocol (eval_phase1_external.py; DEV Cam1+5 / TEST Cam7)
Everything goes to one run folder with a manifest.

    python research/tools/detector/finetune_cctv.py --smoke                  # 1 epoch on 5% of the data
    python research/tools/detector/finetune_cctv.py                           # full run (15 epochs, 640 px)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "research/data/cctv_det/dataset.yaml"))
    ap.add_argument("--weights", default=str(PROJECT_ROOT / "weights/yolo26s_best.pt"))
    ap.add_argument("--name", default=None)
    ap.add_argument("--imgsz", type=int, default=640, help="640: Mendeley CCTV frames are 640x480 (960 measured 6.5 s/it on M4)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 5% of the data, to check the setup")
    ap.add_argument("--eval-imgsz", type=int, nargs="+", default=[640, 960, 1280])
    ap.add_argument("--skip-external", action="store_true")
    ap.add_argument("--eval-only", help="existing run folder: skip training, evaluate its train/weights/best.pt")
    args = ap.parse_args()

    from ultralytics import YOLO

    stem = Path(args.weights).stem.replace("_best", "")
    name = args.name or f"{stem}-cctv-ft{'-SMOKE' if args.smoke else ''}"
    if args.eval_only:
        run = Path(args.eval_only).resolve()
    else:
        run = PROJECT_ROOT / "research/runs" / f"{datetime.now():%Y%m%d-%H%M%S}_detector-{name}"
        run.mkdir(parents=True)
    git = lambda *c: subprocess.run(["git", *c], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    data_manifest = Path(args.data).parent / "manifest.json"
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "args": vars(args), "git_head": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain")), "init_weights_sha256_12": sha12(Path(args.weights)),
                "data_manifest_sha256_12": sha12(data_manifest) if data_manifest.exists() else None,
                "platform": platform.platform(), "python": platform.python_version()}
    if args.eval_only:
        manifest = {**json.loads((run / "manifest.json").read_text()), "eval_only_rerun": datetime.now().isoformat(timespec="seconds")}
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))

    t0 = time.time()
    model = YOLO(args.weights)
    if not args.eval_only:
        model.train(data=args.data, imgsz=args.imgsz, epochs=1 if args.smoke else args.epochs, batch=args.batch,
                fraction=0.05 if args.smoke else 1.0, device=args.device, workers=args.workers, seed=0, deterministic=True,
                lr0=args.lr0, cos_lr=True, warmup_epochs=1, close_mosaic=min(5, args.epochs), patience=args.patience,
                project=str(run), name="train", exist_ok=True, plots=True, verbose=True)
        manifest["train_minutes"] = round((time.time() - t0) / 60, 1)
    best = run / "train" / "weights" / "best.pt"
    manifest["best_weights"] = str(best.relative_to(PROJECT_ROOT))
    manifest["best_weights_sha256_12"] = sha12(best)

    # 1) in-domain test split (and val) — the original detector as reference
    in_domain = {}
    for label, w in (("finetuned", best), ("original", Path(args.weights))):
        for split in ("val", "test", "val_web"):
            try:
                m = YOLO(str(w)).val(data=args.data, split=split, imgsz=args.imgsz, device=args.device, batch=args.batch,
                                     conf=0.001, iou=0.7, plots=False, verbose=False, project=str(run),
                                     name=f"val_{label}_{split}", exist_ok=True)
            except Exception as e:   # one broken split must not lose the others
                in_domain[f"{label}/{split}"] = {"error": f"{type(e).__name__}: {e}"}
                continue
            in_domain[f"{label}/{split}"] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
                                             "precision": float(m.box.mp), "recall": float(m.box.mr),
                                             "per_class_ap50": {m.names[int(c)]: float(a) for c, a in zip(m.box.ap_class_index, m.box.ap50)}}
    manifest["in_domain"] = in_domain
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(in_domain, indent=1))

    # 2) external mock-attack CCTV set, Phase 1 protocol
    if not args.skip_external:
        kind = "rtdetr" if "rtdetr" in stem else "yolo"
        cmd = [sys.executable, str(PROJECT_ROOT / "research/tools/integration/eval_phase1_external.py"), "--out", str(run / "external_mockattack"),
               "--model", f"{name}={kind}:{best}", "--model", f"{stem}-original={kind}:{args.weights}", "--no-wbf",
               "--imgsz", *map(str, args.eval_imgsz), "--n-boot", "100" if args.smoke else "300"]
        subprocess.run(cmd, check=True)
        manifest["external_results"] = str((run / "external_mockattack" / "results_table.md").relative_to(PROJECT_ROOT))
    manifest["total_minutes"] = round((time.time() - t0) / 60, 1)
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("run folder:", run)


if __name__ == "__main__":
    main()
