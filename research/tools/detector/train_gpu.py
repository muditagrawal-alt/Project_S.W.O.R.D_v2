"""
Fine-tune a S.W.O.R.D. detector on the CCTV set at high resolution on a GPU (CUDA), or on Apple MPS for a smoke test.

Self-contained: runs inside the bundle made by package_for_gpu.py (cd sword_cctv; python train_gpu.py ...) or against
the local set (--data research/data/cctv_det/dataset.yaml). Needs: pip install "ultralytics==8.4.67".

    python train_gpu.py --model yolo26s --imgsz 1280                    # CUDA, auto batch (~60% of GPU memory)
    python train_gpu.py --model yolo26m --imgsz 1280 --epochs 80
    python train_gpu.py --model rtdetr-l --imgsz 1280 --batch 4
    python train_gpu.py --model yolo26s --imgsz 1280 --device mps --batch 4 --fraction 0.02 --epochs 1   # smoke test

Rough memory guide at 1280 px (T4 16 GB): YOLO26s batch 8-12, YOLO26m batch 4-6, RT-DETR-L batch 2-4 (or --batch -1).
Model selection uses the CCTV-only `val` split; `val_web` (original web images) is reported as a forgetting check.
Results: runs/<name>/ (best.pt, results.csv, eval.json). Copy best.pt back and run the external evaluation locally:
    python research/tools/integration/eval_phase1_external.py --out research/runs/<id>_external \\
        --model NAME=yolo:path/to/best.pt --no-wbf --imgsz 1280
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INIT = {"yolo26s": "yolo26s_best.pt", "yolo26m": "yolo26m_best.pt", "rtdetr-l": "rtdetr_l_best.pt"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "dataset.yaml"))
    ap.add_argument("--model", choices=sorted(INIT), default="yolo26s")
    ap.add_argument("--init", help="starting checkpoint (default: weights/<model>_best.pt next to this script)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=float, default=-1, help="-1 = Ultralytics auto-batch (CUDA only); or a fixed size")
    ap.add_argument("--lr0", type=float, default=0.002)
    ap.add_argument("--device", default="0", help="CUDA index ('0', '0,1'), 'mps' or 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", default="disk", choices=["ram", "disk", "none"])
    ap.add_argument("--fraction", type=float, default=1.0, help="use a fraction of the training set (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--eval-splits", default="val,test,val_web", help="splits evaluated after training (comma-separated)")
    args = ap.parse_args()

    from ultralytics import RTDETR, YOLO

    init = Path(args.init) if args.init else HERE / "weights" / INIT[args.model]
    if not init.exists():
        raise SystemExit(f"starting checkpoint not found: {init}")
    batch = int(args.batch) if args.batch >= 1 or args.batch == -1 else args.batch
    if args.device in ("mps", "cpu") and batch == -1:
        batch = 4   # auto-batch needs CUDA
    name = args.name or f"{args.model}-cctv-{args.imgsz}"
    model = (RTDETR if args.model.startswith("rtdetr") else YOLO)(str(init))
    t0 = time.time()
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=batch, lr0=args.lr0,
                cos_lr=True, warmup_epochs=1, close_mosaic=min(10, args.epochs), device=args.device, workers=args.workers,
                cache=False if args.cache == "none" else args.cache, fraction=args.fraction, seed=args.seed,
                deterministic=True, amp=args.device not in ("mps", "cpu"), project=str(HERE / "runs"), name=name,
                exist_ok=True, plots=True)
    best = HERE / "runs" / name / "weights" / "best.pt"
    out = {"args": vars(args), "train_minutes": round((time.time() - t0) / 60, 1), "best": str(best)}
    for split in [x for x in args.eval_splits.split(",") if x]:
        try:
            m = (RTDETR if args.model.startswith("rtdetr") else YOLO)(str(best)).val(
                data=args.data, split=split, imgsz=args.imgsz, conf=0.001, iou=0.7, device=args.device, plots=False,
                verbose=False, project=str(HERE / "runs"), name=f"{name}_val_{split}", exist_ok=True)
            out[split] = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map), "precision": float(m.box.mp),
                          "recall": float(m.box.mr)}
        except Exception as e:   # one broken split must not lose the others
            out[split] = {"error": f"{type(e).__name__}: {e}"}
    (HERE / "runs" / name / "eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
