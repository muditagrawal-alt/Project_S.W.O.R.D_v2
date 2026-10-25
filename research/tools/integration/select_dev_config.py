"""Pre-declared selection rule: the Phase 1 configuration with the best class-agnostic AP50 on the DEV cameras
(Cam1 + Cam5); ties -> smaller input size. Prints run_full_pipeline.py arguments. Cam7 (TEST) is never looked at."""

import json
import sys
from pathlib import Path

WEIGHTS = {"YOLO26s": "weights/yolo26s_best.pt", "YOLO26m": "weights/yolo26m_best.pt", "RT-DETR-L": "weights/rtdetr_l_best.pt"}

res = json.loads((Path(sys.argv[1]) / "results.json").read_text())
scored = sorted(((v["splits_agnostic"]["dev_cam1_cam5"]["mAP50"], -int(k.split("@")[1]), k) for k, v in res.items()), reverse=True)
dev_ap, _, best = scored[0]
model, imgsz = best.split("@")
args = ["--name", f"dev-selected-{model.lower()}-{imgsz}", "--weapon-imgsz", imgsz]
args += ["--ensemble"] if model == "WBF" else ["--weapon-model", WEIGHTS[model]]
print(" ".join(args))
print(f"selected {best} (dev AP50 {dev_ap:.4f}) from {len(scored)} configurations", file=sys.stderr)
