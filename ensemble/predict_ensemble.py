"""
CLI inference tool for the SWORD Ensemble Detector.

Visualizes side-by-side detections from all models and the WBF ensemble.
Supports 2-model (YOLO26s + YOLO26m) or 3-model (+ RT-DETR-L) configurations.

Usage:
    # 3-model ensemble (recommended)
    python ensemble/predict_ensemble.py --source path/to/image.jpg \\
        --weights-rtdetr weights/rtdetr_l_best.pt --save

    # 2-model ensemble (YOLO only)
    python ensemble/predict_ensemble.py --source path/to/image.jpg --save

    # Directory of images
    python ensemble/predict_ensemble.py --source path/to/folder/ --save --conf 0.3
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.ensemble_model import SWORDEnsembleDetector, CLASS_NAMES


# Distinct colors for 10 weapon classes (BGR format for OpenCV)
CLASS_COLORS = [
    (0, 0, 255),       # Pistol - Red
    (0, 165, 255),     # Shotgun - Orange
    (0, 255, 255),     # Assault Rifle - Yellow
    (0, 255, 0),       # Grenade - Green
    (255, 255, 0),     # Knife - Cyan
    (255, 0, 0),       # Sword - Blue
    (255, 0, 255),     # Sniper - Magenta
    (128, 0, 128),     # SMG - Purple
    (0, 128, 255),     # Rocket Launcher - Dark Orange
    (128, 128, 0),     # Machine Gun - Teal
]


def draw_detections(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    title: str = "",
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw bounding boxes with labels on an image."""
    vis = img.copy()
    h, w = vis.shape[:2]

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls = int(labels[i])
        score = float(scores[i])
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label_text = f"{CLASS_NAMES[cls]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label_text, (x1 + 2, max(y1 - 4, th + 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    if title:
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

    return vis


def create_comparison_grid(img: np.ndarray, results: dict, model_names: list) -> np.ndarray:
    """Create a multi-panel comparison grid for all models + ensemble."""
    panels = []
    h, w = img.shape[:2]

    for name in model_names:
        det = results[name]
        title = f"{name} ({len(det['boxes'])} dets)"
        panel = draw_detections(img, det["boxes"], det["scores"], det["labels"], title=title)
        panels.append(panel)

    # Add ensemble panel
    det_e = results["ensemble"]
    title_e = f"WBF Ensemble ({len(det_e['boxes'])} dets)"
    panel_e = draw_detections(img, det_e["boxes"], det_e["scores"], det_e["labels"], title=title_e)
    panels.append(panel_e)

    # Stack horizontally
    grid = np.concatenate(panels, axis=1)

    # Add separator lines between panels
    for i in range(1, len(panels)):
        x = i * w
        cv2.line(grid, (x, 0), (x, h), (200, 200, 200), 2)

    return grid


def main():
    parser = argparse.ArgumentParser(description="SWORD Ensemble Prediction Visualizer")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to image, directory, or video")
    parser.add_argument("--weights-s", type=str,
                        default=str(PROJECT_ROOT / "weights" / "yolo26s_best.pt"),
                        help="YOLO26s weights path")
    parser.add_argument("--weights-m", type=str,
                        default=str(PROJECT_ROOT / "weights" / "yolo26m_best.pt"),
                        help="YOLO26m weights path")
    parser.add_argument("--weights-rtdetr", type=str, default=None,
                        help="RT-DETR-L weights path (omit for 2-model ensemble)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thr", type=float, default=0.55, help="WBF IoU threshold")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--save", action="store_true", help="Save output images")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display images")

    args = parser.parse_args()

    # Auto-detect RT-DETR weights if not specified
    rtdetr_default = PROJECT_ROOT / "weights" / "rtdetr_l_best.pt"
    if args.weights_rtdetr is None and rtdetr_default.exists():
        args.weights_rtdetr = str(rtdetr_default)

    # Initialize detector
    detector = SWORDEnsembleDetector(
        weights_s=args.weights_s,
        weights_m=args.weights_m,
        weights_rtdetr=args.weights_rtdetr,
        conf_thr=args.conf,
        iou_thr=args.iou_thr,
    )

    # Determine output directory
    out_dir = args.output_dir or str(PROJECT_ROOT / "results" / "predictions")
    os.makedirs(out_dir, exist_ok=True)

    # Collect image paths
    source = Path(args.source)
    if source.is_dir():
        image_paths = sorted([
            str(source / f) for f in os.listdir(source)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
    elif source.is_file():
        image_paths = [str(source)]
    else:
        print(f"Error: {source} does not exist")
        sys.exit(1)

    print(f"Processing {len(image_paths)} images...")

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping {img_path} (cannot read)")
            continue

        # Get all sets of detections
        results = detector.predict_and_compare(img_path, img_size=args.img_size)

        # Create comparison grid
        grid = create_comparison_grid(img, results, detector.model_names)

        if args.save:
            out_name = f"ensemble_{Path(img_path).stem}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, grid)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(image_paths)}] Saved: {out_path}")

        if args.show:
            cv2.imshow("SWORD Ensemble", grid)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"\nDone. {len(image_paths)} images processed.")
    if args.save:
        print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
