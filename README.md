# Project S.W.O.R.D v2

**S**urveillance-**O**riented **W**eapon **R**ecognition **D**etector — A multi-model ensemble weapon detection system using YOLO and RT-DETR architectures with Weighted Boxes Fusion (WBF).

## Overview

This project implements a robust weapon detection pipeline trained on a custom 10-class dataset (~28,000 images) for surveillance applications. It combines three complementary detection models into an ensemble, and connects them to tracking, pose, action recognition, temporal alarm rules and an incident service.

### Architecture

```
Input Image
    ├── YOLO26s (Small)     → Higher recall, fast inference
    ├── YOLO26m (Medium)    → Higher precision, stronger features
    └── RT-DETR-L (Large)   → transformer attention
                ↓
        Weighted Boxes Fusion (WBF)
                ↓
        Fused Detections
```

### Detection Classes (10)

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | Pistol | 5 | Sword |
| 1 | Shotgun | 6 | Sniper Rifle |
| 2 | Assault Rifle | 7 | SMG |
| 3 | Grenade | 8 | Rocket Launcher |
| 4 | Knife | 9 | Machine Gun |

## Setup

```bash
# Clone the repository
git clone https://github.com/muditagrawal-alt/Project_S.W.O.R.D_v2.git
cd Project_S.W.O.R.D_v2

# Install dependencies
pip install -r requirements.txt

# Download model weights from GitHub Releases
# Place them in the weights/ directory
```

See [weights/README.md](weights/README.md) for weight download instructions.

## Usage

### Ensemble Inference

```python
from ensemble import SWORDEnsembleDetector

# 3-model ensemble (recommended)
detector = SWORDEnsembleDetector(
    weights_s="weights/yolo26s_best.pt",
    weights_m="weights/yolo26m_best.pt",
    weights_rtdetr="weights/rtdetr_l_best.pt",
)

# Run inference
results = detector.predict("path/to/image.jpg")

# Access detections
for det in results:
    print(f"Boxes: {det['boxes'].shape}")
    print(f"Scores: {det['scores']}")
    print(f"Labels: {det['labels']}")
```

### CLI Prediction with Visualization

```bash
# Single image with 3-model ensemble
python ensemble/predict_ensemble.py --source path/to/image.jpg --save

# Directory of images
python ensemble/predict_ensemble.py --source path/to/folder/ --save --conf 0.3

# 2-model ensemble only (no RT-DETR)
python ensemble/predict_ensemble.py --source image.jpg --save \
    --weights-s weights/yolo26s_best.pt \
    --weights-m weights/yolo26m_best.pt
```

### Interactive Web Dashboard (Streamlit)

Launch the full-featured threat detection HUD with image upload, video processing, and live camera recording:

```bash
streamlit run app.py
```

- **Image Analysis**: Upload images or load validation benchmarks for 4-panel multi-model inspection.
- **Video Analysis**: Frame-by-frame WBF ensemble inference with progress tracking and MP4 download.
- **Live Camera & Recording**: Real-time webcam feed with live WBF inference and on-demand video recording that burns in detections, timestamps, and threat telemetry.
- **Benchmark Hub**: Interactive visualization of the 500-image evaluation report.


### Evaluation

```bash
# Corrected protocol: mAP from all predictions (conf >= 0.001), COCO metrics, bootstrap CIs, batch-1 latency.
# Tune ensemble parameters / operating threshold on one split, report on another:
python ensemble/benchmark.py --tune-list splits/val.txt --test-list splits/test.txt

# Reproduce the legacy numbers (for comparison only)
python ensemble/benchmark.py --legacy-protocol
```

`ensemble/evaluate_ensemble.py`, `run_sota_benchmarks.py`, `optimize_*.py` and `archive_and_compare_benchmarks.py`
are kept only to reproduce the legacy tables and warn when run.

### Connected system (Phases 1, 2-slot, 3, 4, 5 → 6)

```
frame ─► Phase 2 slot (persons + track IDs + trajectories)      tracking/interface.py   (stand-in until the team's module)
      ─► Phase 3 RTMPose-m on tracked boxes (CoreML on Apple)    pose_pipeline.py
      ─► Phase 1 weapon detector ─► weapon tracks ─► weapon↔wrist/person association   tracking/pipeline.py
      ─► Phase 4 PoseC3D NTU-120, per-person 3 s windows every 1 s (sync or background thread)   action_recognition/
      ─► Phase 5 temporal threat fusion: NORMAL / ELEVATED / SUSPICIOUS_ACTIVITY / WEAPON_RELATED_EVENT   fusion/
      ─► Phase 6 incidents, evidence clips (H.264) + SHA-256, events, audit log, WebSocket, dashboard   service/
```

```bash
# research run on a video, an image-sequence JSON (with timestamps) or a camera: records + baseline ladder
python -m pipeline.run --video clip.mp4 --out-dir research/runs/<id> [--gt gt.json] [--save-video] [--weapon-imgsz 1280]

# Phase 6 service: dashboard at http://127.0.0.1:8000 (SQLite by default; SWORD_DATABASE_URL for PostgreSQL)
python -m service
```

- **Phase 2 is a slot.** Plug a tracker in with `SwordSystem(phase2=...)` and check it with `python -m tracking.contract`.
  See [tracking/PHASE2_INTERFACE.md](tracking/PHASE2_INTERFACE.md).
- **Phase 5** is a transparent rule engine (`fusion/rules.py`: smoothed evidence, persistence, hysteresis, ablation
  switches). It is also the baseline that the Temporal Transformer (`fusion/transformer.py`: model and training code)
  has to beat once interval labels exist. `fusion/evaluate.py` turns one recorded run into the baseline ladder
  B0 (frame-level detection) … B4 (full temporal rules) for the research hypothesis.
- Levels describe observable events, never a person's identity or intent.
- The service has no authentication and binds to 127.0.0.1. Add authentication before exposing it on a network.
- **Interval labels** for training Phase 5: format in `fusion/labels.py`, rules in
  [fusion/LABELING_GUIDELINES.md](fusion/LABELING_GUIDELINES.md). Review tool: `python -m service`, then
  http://127.0.0.1:8000/labeler (pre-filled derived labels, revisions with annotator and time, CSV export).
- **CCTV fine-tuning** of the detectors: `research/tools/data/` (public CCTV datasets, leakage-free splits; nothing in
  `dataset/` is modified) and `research/tools/detector/finetune_cctv.py`.

Per-frame records follow `tracking/schema.py` (`sword.frames.v1`), a superset of the Phase 4 tracking/pose schema.
Phases 1–3 alone: `python -m tracking.pipeline --video clip.mp4 --out clip_frames.json --pose`.

### Tests

```bash
python -m pytest tests/ -q
```

### Training (Reproducibility)

```bash
# Train YOLO26s
python models/yolo26s/train_yolo26s.py --data dataset/dataset.yaml

# Train YOLO26m
python models/yolo26m/train_yolo26m.py --data dataset/dataset.yaml

# Train RT-DETR-L
python models/rt_detr/train_rtdetr.py --data dataset/dataset.yaml
```

## Project Structure

```
├── ensemble/                    # Ensemble detection pipeline
│   ├── ensemble_model.py       # 3-model WBF ensemble detector
│   ├── wbf.py                  # Weighted Boxes Fusion implementation
│   ├── predict_ensemble.py     # CLI inference & visualization
│   ├── evaluate_ensemble.py    # Evaluation pipeline
│   └── optimize_phase1.py      # Class-threshold optimization
├── models/                      # Training scripts
│   ├── yolo26s/                # YOLO26s training
│   ├── yolo26m/                # YOLO26m training & resume
│   └── rt_detr/                # RT-DETR-L training & resume
├── tracking/                    # Phase 2 slot (interface, stand-in, contract check), weapon tracks, association, schema
├── pose_pipeline.py             # Phase 3 RTMPose-m (rtmlib)
├── action_recognition/          # Phase 4 PoseC3D (pure PyTorch, verified against MMAction2), online recognizer
├── fusion/                      # Phase 5 features, rule engine, baseline ladder/metrics, Temporal Transformer
├── pipeline/                    # SwordSystem (Phases 1-5 per frame) and the run/evaluate entry point
├── service/                     # Phase 6 FastAPI + WebSocket, SQL database, evidence clips, audit log, dashboard
├── research/tools/              # dataset audit, Phase 4 equivalence, integration tests (results are untracked)
├── weights/                     # Model weights (download separately; PoseC3D checkpoint from OpenMMLab)
└── dataset/dataset.yaml         # Dataset configuration
```

## Technical Details

### Weighted Boxes Fusion (WBF)

Unlike NMS which discards overlapping boxes, WBF merges them by computing the
confidence-weighted average of coordinates, producing tighter bounding boxes.

**Configuration:**
- Model weights: `[1.0, 1.1, 1.1]` (YOLO26s, YOLO26m, RT-DETR-L)
- IoU threshold: 0.55
- Confidence type: Average (penalizes single-model detections)

### Training Configuration

All models were trained on Kaggle Tesla T4 GPUs with:
- **Dataset**: 22,709 train / 5,679 validation images
- **Image size**: 640×640
- **Optimizer**: Ultralytics `optimizer=auto` for YOLO26s/m (AdamW only for RT-DETR-L), cosine LR schedule
- **Early stopping**: the published runs used patience 5 (YOLO) / 10 (RT-DETR) and stopped while mAP was still rising;
  the training scripts now default to patience 50 / 30, a fixed batch of 16 and an explicit seed
- **Augmentation**: Mosaic (disabled in final epochs)

## Citation

If you use this work, please cite:

```bibtex
@misc{sword_v2,
    title={Project S.W.O.R.D v2: Multi-Model Ensemble Weapon Detection},
    author={Mudit Agrawal},
    year={2025},
    url={https://github.com/muditagrawal-alt/Project_S.W.O.R.D_v2}
}
```

## License

This project is for research and academic purposes.
