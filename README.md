# Project S.W.O.R.D v2

**S**urveillance-**O**riented **W**eapon **R**ecognition **D**etector — A multi-model ensemble weapon detection system using YOLO and RT-DETR architectures with Weighted Boxes Fusion (WBF).

## Overview

This project implements a robust weapon detection pipeline trained on a custom 10-class dataset (~28,000 images) for surveillance applications. It combines three complementary detection models into an ensemble that achieves superior precision-recall tradeoffs compared to any single model.

### Architecture

```
Input Image
    ├── YOLO26s (Small)     → Higher recall, fast inference
    ├── YOLO26m (Medium)    → Higher precision, stronger features
    └── RT-DETR-L (Large)   → Best standalone mAP, transformer attention
                ↓
        Weighted Boxes Fusion (WBF)
                ↓
        Fused Detections (improved precision + recall)
```

### Detection Classes (10)

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | Pistol | 5 | Sword |
| 1 | Shotgun | 6 | Sniper Rifle |
| 2 | Assault Rifle | 7 | SMG |
| 3 | Grenade | 8 | Rocket Launcher |
| 4 | Knife | 9 | Machine Gun |

## Results

### Individual Model Performance

| Model | Params | Epochs | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| YOLO26s | ~9.4M | 75 | 0.801 | 0.611 | 0.865 | 0.806 | 0.831 |
| YOLO26m | ~21.7M | 40 | 0.799 | 0.597 | 0.923 | 0.768 | 0.832 |
| **RT-DETR-L** | **~32M** | **100** | **0.847** | **0.644** | **0.888** | **0.803** | **0.840** |

### Ensemble Performance

| Configuration | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 2-Model (YOLO26s + YOLO26m) | 0.807 | 0.595 | 0.914 | 0.811 | **0.856** |
| 3-Model Consensus WBF | **0.822** | 0.589 | **0.905** | 0.803 | 0.847 |

**Key findings:**
- RT-DETR-L achieves the best standalone performance across all metrics
- The 2-model YOLO ensemble yields the highest F1-score (0.856)
- WBF fusion suppresses single-model false positives while preserving true detections

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

### Evaluation

```bash
# Full evaluation (all models + ensembles)
python ensemble/evaluate_ensemble.py --mode all

# Quick test (200 images)
python ensemble/evaluate_ensemble.py --max-images 200
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
├── results/evaluation/          # Evaluation reports (JSON)
├── weights/                     # Model weights (download separately)
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
- **Optimizer**: AdamW with cosine annealing LR schedule
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
