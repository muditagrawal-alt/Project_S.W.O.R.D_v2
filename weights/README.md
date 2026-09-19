# Model Weights

The trained model weights are not stored in git due to their size (~434 MB total).

## Download

Download `sword_v2_weights.zip` from the [GitHub Releases](https://github.com/muditagrawal-alt/Project_S.W.O.R.D_v2/releases) page.

Extract the weights into this directory:

```bash
cd weights/
unzip sword_v2_weights.zip
```

After extraction, you should have:

```
weights/
├── yolo26s_best.pt      # YOLO26s fine-tuned on SWORD v2 (~57 MB)
├── yolo26m_best.pt      # YOLO26m fine-tuned on SWORD v2 (~125 MB)
└── rtdetr_l_best.pt     # RT-DETR-L fine-tuned on SWORD v2 (~252 MB)
```

## Model Details

| Model | Architecture | Params | Epochs | mAP@50 | mAP@50:95 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| YOLO26s | YOLO v11 Small | ~9.4M | 75 | 0.801 | 0.611 | 0.865 | 0.806 |
| YOLO26m | YOLO v11 Medium | ~21.7M | 40 | 0.799 | 0.597 | 0.923 | 0.768 |
| RT-DETR-L | RT-DETR Large | ~32M | 100 | 0.847 | 0.644 | 0.888 | 0.803 |

All models were trained on the SWORD v2 dataset (10 weapon classes, ~28k images) using Kaggle Tesla T4 GPUs.
