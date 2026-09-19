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

YOLO26s, YOLO26m and RT-DETR-L, trained on the SWORD v2 dataset (10 weapon classes, ~28k images).

