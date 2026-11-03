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

## CCTV weights and system profiles (research)

These files are produced locally by the research tools and are not in git either:

| File | Produced by | Used for |
|---|---|---|
| `yolo26s_cctv_best.pt` | `research/tools/detector/finetune_cctv.py` | YOLO26s fine-tuned on CCTV frames (Mendeley firearm set, CCTV-Gun/UCF-Crime) |
| `yolo26s_cctv_zoom_best.pt` | the same, with zoom crops (`research/tools/data/add_zoom_crops.py`) | detector for full frame + person-zoom crops |
| `posec3d_weapon_head_v1.json` | `research/tools/phase4/train_weapon_head.py --install` | learned weapon-action layer on PoseC3D's NTU-120 probabilities |
| `provenance.json` | `research/tools/integration/install_profile.py` | source run and SHA-256 of every installed detector |

A system profile (`config/profiles.json`, read by `pipeline/profiles.py`) names one evaluated configuration:
detector, input size, person-zoom, detection filters and fusion settings. Install one from a training run and a
dev-tuning result with `install_profile.py`, then run it by name: `python -m pipeline.run --profile <name> ...`, the
service's `profile` field, or the command centre (`streamlit run command_center/app.py`). Without a profiles file
only the built-in `original` profile (`yolo26s_best.pt`, 640 px) exists.
