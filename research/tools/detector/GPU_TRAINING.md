# Training the CCTV detector on a GPU (1280 px)

The Apple M4 trains at 640 px (960 px measured 6.5 s per iteration). Small CCTV weapons need higher resolution and
longer training, so use a CUDA GPU: free Colab T4, free Kaggle P100/T4 (30 h/week), or a university server.

## 1. Package locally (once)

```bash
python research/tools/detector/package_for_gpu.py --out research/data/sword_cctv_gpu_bundle.zip
```

This makes one zip (about 1–2 GB) with the CCTV training set (real files, not symlinks), the S.W.O.R.D. starting
checkpoints, `train_gpu.py` and a manifest with SHA-256 hashes. The mock-attack test set is not included; it stays
local.

## 2a. Google Colab

1. Upload the zip to Google Drive.
2. Runtime → Change runtime type → GPU.
3. Run:

```python
from google.colab import drive; drive.mount('/content/drive')
!cp /content/drive/MyDrive/sword_cctv_gpu_bundle.zip /content/ && cd /content && unzip -q sword_cctv_gpu_bundle.zip
!pip -q install "ultralytics==8.4.67"
%cd /content/sword_cctv
!python train_gpu.py --model yolo26s --imgsz 1280 --epochs 60
# optional, if time allows:
!python train_gpu.py --model yolo26m --imgsz 1280 --epochs 60
!cp -r runs /content/drive/MyDrive/sword_cctv_runs   # keep the results
```

Free Colab sessions stop after a few hours. Ultralytics can resume:
`yolo train resume model=runs/yolo26s-cctv-1280/weights/last.pt`.

## 2b. Kaggle

1. Create a private Kaggle Dataset from the zip. Kaggle unpacks it under `/kaggle/input/<name>/`.
2. In a notebook with GPU enabled:

```bash
cp -r /kaggle/input/<name>/sword_cctv /kaggle/working/ && cd /kaggle/working/sword_cctv
pip -q install "ultralytics==8.4.67"
python train_gpu.py --model yolo26s --imgsz 1280 --epochs 60 --workers 4
```

Keep the dataset **private**. The CCTV sources are licensed for non-commercial academic use (Mendeley CC BY-NC 3.0,
UCF-Crime research use) and must not be re-published.

## 3. Bring the model back and evaluate locally

Copy `runs/<name>/weights/best.pt` into the project (e.g. `research/runs/<id>_detector-gpu-1280/best.pt`), then run:

```bash
python research/tools/integration/eval_phase1_external.py --out research/runs/<id>_external-gpu1280 \
    --model gpu1280=yolo:research/runs/<id>_detector-gpu-1280/best.pt --no-wbf --imgsz 1280
python research/tools/integration/detector_replay.py --recording research/runs/20261024-231132_full-pipeline-default-yolo26s-640 \
    --weights research/runs/<id>_detector-gpu-1280/best.pt --imgsz 1280 --out research/runs/<id>_replay-gpu1280
python research/tools/integration/run_full_pipeline.py --name replay-gpu1280 --out research/runs/<id>_replay-gpu1280 --reevaluate
```

The selection rule stays the same: compare configurations on the mock-attack DEV cameras (Cam1+Cam5), and report
the held-out Cam7.

## Smoke test (on this Mac, MPS)

```bash
python research/tools/detector/train_gpu.py --data research/data/cctv_det/dataset.yaml --init weights/yolo26s_best.pt \
    --imgsz 1280 --device mps --batch 4 --fraction 0.02 --epochs 1 --workers 4 --cache none
```
