"""
Package the CCTV fine-tuning set for training on a remote GPU (Colab, Kaggle, a university server).

The local set (research/data/cctv_det) uses symlinks for the replay images; this writes a self-contained zip with real
files, a dataset.yaml with relative paths, the starting checkpoints, train_gpu.py and a manifest with SHA-256 hashes.
The mock-attack test set is NOT included (it stays local for the external evaluation).

    python research/tools/detector/package_for_gpu.py --out research/data/sword_cctv_gpu_bundle.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(PROJECT_ROOT / "research/data/cctv_det"))
    ap.add_argument("--weights", nargs="+", default=[str(PROJECT_ROOT / "weights" / n) for n in
                                                     ("yolo26s_best.pt", "yolo26m_best.pt", "rtdetr_l_best.pt")])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "research/data/sword_cctv_gpu_bundle.zip"))
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    names = [l.split(":", 1)[1].strip() for l in (data / "dataset.yaml").read_text().splitlines()
             if l.startswith("  ") and ":" in l]
    yaml = ("# S.W.O.R.D. CCTV fine-tuning set (self-contained bundle)\npath: .\ntrain: images/train\nval: images/val\n"
            "test: images/test\nval_web: images/val_web\nnc: %d\nnames:\n%s\n" % (len(names), "\n".join(f"  {i}: {n}" for i, n in enumerate(names))))
    counts, n_files = {}, 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:   # JPEGs do not compress further
        z.writestr("sword_cctv/dataset.yaml", yaml)
        for split in ("train", "val", "test", "val_web"):
            imgs = sorted((data / "images" / split).glob("*"))
            counts[split] = len(imgs)
            for img in imgs:
                lab = data / "labels" / split / (img.stem + ".txt")
                z.write(img.resolve(), f"sword_cctv/images/{split}/{img.name}")   # resolve: symlinks -> real files
                z.writestr(f"sword_cctv/labels/{split}/{img.stem}.txt", lab.resolve().read_text() if lab.exists() else "")
                n_files += 1
        hashes = {}
        for w in map(Path, args.weights):
            if w.exists():
                z.write(w, f"sword_cctv/weights/{w.name}")
                hashes[w.name] = sha256(w)
        z.write(PROJECT_ROOT / "research/tools/detector/train_gpu.py", "sword_cctv/train_gpu.py")
        z.write(PROJECT_ROOT / "research/tools/detector/GPU_TRAINING.md", "sword_cctv/GPU_TRAINING.md")
        manifest = {"created": datetime.now().isoformat(timespec="seconds"), "images": counts, "weights_sha256": hashes,
                    "source_manifest": json.loads((data / "manifest.json").read_text()).get("counts"),
                    "note": "mock-attack (USRT) test set not included; evaluate best.pt locally with eval_phase1_external.py"}
        z.writestr("sword_cctv/bundle_manifest.json", json.dumps(manifest, indent=2))
    print(f"{n_files} images -> {out} ({out.stat().st_size / 1e9:.2f} GB); sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
