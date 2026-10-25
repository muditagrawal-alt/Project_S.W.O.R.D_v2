"""
Build the external CCTV test set from the "US Real-time gun detection in CCTV: an open problem" mock-attack frames.

Source (CC BY-NC 4.0, academic use; cite Salazar González et al., Neural Networks 2020, doi:10.1016/j.neunet.2020.09.013):
https://github.com/Deepknowledge-US/US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset

Input: the extracted 2-fps frames + Pascal-VOC XML (Images/<video>_frame_<n>.jpg/.xml). Nothing is modified.
Output (--out): one sequence file per video segment, which is both the pipeline input (images + timestamps, read by
pipeline.run as an image-sequence source) and the ground truth (weapon boxes per frame):

    {"fps": 2.0, "segment": ..., "camera": ..., "width", "height",
     "frames": [{"frame_idx", "frame_number", "timestamp", "image", "boxes": [{"bbox": [x1,y1,x2,y2], "label", "sword_cls"}]}]}

Timestamps come from the frame numbers ((n - n_first) / fps), so the gaps inside two Cam7 segments keep their true
duration instead of being squeezed out. Labels map to S.W.O.R.D. classes as Handgun -> Pistol, Knife -> Knife,
Short_rifle -> Assault Rifle (a long gun; evaluations should also report class-agnostic and grouped results).
Also writes index.json (per-segment statistics), images.txt and a manifest with input hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.classes import CLASS_NAMES  # noqa: E402

FPS = 2.0
LABEL_TO_SWORD = {"Handgun": CLASS_NAMES.index("Pistol"), "Knife": CLASS_NAMES.index("Knife"),
                  "Short_rifle": CLASS_NAMES.index("Assault Rifle")}
NAME_RE = re.compile(r"(?P<seg>.+)_frame_(?P<n>\d+)\.xml$")


def parse_voc(path: Path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = int(size.findtext("width")), int(size.findtext("height"))
    boxes = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        label = o.findtext("name")
        boxes.append({"bbox": [x1, y1, x2, y2], "label": label, "sword_cls": LABEL_TO_SWORD.get(label)})
    return w, h, boxes


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=str(PROJECT_ROOT / "research/data/mock_attack/Images"))
    ap.add_argument("--zip", default=str(PROJECT_ROOT / "research/data/mock_attack/weapons_images_2fps.zip"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "research/data/mock_attack/sequences"))
    args = ap.parse_args()
    images, out = Path(args.images).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    segs = defaultdict(dict)
    for x in images.glob("*.xml"):
        m = NAME_RE.match(x.name)
        if m:
            segs[m.group("seg")][int(m.group("n"))] = x
    index, all_images, labels = [], [], Counter()
    for seg, by_n in sorted(segs.items()):
        ns = sorted(by_n)
        frames, size = [], None
        for i, n in enumerate(ns):
            jpg = by_n[n].with_suffix(".jpg")
            if not jpg.exists():
                raise FileNotFoundError(jpg)
            w, h, boxes = parse_voc(by_n[n])
            size = size or (w, h)
            labels.update(b["label"] for b in boxes)
            frames.append({"frame_idx": i, "frame_number": n, "timestamp": round((n - ns[0]) / FPS, 3),
                           "image": str(jpg), "boxes": boxes})
            all_images.append(str(jpg))
        gaps = [b - a - 1 for a, b in zip(ns, ns[1:]) if b - a != 1]
        weapon = [bool(f["boxes"]) for f in frames]
        runs = sum(1 for k, v in enumerate(weapon) if v and (k == 0 or not weapon[k - 1]))
        doc = {"dataset": "US-Real-time-gun-detection-in-CCTV (mock attack, 2 fps)", "license": "CC BY-NC 4.0",
               "segment": seg, "camera": seg.split("-")[0], "fps": FPS, "width": size[0], "height": size[1],
               "frames": frames}
        (out / f"{seg}.json").write_text(json.dumps(doc))
        index.append({"segment": seg, "camera": doc["camera"], "frames": len(frames),
                      "duration_s": frames[-1]["timestamp"], "weapon_frames": int(sum(weapon)),
                      "weapon_frame_share": round(sum(weapon) / len(frames), 3), "weapon_runs": runs,
                      "missing_frame_gaps": len(gaps), "missing_frames": int(sum(gaps)),
                      "objects": dict(Counter(b["label"] for f in frames for b in f["boxes"]))})
    (out / "index.json").write_text(json.dumps(index, indent=2))
    (out / "images.txt").write_text("\n".join(all_images) + "\n")
    zip_path = Path(args.zip)
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "script": Path(__file__).name,
                "images_dir": str(images), "source_zip": str(zip_path),
                "source_zip_sha256": sha256(zip_path) if zip_path.exists() else None,
                "segments": len(index), "frames": len(all_images), "objects": dict(labels),
                "label_to_sword": {k: CLASS_NAMES[v] for k, v in LABEL_TO_SWORD.items()}, "fps": FPS,
                "citation": "Salazar González J.L. et al., Real-time gun detection in CCTV: An open problem. "
                            "Neural Networks 132 (2020) 297-308, doi:10.1016/j.neunet.2020.09.013"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("segments", "frames", "objects")}, indent=1))
    for r in index:
        print(f"  {r['segment'][:52]:52s} {r['frames']:5d} fr  {r['duration_s']:7.1f} s  weapon {r['weapon_frame_share']:.2f}"
              f"  runs {r['weapon_runs']:3d}  gaps {r['missing_frame_gaps']}")


if __name__ == "__main__":
    main()
