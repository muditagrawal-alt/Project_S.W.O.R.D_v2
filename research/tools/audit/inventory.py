"""Stage 1 — file inventory, image/label pairing and stray-file detection (read-only)."""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import (IMAGE_EXTS, RAW_FOLDER_TO_CLASS, RAW_ROOT, YOLO_ROOT, ZIP_SNAPSHOTS,
                    filename_prefix, prefix_class)


def _kind(path: Path, in_label_dir: bool) -> str:
    name, ext = path.name, path.suffix.lower()
    if name == ".DS_Store":
        return "os_junk"
    if ext in IMAGE_EXTS:
        return "image"
    if name == "classes.txt":
        return "classes_txt"
    if ext == ".txt" and in_label_dir:
        return "label"
    if ext == ".cache":
        return "ultralytics_cache"
    if ext in {".yaml", ".yml"}:
        return "config"
    return "other"


def _record(path: Path, root_name: str, split: str, folder_class: int, role_dir: str) -> dict:
    st = path.stat()
    in_label_dir = role_dir == "labels"
    return {
        "root": root_name,
        "split": split,
        "role_dir": role_dir,
        "relpath": path.relative_to(YOLO_ROOT if root_name == "yolo" else RAW_ROOT).as_posix(),
        "abspath": str(path),
        "name": path.name,
        "stem": path.stem,
        "ext": path.suffix.lower(),
        "kind": _kind(path, in_label_dir),
        "size_bytes": st.st_size,
        "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "mtime_date": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d"),
        "prefix": filename_prefix(path.name),
        "prefix_class": prefix_class(path.name),
        "folder_class": folder_class,
    }


def build_inventory() -> pd.DataFrame:
    rows = []
    # YOLO dataset: images/{train,val}, labels/{train,val} plus any loose files.
    for dirpath, _, files in os.walk(YOLO_ROOT):
        d = Path(dirpath)
        rel = d.relative_to(YOLO_ROOT).parts
        role_dir = rel[0] if rel else "root"
        split = rel[1] if len(rel) > 1 else "none"
        for f in files:
            rows.append(_record(d / f, "yolo", split, -1, role_dir))
    # Raw class folders: <N name>/all_data/{all_images,all_labels}
    for dirpath, _, files in os.walk(RAW_ROOT):
        d = Path(dirpath)
        rel = d.relative_to(RAW_ROOT).parts
        folder = rel[0] if rel else "root"
        role_dir = {"all_images": "images", "all_labels": "labels"}.get(rel[-1] if rel else "", "other")
        for f in files:
            rows.append(_record(d / f, "raw", "raw", RAW_FOLDER_TO_CLASS.get(folder, -1), role_dir))
            rows[-1]["raw_folder"] = folder
    df = pd.DataFrame(rows)
    df["raw_folder"] = df.get("raw_folder")
    return df


def pairing_report(inv: pd.DataFrame) -> dict:
    """Images without labels, labels without images, and stem collisions — per split / per raw folder."""
    out = {}
    groups = []
    for split in ("train", "val"):
        groups.append((f"yolo/{split}", inv[(inv.root == "yolo") & (inv.split == split)]))
    for folder, sub in inv[inv.root == "raw"].groupby("raw_folder"):
        groups.append((f"raw/{folder}", sub))
    for key, sub in groups:
        imgs = sub[(sub.kind == "image") & (sub.role_dir == "images")]
        lbls = sub[(sub.kind == "label") & (sub.role_dir == "labels")]
        img_stems = defaultdict(list)
        for s, n in zip(imgs.stem, imgs.name):
            img_stems[s].append(n)
        lbl_stems = set(lbls.stem)
        out[key] = {
            "n_images": int(len(imgs)),
            "n_labels": int(len(lbls)),
            "images_without_label": sorted(n for s, ns in img_stems.items() if s not in lbl_stems for n in ns),
            "labels_without_image": sorted(f"{s}.txt" for s in lbl_stems if s not in img_stems),
            "stem_collisions": {s: ns for s, ns in img_stems.items() if len(ns) > 1},
        }
    return out


def suspicious_files(inv: pd.DataFrame) -> pd.DataFrame:
    """Non-image files inside image folders, zero-byte files, OS junk, odd extensions."""
    rows = []
    for r in inv.itertuples():
        reasons = []
        if r.role_dir == "images" and r.kind != "image":
            reasons.append("non-image file inside an image folder")
        if r.size_bytes == 0 and r.kind != "label":  # empty label files are analysed separately (§4)
            reasons.append("zero-byte file")
        if r.kind == "os_junk":
            reasons.append("OS metadata file (.DS_Store)")
        if r.kind == "image" and r.ext == ".jpeg":
            reasons.append("'.jpeg' extension (rest of dataset uses '.jpg')")
        if r.role_dir == "labels" and r.kind == "other":
            reasons.append("non-label file inside a label folder")
        if r.kind == "other" and r.role_dir not in ("images", "labels"):
            reasons.append("unexpected file in dataset tree")
        if reasons:
            head = b""
            try:
                with open(r.abspath, "rb") as fp:
                    head = fp.read(48)
            except OSError:
                pass
            rows.append({"root": r.root, "relpath": r.relpath, "size_bytes": r.size_bytes, "kind": r.kind,
                         "reasons": "; ".join(reasons), "first_bytes": head[:48].hex()})
    return pd.DataFrame(rows)


def scripts_in_image_dirs(inv: pd.DataFrame) -> list[dict]:
    """Code files left inside image folders (e.g. an augmentation script renamed along with the images)."""
    found = []
    stray = inv[(inv.role_dir == "images") & (inv.kind == "other") & inv.ext.isin([".py", ".txt", ".ipynb", ".sh"])]
    for r in stray.itertuples():
        try:
            text = Path(r.abspath).read_text(errors="replace")
        except OSError:
            continue
        if "import" not in text:
            continue
        target = re.search(r"target_total\s*=\s*(\d+)", text)
        found.append({
            "root": r.root, "relpath": r.relpath, "sha256_prefix": hashlib.sha256(text.encode()).hexdigest()[:12],
            "uses_albumentations": "albumentations" in text,
            "transforms": re.findall(r"A\.(\w+)\(", text),
            "target_total": int(target.group(1)) if target else None,
            "output_name_pattern": (re.search(r'new_name\s*=\s*f"([^"]+)"', text) or [None, None])[1],
            "transforms_boxes": "bbox_params" in text,
        })
    return found


def cross_split_name_overlap(inv: pd.DataFrame) -> dict:
    """Same file name present in both train and val (filename-level leakage)."""
    imgs = inv[(inv.root == "yolo") & (inv.kind == "image")]
    tr = set(imgs[imgs.split == "train"].name)
    va = set(imgs[imgs.split == "val"].name)
    both = sorted(tr & va)
    return {"n": len(both), "names": both[:200]}


def zip_snapshot_listing(zip_path: Path) -> pd.DataFrame:
    """Read only the zip central directory (names, sizes, CRC32, dates) — nothing is extracted."""
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        for zi in zf.infolist():
            if zi.is_dir():
                continue
            rows.append({"member": zi.filename, "size": zi.file_size, "crc32": zi.CRC,
                         "date": "%04d-%02d-%02d" % zi.date_time[:3]})
    return pd.DataFrame(rows)


def compare_zip_to_current(inv: pd.DataFrame, crc_by_relpath: dict) -> dict:
    """Did the YOLO dataset change after each zip snapshot was taken? Compares CRC32 per relative path."""
    cur = inv[inv.root == "yolo"].set_index("relpath")
    out: dict[str, dict] = {}
    for label, zp in ZIP_SNAPSHOTS.items():
        if not zp.exists():
            out[label] = {"available": False}
            continue
        z = zip_snapshot_listing(zp)
        z = z[~z.member.str.startswith("__MACOSX/") & ~z.member.str.contains(r"(^|/)\._", regex=True)]
        z["relpath"] = z.member.str.replace(r"^yolo_dataset/", "", regex=True)
        z = z[~z.relpath.str.endswith(".DS_Store")]
        in_zip, in_cur = set(z.relpath), set(cur.index)
        same = changed = 0
        changed_examples: list[str] = []
        for r in z.itertuples():
            if r.relpath not in cur.index:
                continue
            crc = crc_by_relpath.get(r.relpath)
            if crc is None:
                continue
            if int(crc) == int(r.crc32):
                same += 1
            else:
                changed += 1
                if len(changed_examples) < 25:
                    changed_examples.append(r.relpath)
        out[label] = {
            "available": True,
            "zip_members": int(len(z)),
            "zip_date_histogram": z.date.value_counts().sort_index().to_dict(),
            "only_in_zip": sorted(in_zip - in_cur)[:50],
            "n_only_in_zip": len(in_zip - in_cur),
            "n_only_in_current": len(in_cur - in_zip),
            "only_in_current_examples": sorted(in_cur - in_zip)[:50],
            "identical_crc": same,
            "different_crc": changed,
            "different_examples": changed_examples,
        }
    return out
