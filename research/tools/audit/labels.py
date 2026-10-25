"""Stage 3 — YOLO label parsing and bounding-box validity checks (read-only)."""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from common import NUM_CLASSES

# Thresholds for box flags (documented in the report).
TOL = 1e-6                 # tolerance on normalised coordinates
TOUCH_PX = 1.0             # box edge within 1 px of the image border = "touches border"
DEGENERATE_PX = 2.0        # width or height below 2 px = degenerate
TINY_AREA_NORM = 0.0005    # < 0.05 % of the image area
HUGE_AREA_NORM = 0.95      # > 95 % of the image area
EXTREME_ASPECT = 15.0      # long side / short side
DUP_IOU = 0.95             # same-class boxes overlapping this much are duplicates


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_label_file(path: str, img_w: float | None, img_h: float | None) -> tuple[dict, list[dict]]:
    raw = Path(path).read_bytes()
    file_rec = {
        "label_abspath": path,
        "label_bytes": len(raw),
        "label_crc32": zlib.crc32(raw) & 0xFFFFFFFF,
        "has_bom": raw.startswith(b"\xef\xbb\xbf"),
        "has_crlf": b"\r\n" in raw,
        "n_lines": 0, "n_blank_lines": 0, "n_boxes": 0, "n_bad_lines": 0,
        "n_polygon_lines": 0, "n_float_class_ids": 0, "n_exact_dup_lines": 0,
    }
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    boxes: list[dict] = []
    seen_lines = set()
    for ln_no, line in enumerate(text.splitlines(), 1):
        file_rec["n_lines"] += 1
        s = line.strip()
        if not s:
            file_rec["n_blank_lines"] += 1
            continue
        if s in seen_lines:
            file_rec["n_exact_dup_lines"] += 1
        seen_lines.add(s)
        parts = s.split()
        flags = []
        if len(parts) != 5:
            if len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                file_rec["n_polygon_lines"] += 1
                flags.append("polygon_or_extra_fields")
            else:
                file_rec["n_bad_lines"] += 1
                boxes.append({"line": ln_no, "raw": s[:80], "flags": "wrong_field_count", "valid": False})
                continue
        try:
            cls_f = float(parts[0])
            vals = [float(v) for v in parts[1:5]]
        except ValueError:
            file_rec["n_bad_lines"] += 1
            boxes.append({"line": ln_no, "raw": s[:80], "flags": "non_numeric", "valid": False})
            continue
        if "." in parts[0]:
            file_rec["n_float_class_ids"] += 1
            flags.append("float_class_id")
        cls = int(cls_f) if math.isfinite(cls_f) and cls_f == int(cls_f) else -1
        cx, cy, w, h = vals
        if cls < 0 or cls >= NUM_CLASSES:
            flags.append("class_id_out_of_range")
        if not all(math.isfinite(v) for v in vals):
            flags.append("non_finite")
        if w <= 0 or h <= 0:
            flags.append("non_positive_size")
        if any(v < -TOL or v > 1 + TOL for v in vals):
            flags.append("coord_outside_0_1")
        x1n, y1n, x2n, y2n = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if x1n < -TOL or y1n < -TOL or x2n > 1 + TOL or y2n > 1 + TOL:
            flags.append("box_crosses_border")
        rec = {"line": ln_no, "cls": cls, "cx": cx, "cy": cy, "w": w, "h": h,
               "area_norm": max(w, 0) * max(h, 0)}
        if img_w and img_h:
            x1, y1, x2, y2 = x1n * img_w, y1n * img_h, x2n * img_w, y2n * img_h
            wp, hp = w * img_w, h * img_h
            rec.update(x1=x1, y1=y1, x2=x2, y2=y2, w_px=wp, h_px=hp, area_px=max(wp, 0) * max(hp, 0))
            if min(x1, y1) <= TOUCH_PX or x2 >= img_w - TOUCH_PX or y2 >= img_h - TOUCH_PX:
                flags.append("touches_border")
            if wp < DEGENERATE_PX or hp < DEGENERATE_PX:
                flags.append("degenerate_lt_2px")
            if wp > 0 and hp > 0 and max(wp / hp, hp / wp) > EXTREME_ASPECT:
                flags.append("extreme_aspect")
        if rec["area_norm"] < TINY_AREA_NORM:
            flags.append("tiny_area")
        if rec["area_norm"] > HUGE_AREA_NORM:
            flags.append("huge_area")
        rec["flags"] = ";".join(flags)
        hard = {"class_id_out_of_range", "non_finite", "non_positive_size", "coord_outside_0_1",
                "degenerate_lt_2px", "wrong_field_count", "non_numeric"}
        rec["valid"] = not (set(flags) & hard)
        boxes.append(rec)
        file_rec["n_boxes"] += 1
    # Near-duplicate boxes (same class, IoU > DUP_IOU) inside one file.
    n_dup = 0
    good = [b for b in boxes if b.get("valid") and "cx" in b]
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            if good[i]["cls"] != good[j]["cls"]:
                continue
            bi = np.array([good[i]["cx"] - good[i]["w"] / 2, good[i]["cy"] - good[i]["h"] / 2,
                           good[i]["cx"] + good[i]["w"] / 2, good[i]["cy"] + good[i]["h"] / 2])
            bj = np.array([good[j]["cx"] - good[j]["w"] / 2, good[j]["cy"] - good[j]["h"] / 2,
                           good[j]["cx"] + good[j]["w"] / 2, good[j]["cy"] + good[j]["h"] / 2])
            if _iou(bi, bj) > DUP_IOU:
                n_dup += 1
                good[j]["flags"] = (good[j]["flags"] + ";duplicate_box").strip(";")
    file_rec["n_duplicate_boxes"] = n_dup
    file_rec["n_invalid_boxes"] = sum(1 for b in boxes if not b.get("valid"))
    file_rec["classes_present"] = ",".join(str(c) for c in sorted({b["cls"] for b in boxes if b.get("cls", -1) >= 0}))
    return file_rec, boxes


def scan_labels(label_rows: pd.DataFrame, dims_by_stem_key: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """label_rows: inventory rows of kind 'label'. dims_by_stem_key maps (root, split/folder, stem) -> (w, h)."""
    files, boxes = [], []
    for r in label_rows.itertuples():
        key = (r.root, r.split if r.root == "yolo" else r.raw_folder, r.stem)
        w, h = dims_by_stem_key.get(key, (None, None))
        frec, brecs = parse_label_file(r.abspath, w, h)
        frec.update(root=r.root, split=r.split, raw_folder=getattr(r, "raw_folder", None), stem=r.stem,
                    relpath=r.relpath, prefix=r.prefix, prefix_class=r.prefix_class,
                    has_image_dims=w is not None)
        files.append(frec)
        for b in brecs:
            b.update(root=r.root, split=r.split, stem=r.stem, relpath=r.relpath,
                     prefix=r.prefix, prefix_class=r.prefix_class)
            boxes.append(b)
    return pd.DataFrame(files), pd.DataFrame(boxes)


def read_classes_txt(paths: list[str]) -> dict:
    out = {}
    for p in paths:
        try:
            out[p] = [ln.strip() for ln in Path(p).read_text(errors="replace").splitlines() if ln.strip()]
        except OSError as e:
            out[p] = [f"<unreadable: {e}>"]
    return out
