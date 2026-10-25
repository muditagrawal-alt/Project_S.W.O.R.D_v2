"""Stages 7–11 — analysis, figures, contact sheets, remediation proposals and REPORT.md.

Run after run_audit.py (re-runnable: it only reads the run's cache, so thresholds can be recalibrated):
    venv/bin/python research/tools/audit/report.py --run-id <run_id>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CANONICAL_NAMES, CURRENT_YAML_NAMES, DISPLAY_NAMES, NUM_CLASSES, PROJECT_ROOT,  # noqa: E402
                    RAW_ROOT, RESEARCH_ROOT, YOLO_ROOT, append_index, dataset_fingerprint, setup_logging,
                    write_json)
import inventory as inv_mod  # noqa: E402
import similarity as sim  # noqa: E402

CONF_ORDER = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
EVIDENCE_COLS = ["a", "b", "ph_id", "ph_min", "T", "dh_id", "dh_T", "ah_T", "cos", "pixel_eq", "same_dims", "area_ratio"]
COCO_SMALL, COCO_MEDIUM = 32 ** 2, 96 ** 2

try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
except OSError:
    FONT = ImageFont.load_default()


# ───────────────────────────────────── helpers ─────────────────────────────────────
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if pd.isna(v) else floatfmt.format(v)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def pct(n: float, d: float) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def tile(path: str, size: int = 240, caption: str = "", boxes=None, color=(255, 60, 60),
         boxes2=None, color2=(60, 160, 255)) -> Image.Image:
    """Thumbnail with optional boxes (x1, y1, x2, y2, label) in original pixels; `boxes2` drawn in a second colour."""
    cap_h = 30
    canvas = Image.new("RGB", (size, size + cap_h), (25, 25, 25))
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = size / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
        ox, oy = (size - im.width) // 2, (size - im.height) // 2
        canvas.paste(im, (ox, oy))
        d = ImageDraw.Draw(canvas)
        for group, colr in ((boxes or [], color), (boxes2 or [], color2)):
            for x1, y1, x2, y2, lab in group:
                d.rectangle([ox + x1 * s, oy + y1 * s, ox + x2 * s, oy + y2 * s], outline=colr, width=2)
                if lab:
                    d.text((ox + x1 * s + 2, oy + y1 * s + 1), lab, fill=colr, font=FONT)
    except Exception as e:  # unreadable images still get a tile
        ImageDraw.Draw(canvas).text((5, 5), f"unreadable: {e}"[:40], fill=(255, 80, 80), font=FONT)
    d = ImageDraw.Draw(canvas)
    for i, line in enumerate(caption.split("\n")[:2]):
        d.text((4, size + 2 + 13 * i), line[:44], fill=(230, 230, 230), font=FONT)
    return canvas


def sheet(tiles: list[Image.Image], cols: int, out: Path, title: str) -> None:
    if not tiles:
        return
    tw, th = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    img = Image.new("RGB", (cols * tw, rows * th + 28), (10, 10, 10))
    ImageDraw.Draw(img).text((8, 7), title, fill=(255, 255, 255), font=FONT)
    for i, t in enumerate(tiles):
        img.paste(t, ((i % cols) * tw, 28 + (i // cols) * th))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=88)


def savefig(fig, out_base: Path, pdf: bool = True) -> None:
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    if pdf:
        fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def gt_boxes_for(boxes: pd.DataFrame, split: str, stem: str) -> list:
    sub = boxes[(boxes.split == split) & (boxes.stem == stem) & boxes.valid.fillna(False)]
    return [(r.x1, r.y1, r.x2, r.y2, DISPLAY_NAMES[int(r.cls)] if 0 <= r.cls < NUM_CLASSES else "?")
            for r in sub.itertuples() if pd.notna(getattr(r, "x1", np.nan))]


# ───────────────────────────────────── loading ─────────────────────────────────────
class Audit:
    def __init__(self, run_dir: Path, logger):
        self.run_dir, self.log = run_dir, logger
        c = run_dir / "cache"
        self.inv = pd.read_pickle(c / "inventory.pkl")
        self.files = pd.read_pickle(c / "files.pkl")
        self.contents = pd.read_pickle(c / "contents.pkl")
        self.lab_files, self.boxes = pd.read_pickle(c / "labels.pkl")
        ev = sim.classify_edges(pd.read_pickle(c / "edges.pkl")[EVIDENCE_COLS])  # re-classified with the current RULES
        self.verified = (c / "verification.pkl").exists()
        self.edges = sim.finalize_edges(ev, pd.read_pickle(c / "verification.pkl")) if self.verified else ev
        sens = run_dir / "results" / "detector_sensitivity.json"
        self.sensitivity = json.loads(sens.read_text()) if sens.exists() else None
        self.rand_cos = np.load(c / "random_pair_cos.npy") if (c / "random_pair_cos.npy").exists() else np.array([])
        r = run_dir / "results"
        self.pairing = json.loads((r / "pairing.json").read_text())
        self.suspicious = inv_mod.suspicious_files(self.inv)
        self.suspicious.to_csv(r / "suspicious_files.csv", index=False)
        self.scripts = inv_mod.scripts_in_image_dirs(self.inv)
        write_json(r / "scripts_in_image_dirs.json", self.scripts)
        self.zip_prov = json.loads((r / "zip_provenance.json").read_text())
        self.selftest = json.loads((r / "selftest.json").read_text())
        self.classes_txt = json.loads((r / "classes_txt_contents.json").read_text())
        self.manifest = json.loads((run_dir / "manifest.json").read_text())
        self.summary: dict = {}
        self.md: list[str] = []
        self._prepare()

    def _prepare(self):
        f = self.files
        for col in ("pil_verify_error", "pil_decode_error", "error", "exif_orientation", "jpeg_trailing_bytes", "jpeg_has_eoi",
                    "exif_software", "jpeg_quality", "jpeg_qsig"):
            if col not in f:
                f[col] = np.nan
        f["content_id"] = f["content_id"].astype("Int64")
        lab = self.lab_files
        key_lab = lab.assign(k=lab.root + "|" + lab.split.fillna("") + "|" + lab.raw_folder.fillna("") + "|" + lab.stem)
        lab_map = key_lab.set_index("k")[["n_boxes", "n_invalid_boxes", "classes_present", "label_abspath"]]
        lab_map = lab_map[~lab_map.index.duplicated()]
        f["k"] = f.root + "|" + f.split.fillna("") + "|" + f.raw_folder.fillna("") + "|" + f.stem
        f = f.join(lab_map, on="k")
        f["has_label"] = f.label_abspath.notna()
        f["empty_label"] = f.has_label & (f.n_boxes == 0)
        self.files = f
        self.yolo = f[f.root == "yolo"].copy()
        self.raw = f[f.root == "raw"].copy()
        cid = lambda df: set(df.content_id.dropna().astype(int))
        self.c_train = cid(self.yolo[self.yolo.split == "train"])
        self.c_val = cid(self.yolo[self.yolo.split == "val"])
        self.c_yolo = self.c_train | self.c_val
        strong = self.edges[self.edges.confidence.isin(["HIGH", "MEDIUM"])]
        self.gid = sim.lineage_groups(len(self.contents), strong)
        self.contents["lineage_id"] = self.gid
        f["lineage_id"] = f.content_id.map(lambda c: self.gid[int(c)] if pd.notna(c) else -1)
        self.yolo["lineage_id"] = self.yolo.content_id.map(lambda c: self.gid[int(c)] if pd.notna(c) else -1)
        self.raw["lineage_id"] = self.raw.content_id.map(lambda c: self.gid[int(c)] if pd.notna(c) else -1)

    def add(self, text: str):
        self.md.append(text)

    # ─────────────────────────── 1. duplicates ───────────────────────────
    def check_duplicates(self):
        y, out = self.yolo.dropna(subset=["sha256"]), {}
        g = y.groupby("sha256")
        dup = g.filter(lambda d: len(d) > 1)
        rows = []
        for sha, d in dup.groupby("sha256"):
            splits = set(d.split)
            kind = "cross_split (LEAKAGE)" if splits == {"train", "val"} else f"within_{next(iter(splits))}"
            rows.append({"sha256": sha[:16], "n_files": len(d), "kind": kind, "splits": ",".join(sorted(splits)),
                         "files": "; ".join(sorted(d.relpath))})
        ex = pd.DataFrame(rows)
        ex.to_csv(self.run_dir / "results" / "exact_duplicates_yolo.csv", index=False)
        r = self.raw.dropna(subset=["sha256"])
        rdup = r.groupby("sha256").filter(lambda d: len(d) > 1)
        rrows = [{"sha256": s[:16], "n_files": len(d), "folders": ",".join(sorted(set(d.raw_folder))),
                  "files": "; ".join(sorted(d.relpath))} for s, d in rdup.groupby("sha256")]
        pd.DataFrame(rrows).to_csv(self.run_dir / "results" / "exact_duplicates_raw.csv", index=False)
        # pixel-identical but byte-different
        pix = y.groupby("pixel_sha256").sha256.nunique()
        n_pix = int((pix > 1).sum())
        # YOLO <-> raw byte mapping
        raw_sha = set(r.sha256)
        yolo_not_in_raw = y[~y.sha256.isin(raw_sha)]
        raw_not_in_yolo = r[~r.sha256.isin(set(y.sha256))]
        # near-duplicate edges touching YOLO images
        e = self.edges
        in_y = e.a.isin(self.c_yolo) & e.b.isin(self.c_yolo)
        ey = e[in_y]
        cross = ey[(ey.a.isin(self.c_train) & ey.b.isin(self.c_val)) | (ey.b.isin(self.c_train) & ey.a.isin(self.c_val))]
        groups = self.contents[self.contents.content_id.isin(self.c_yolo)].groupby("lineage_id").size()
        out = {
            "exact_duplicate_groups_yolo": int(len(ex)),
            "exact_duplicate_files_yolo": int(ex.n_files.sum()) if len(ex) else 0,
            "exact_duplicate_redundant_copies_yolo": int((ex.n_files - 1).sum()) if len(ex) else 0,
            "exact_duplicate_groups_by_kind": ex.kind.value_counts().to_dict() if len(ex) else {},
            "exact_duplicate_groups_raw": len(rrows),
            "pixel_identical_byte_different_groups_yolo": n_pix,
            "yolo_files_without_byte_identical_raw_copy": int(len(yolo_not_in_raw)),
            "raw_images_not_in_yolo_dataset": int(len(raw_not_in_yolo)),
            "raw_images_not_in_yolo_by_folder": raw_not_in_yolo.raw_folder.value_counts().to_dict(),
            "near_dup_edges_among_yolo_by_confidence": ey.confidence.value_counts().to_dict(),
            "near_dup_edges_among_yolo_by_relation_highmed": ey[ey.confidence != "LOW"].relation.value_counts().to_dict(),
            "cross_split_edges_by_confidence": cross.confidence.value_counts().to_dict(),
            "near_dup_clusters_yolo_size_ge2": int((groups >= 2).sum()),
            "images_in_near_dup_clusters_yolo": int(groups[groups >= 2].sum()),
        }
        self.summary["duplicates"] = out
        # detailed edge export with file names
        name = self.contents.abspath.map(lambda p: Path(p).name)
        split_of = {}
        for cidx, s in zip(self.yolo.content_id.dropna().astype(int), self.yolo.split):
            split_of.setdefault(cidx, set()).add(s)
        ex_edges = ey[ey.confidence != "LOW"].copy()
        ex_edges["file_a"] = ex_edges.a.map(name)
        ex_edges["file_b"] = ex_edges.b.map(name)
        ex_edges["splits_a"] = ex_edges.a.map(lambda c: ",".join(sorted(split_of.get(c, {"raw-only"}))))
        ex_edges["splits_b"] = ex_edges.b.map(lambda c: ",".join(sorted(split_of.get(c, {"raw-only"}))))
        ex_edges["lineage_id"] = ex_edges.a.map(lambda c: self.gid[c])
        ex_edges.to_csv(self.run_dir / "results" / "near_duplicate_edges_yolo.csv", index=False)
        self.add("## 1. Duplicates\n")
        self.add(f"* **Exact duplicates (SHA-256 of file bytes) inside the YOLO dataset:** {out['exact_duplicate_groups_yolo']} groups, "
                 f"{out['exact_duplicate_redundant_copies_yolo']} redundant copies. By kind: {out['exact_duplicate_groups_by_kind']}.")
        self.add(f"* Exact duplicates inside the raw class folders: {out['exact_duplicate_groups_raw']} groups. "
                 f"Pixel-identical but byte-different groups (YOLO): {n_pix}.")
        self.add(f"* Every YOLO image should be a byte copy of a raw image: {out['yolo_files_without_byte_identical_raw_copy']} are not. "
                 f"Raw images that never entered the YOLO dataset: {out['raw_images_not_in_yolo_dataset']} "
                 f"({out['raw_images_not_in_yolo_by_folder']}).")
        self.add(f"* **Near-duplicate pairs among YOLO images** (HIGH/MEDIUM confidence): "
                 f"{sum(v for k, v in out['near_dup_edges_among_yolo_by_confidence'].items() if k != 'LOW')}; "
                 f"in {out['near_dup_clusters_yolo_size_ge2']} clusters covering {out['images_in_near_dup_clusters_yolo']} unique images.")
        self.add(f"* By relation: {out['near_dup_edges_among_yolo_by_relation_highmed']}")
        self.add(f"* **Cross-split (train↔val) near-duplicate pairs:** {out['cross_split_edges_by_confidence']} → flagged as DATA LEAKAGE (see §9).")
        self.add("* Files: `results/exact_duplicates_yolo.csv`, `results/exact_duplicates_raw.csv`, `results/near_duplicate_edges_yolo.csv`, `results/lineage_groups.csv`.\n")

    # ─────────────────────────── 2. integrity ───────────────────────────
    def check_integrity(self):
        f = self.files
        status = f.status.fillna("not_scanned").value_counts().to_dict()
        ext_mismatch = f[((f.ext.isin([".jpg", ".jpeg"])) & (f.magic != "jpeg")) | ((f.ext == ".png") & (f.magic != "png"))]
        no_eoi = f[(f.magic == "jpeg") & (f.jpeg_has_eoi == False)]  # noqa: E712
        trailing = f[(f.magic == "jpeg") & (f.jpeg_trailing_bytes.fillna(0) > 0)]
        mn = f[["width", "height"]].min(axis=1)
        mx = f[["width", "height"]].max(axis=1)
        tiny = f[mn < 32]
        huge = f[mx > 8000]
        aspect = f[(mx / mn) > 6]
        orient = f[f.exif_orientation.fillna(1).astype(float) != 1]
        modes = f["mode"].value_counts().to_dict()
        bad = f[~f.status.isin(["ok"])]
        bad[["root", "relpath", "status", "magic", "pil_verify_error", "pil_decode_error"]].to_csv(
            self.run_dir / "results" / "integrity_problems.csv", index=False)
        orient[["root", "relpath", "exif_orientation", "width", "height"]].to_csv(
            self.run_dir / "results" / "exif_orientation_flags.csv", index=False)
        out = {
            "image_files_scanned": int(len(f)),
            "status_counts": status,
            "extension_vs_magic_mismatch": int(len(ext_mismatch)),
            "jpeg_missing_eoi_truncated": int(len(no_eoi)),
            "jpeg_trailing_bytes_after_eoi": int(len(trailing)),
            "min_side_lt_32px": int(len(tiny)), "max_side_gt_8000px": int(len(huge)), "aspect_gt_6": int(len(aspect)),
            "exif_orientation_not_1": int(len(orient)),
            "exif_orientation_not_1_yolo": int((orient.root == "yolo").sum()),
            "colour_modes": modes,
            "suspicious_files": int(len(self.suspicious)),
            "suspicious_non_image_in_image_dirs": int(self.suspicious.reasons.str.contains("non-image").sum()) if len(self.suspicious) else 0,
        }
        self.summary["integrity"] = out
        self.add("## 2. Corrupted / unreadable images\n")
        self.add(f"* {out['image_files_scanned']} image files fully decoded with Pillow (strict, truncation = error) and OpenCV. Status: {status}.")
        self.add(f"* Extension/content mismatches: {out['extension_vs_magic_mismatch']}; JPEGs missing the end-of-image marker (truncated): "
                 f"{out['jpeg_missing_eoi_truncated']}; JPEGs with bytes after the end marker: {out['jpeg_trailing_bytes_after_eoi']}.")
        self.add(f"* Extreme dimensions — min side < 32 px: {out['min_side_lt_32px']}, max side > 8000 px: {out['max_side_gt_8000px']}, "
                 f"aspect ratio > 6: {out['aspect_gt_6']}. Colour modes: {modes}.")
        self.add(f"* **EXIF orientation ≠ 1:** {out['exif_orientation_not_1']} files ({out['exif_orientation_not_1_yolo']} in the YOLO set). "
                 "Pillow ignores this tag but OpenCV (used by Ultralytics) applies it, so boxes on these images may be rotated relative to what the model sees.")
        if len(self.suspicious):
            self.add(f"* **Suspicious files ({len(self.suspicious)}):**\n")
            self.add(md_table(self.suspicious[["root", "relpath", "size_bytes", "reasons"]].head(30)))
        for sc in self.scripts:
            self.add(f"\n* **Code found inside an image folder:** `{sc['root']}/{sc['relpath']}` — "
                     + ("an Albumentations augmentation script" if sc["uses_albumentations"] else "a Python script")
                     + (f" that pads the folder to `target_total = {sc['target_total']}` images" if sc["target_total"] else "")
                     + (f", writing copies named `{sc['output_name_pattern']}`" if sc["output_name_pattern"] else "")
                     + f" with transforms {sc['transforms']}"
                     + (" (images only — boxes are NOT transformed, so the copies start unlabelled)" if not sc["transforms_boxes"] else "")
                     + ". It was evidently renamed by the same batch renamer as the images.")
        self.add("\n* Files: `results/integrity_problems.csv`, `results/exif_orientation_flags.csv`, `results/suspicious_files.csv`.\n")

    # ─────────────────────────── 3. class names ───────────────────────────
    def check_class_names(self):
        # Scan the COMMITTED code (git HEAD), so local fixes made after the audit started don't hide the original state.
        import subprocess

        code_lists = {}
        tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
        for rel in tracked:
            txt = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout
            for m in re.finditer(r"CLASS_NAMES\s*=\s*\[(.*?)\]", txt, re.S):
                names = re.findall(r"[\"']([^\"']+)[\"']", m.group(1))
                if len(names) == NUM_CLASSES:
                    code_lists[rel + " @HEAD"] = names
        yaml_lists = {}
        for yp in [PROJECT_ROOT / "dataset" / "dataset.yaml", YOLO_ROOT / "dataset.yaml"]:
            if yp.exists():
                import yaml

                yaml_lists[yp.relative_to(PROJECT_ROOT).as_posix()] = yaml.safe_load(yp.read_text()).get("names")
        ckpt = {}
        for wp in [PROJECT_ROOT / "weights" / n for n in ("yolo26s_best.pt", "yolo26m_best.pt", "rtdetr_l_best.pt")] + \
                  [PROJECT_ROOT / "archive" / "best.pt"]:
            if wp.exists():
                try:
                    import torch

                    ck = torch.load(wp, map_location="cpu", weights_only=False)
                    model = ck.get("ema") or ck.get("model")
                    names = getattr(model, "names", None) or ck.get("names")
                    ckpt[wp.relative_to(PROJECT_ROOT).as_posix()] = list(names.values()) if isinstance(names, dict) else names
                except Exception as e:
                    ckpt[wp.name] = f"<could not read: {e}>"[:120]
        variants = {i: Counter() for i in range(NUM_CLASSES)}
        for src in list(code_lists.values()) + list(yaml_lists.values()) + [v for v in ckpt.values() if isinstance(v, list)]:
            for i, n in enumerate(src[:NUM_CLASSES]):
                variants[i][n] += 1
        folders = {}
        for folder in sorted(self.raw.raw_folder.dropna().unique()):
            cls = int(self.raw[self.raw.raw_folder == folder].folder_class.iloc[0])
            folders[folder] = cls
            variants[cls][folder.split(" ", 1)[1]] += 1
        prefixes = self.files.groupby(["prefix", "prefix_class"]).size().reset_index(name="n_files")
        ids_present = self.boxes[self.boxes.valid.fillna(False)].groupby(["root", "cls"]).size().unstack(0).fillna(0).astype(int)
        rows = []
        for i in range(NUM_CLASSES):
            rows.append({"id": i, "canonical": CANONICAL_NAMES[i], "display": DISPLAY_NAMES[i],
                         "current dataset.yaml": CURRENT_YAML_NAMES[i],
                         "variants seen": "; ".join(f"'{k}'×{v}" for k, v in variants[i].most_common())})
        mapping = pd.DataFrame(rows)
        mapping.to_csv(self.run_dir / "results" / "class_name_variants.csv", index=False)
        with open(self.run_dir / "proposed" / "canonical_classes.yaml", "w") as fp:
            fp.write("# PROPOSED canonical class mapping — IDs unchanged (label files and checkpoints stay valid).\n")
            fp.write("# Not applied anywhere automatically.\nnames:\n")
            for i in range(NUM_CLASSES):
                fp.write(f"  {i}: {CANONICAL_NAMES[i]}   # display: \"{DISPLAY_NAMES[i]}\"; was: \"{CURRENT_YAML_NAMES[i]}\"\n")
        issues = []
        for i in range(NUM_CLASSES):
            vs = list(variants[i])
            low = {re.sub(r"[\s_]+", " ", v.strip().lower()) for v in vs}
            if len(vs) > 1:
                issues.append(f"class {i}: {len(vs)} spellings {vs}" + (" (case/space only)" if len(low) == 1 else ""))
        self.summary["class_names"] = {"code_lists": code_lists, "yaml": yaml_lists, "checkpoints": ckpt,
                                       "raw_folders": folders, "classes_txt": self.classes_txt, "issues": issues,
                                       "class_ids_present": ids_present.to_dict()}
        self.add("## 3. Class-name standardisation\n")
        self.add("Sources compared: both `dataset.yaml` files, raw folder names, `classes.txt` files, filename prefixes, every "
                 "`CLASS_NAMES` list in the code, and the class names stored inside each checkpoint.\n")
        self.add(md_table(mapping))
        self.add("\n**Inconsistencies found:**\n")
        self.add("\n".join(f"* {s}" for s in issues) or "* none")
        ctxt = {Path(k).parent.parent.parent.name: v for k, v in self.classes_txt.items()}
        distinct = {json.dumps(v) for v in ctxt.values()}
        self.add(f"\n* `classes.txt` files: {len(ctxt)} found, {len(distinct)} distinct contents"
                 + (f" — e.g. `{next(iter(ctxt.values()))}`" if ctxt else "") + ".")
        self.add(f"* Filename prefixes → implied class:\n\n{md_table(prefixes)}\n")
        self.add(f"* Class IDs actually present in label files (boxes):\n\n{md_table(ids_present.reset_index())}\n")
        self.add("* Proposed canonical mapping (IDs unchanged): `proposed/canonical_classes.yaml`. Labels were **not** changed.\n")

    # ─────────────────────────── 4. annotation format ───────────────────────────
    def check_format(self):
        lf = self.lab_files
        y = self.yolo
        rows = []
        for key, p in self.pairing.items():
            rows.append({"set": key, "images": p["n_images"], "labels": p["n_labels"],
                         "images w/o label": len(p["images_without_label"]),
                         "labels w/o image": len(p["labels_without_image"]),
                         "stem collisions": len(p["stem_collisions"])})
        pair_df = pd.DataFrame(rows)
        empties = y[y.empty_label]
        emp_by = empties.groupby(["split", "prefix"]).size().unstack(0).fillna(0).astype(int)
        out = {
            "format": "YOLO txt (class cx cy w h, normalised) — no COCO/VOC files present",
            "label_files": int(len(lf)),
            "polygon_lines": int(lf.n_polygon_lines.sum()), "bad_lines": int(lf.n_bad_lines.sum()),
            "float_class_ids": int(lf.n_float_class_ids.sum()), "files_with_bom": int(lf.has_bom.sum()),
            "files_with_crlf": int(lf.has_crlf.sum()), "blank_lines": int(lf.n_blank_lines.sum()),
            "exact_duplicate_lines": int(lf.n_exact_dup_lines.sum()),
            "yolo_empty_label_files": {s: int((empties.split == s).sum()) for s in ("train", "val")},
            "yolo_empty_label_rate": {s: float((empties.split == s).sum() / max(1, (y.split == s).sum())) for s in ("train", "val")},
            "pairing": rows,
        }
        self.summary["annotation_format"] = out
        self.add("## 4. Annotation format\n")
        self.add(f"* Format: **{out['format']}**. {out['label_files']} label files parsed.")
        self.add(f"* Malformed lines: {out['bad_lines']}; polygon/extra-field lines: {out['polygon_lines']}; float class IDs: "
                 f"{out['float_class_ids']}; BOM: {out['files_with_bom']}; CRLF: {out['files_with_crlf']}; blank lines: {out['blank_lines']}; "
                 f"identical duplicate lines: {out['exact_duplicate_lines']}.")
        self.add(f"* Image ↔ label pairing:\n\n{md_table(pair_df)}\n")
        self.add(f"* **Empty label files in the YOLO set:** train {out['yolo_empty_label_files']['train']} "
                 f"({100 * out['yolo_empty_label_rate']['train']:.1f}%), val {out['yolo_empty_label_files']['val']} "
                 f"({100 * out['yolo_empty_label_rate']['val']:.1f}%). Ultralytics treats these as weapon-free background images. "
                 "Because `clean_labels.py` deleted every box whose class differed from the folder, and the auto-labellers wrote nothing when "
                 "they found nothing, most of these are probably *unlabelled weapons*, not negatives (see the model probe in §5 and "
                 "`samples/empty_labels_random.jpg`).")
        self.add(f"\n  Empty label files by filename prefix:\n\n{md_table(emp_by.reset_index())}\n")
        self.add("* **Recommendation:** keep YOLO txt as the training format (native to Ultralytics), generated from a single manifest; also "
                 "export COCO JSON from the same manifest for evaluation with an independent COCO evaluator.\n")

    # ─────────────────────────── 5. boxes ───────────────────────────
    def check_boxes(self, probe: pd.DataFrame | None):
        b = self.boxes[self.boxes.root == "yolo"].copy()
        flags: Counter = Counter()
        for fl in b["flags"].fillna(""):
            for x in fl.split(";"):
                if x:
                    flags[x] += 1
        valid = b[b.valid.fillna(False) & b.cls.between(0, NUM_CLASSES - 1)]
        valid = valid.assign(bucket=pd.cut(valid.area_px, [-1, COCO_SMALL, COCO_MEDIUM, np.inf], labels=["small", "medium", "large"]))
        # area in training pixels (letterbox to 640 keeps aspect): scale = 640 / max(w, h)
        dims = self.yolo.set_index(self.yolo.split + "|" + self.yolo.stem)[["width", "height"]]
        dims = dims[~dims.index.duplicated()]
        k = valid.split + "|" + valid.stem
        mx = dims.reindex(k)[["width", "height"]].max(axis=1).to_numpy()
        valid["area_train_px"] = valid.area_px.to_numpy() * (640.0 / mx) ** 2
        valid["bucket_640"] = pd.cut(valid.area_train_px, [-1, COCO_SMALL, COCO_MEDIUM, np.inf], labels=["small", "medium", "large"])
        q = valid.groupby("cls").area_norm.describe(percentiles=[0.05, 0.5, 0.95])[["count", "5%", "50%", "95%"]]
        q.index = [DISPLAY_NAMES[int(i)] for i in q.index]
        buckets = pd.crosstab(valid.split, valid.bucket, normalize="index").round(4)
        buckets640 = pd.crosstab(valid.split, valid.bucket_640, normalize="index").round(4)
        out = {"boxes_total_yolo": int(len(b)), "boxes_valid": int(b.valid.fillna(False).sum()),
               "boxes_invalid": int((~b.valid.fillna(False)).sum()), "flag_counts": dict(flags),
               "coco_size_buckets_original_px": buckets.to_dict(orient="index"),
               "coco_size_buckets_at_640_letterbox": buckets640.to_dict(orient="index")}
        if probe is not None and len(probe):
            pe = probe[probe.empty_label]
            pl = probe[~probe.empty_label]
            out["model_probe"] = {
                "empty_label_images_probed": {s: int((pe.split == s).sum()) for s in ("train", "val")},
                "empty_label_with_consensus_detection": {s: int(((pe.split == s) & (pe.n_consensus_unexplained > 0)).sum()) for s in ("train", "val")},
                "empty_label_with_rtdetr_detection": {s: int(((pe.split == s) & (pe.n_rtdetr_hi > 0)).sum()) for s in ("train", "val")},
                "labelled_images_probed": {s: int((pl.split == s).sum()) for s in ("train", "val")},
                "labelled_with_consensus_unexplained_object": {s: int(((pl.split == s) & (pl.n_consensus_unexplained > 0)).sum()) for s in ("train", "val")},
            }
        self.summary["boxes"] = out
        b.to_csv(self.run_dir / "results" / "boxes_yolo_flagged.csv", index=False)
        self._fig_boxes(valid)
        self._sheets_boxes(b, probe)
        self.add("## 5. Bounding-box validity\n")
        self.add(f"* {out['boxes_total_yolo']} boxes in the YOLO set; **invalid: {out['boxes_invalid']}** "
                 "(non-finite, non-positive size, outside [0,1], wrong class ID, < 2 px, or malformed).")
        self.add(f"* Flags (a box can have several): {dict(flags)}")
        self.add(f"* Box area (fraction of image) per class:\n\n{md_table(q.reset_index().rename(columns={'index': 'class'}), '{:.4f}')}\n")
        self.add(f"* COCO size buckets by original pixels (small < 32², medium < 96²):\n\n{md_table(buckets.reset_index(), '{:.3f}')}\n")
        self.add(f"* Same buckets after the 640-px letterbox the models actually see:\n\n{md_table(buckets640.reset_index(), '{:.3f}')}\n")
        self.add("  Small objects are rare, so **AP_small is not estimable** on this data. The old benchmark's AP_small = 0.000 "
                 "means 'no small objects', not 'fails on small objects'; a dedicated small/occluded subset (roadmap Phase 1) is needed.")
        if "model_probe" in out:
            mp = out["model_probe"]
            self.add(f"* **Model-assisted completeness probe** (RT-DETR-L + YOLO26s at conf ≥ 0.5, two-model agreement required; indicative only):")
            self.add(f"  * Empty-label images with a confident consensus detection: train {mp['empty_label_with_consensus_detection']['train']}/"
                     f"{mp['empty_label_images_probed']['train']}, val {mp['empty_label_with_consensus_detection']['val']}/{mp['empty_label_images_probed']['val']}.")
            self.add(f"  * Labelled images with a confident consensus object that no label explains: train "
                     f"{mp['labelled_with_consensus_unexplained_object']['train']}/{mp['labelled_images_probed']['train']}, val "
                     f"{mp['labelled_with_consensus_unexplained_object']['val']}/{mp['labelled_images_probed']['val']}.")
            self.add("  * Train-split numbers are biased low (the models learned these images as negatives). See `samples/model_probe_*.jpg`.")
        self.add("* Visual checks: `samples/boxes_random.jpg`, `samples/boxes_suspicious.jpg`, `samples/empty_labels_random.jpg`. "
                 "Figures: `figures/fig_box_area.png`. Data: `results/boxes_yolo_flagged.csv`.\n")

    def _fig_boxes(self, valid: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        data = [np.sqrt(valid[valid.cls == c].area_norm) for c in range(NUM_CLASSES)]
        axes[0].boxplot(data, showfliers=False)
        axes[0].set_xticks(range(1, NUM_CLASSES + 1), DISPLAY_NAMES, rotation=35, ha="right")
        axes[0].set_ylabel("sqrt(box area / image area)")
        axes[0].set_title("Relative box size per class (YOLO set)")
        ct = pd.crosstab(valid.cls, valid.bucket_640, normalize="index")
        bottom = np.zeros(len(ct))
        for col, colr in zip(ct.columns, ["#d62728", "#ff7f0e", "#2ca02c"]):
            axes[1].bar(range(len(ct)), ct[col], bottom=bottom, label=str(col), color=colr)
            bottom += ct[col].to_numpy()
        axes[1].set_xticks(range(len(ct)), [DISPLAY_NAMES[int(i)] for i in ct.index], rotation=35, ha="right")
        axes[1].set_ylabel("share of boxes")
        axes[1].set_title("COCO size bucket at 640-px letterbox")
        axes[1].legend(loc="lower right")
        savefig(fig, self.run_dir / "figures" / "fig_box_area")

    def _sheets_boxes(self, allb, probe):
        rng = np.random.default_rng(0)
        y = self.yolo[self.yolo.has_label & ~self.yolo.empty_label]
        pick = y.iloc[rng.choice(len(y), min(24, len(y)), replace=False)]
        tiles = [tile(r.abspath, caption=f"{r.split}/{r.name}", boxes=gt_boxes_for(self.boxes, r.split, r.stem)) for r in pick.itertuples()]
        sheet(tiles, 6, self.run_dir / "samples" / "boxes_random.jpg", "Random labelled YOLO images with ground-truth boxes")
        sus = allb[allb["flags"].fillna("").str.contains("crosses|degenerate|tiny|huge|duplicate|extreme")]
        sus = sus.drop_duplicates(["split", "stem"]).head(24)
        tiles = []
        for r in sus.itertuples():
            row = self.yolo[(self.yolo.split == r.split) & (self.yolo.stem == r.stem)]
            if len(row):
                tiles.append(tile(row.iloc[0].abspath, caption=f"{r.stem}\n{r.flags}"[:88], boxes=gt_boxes_for(self.boxes, r.split, r.stem)))
        sheet(tiles, 6, self.run_dir / "samples" / "boxes_suspicious.jpg", "Boxes flagged as crossing border / tiny / huge / duplicate / extreme aspect")
        emp = self.yolo[self.yolo.empty_label]
        if len(emp):
            pick = emp.iloc[rng.choice(len(emp), min(24, len(emp)), replace=False)]
            sheet([tile(r.abspath, caption=f"{r.split}/{r.name}\nlabel file EMPTY") for r in pick.itertuples()], 6,
                  self.run_dir / "samples" / "empty_labels_random.jpg", "Random images whose label file is EMPTY (treated as weapon-free)")
        if probe is not None and len(probe):
            for subset, title in (("empty", "EMPTY label file but both detectors agree on a confident weapon"),
                                  ("labelled", "Labelled image with a confident consensus object that no label explains")):
                p = probe[(probe.empty_label if subset == "empty" else ~probe.empty_label) & (probe.n_consensus_unexplained > 0)]
                p = p.sort_values("max_conf_rtdetr", ascending=False).head(24)
                tiles = []
                for r in p.itertuples():
                    row = self.yolo[self.yolo.relpath == r.relpath]
                    if not len(row):
                        continue
                    pred = [(*bb, "") for bb in r.consensus_boxes]
                    gt = gt_boxes_for(self.boxes, r.split, row.iloc[0].stem)
                    cls = ",".join(DISPLAY_NAMES[int(c)] for c in r.consensus_classes.split(",") if c)
                    tiles.append(tile(row.iloc[0].abspath, caption=f"{r.split}/{Path(r.relpath).name}\nblue=pred {cls}; red=GT",
                                      boxes=gt, boxes2=pred))
                sheet(tiles, 6, self.run_dir / "samples" / f"model_probe_{subset}.jpg", title)

    # ─────────────────────────── 6. imbalance ───────────────────────────
    def check_imbalance(self):
        b = self.boxes[(self.boxes.root == "yolo") & self.boxes.valid.fillna(False) & self.boxes.cls.between(0, NUM_CLASSES - 1)]
        per_img = b.groupby(["split", "stem", "cls"]).size().reset_index(name="n")
        rows = []
        for c in range(NUM_CLASSES):
            pc = per_img[per_img.cls == c]
            rows.append({
                "class": DISPLAY_NAMES[c],
                "images_train": int((pc.split == "train").sum()), "images_val": int((pc.split == "val").sum()),
                "boxes_train": int(pc[pc.split == "train"].n.sum()), "boxes_val": int(pc[pc.split == "val"].n.sum()),
                "mean_inst_per_img": float(pc.n.mean()) if len(pc) else 0.0,
                "median_inst_per_img": float(pc.n.median()) if len(pc) else 0.0,
                "max_inst_per_img": int(pc.n.max()) if len(pc) else 0,
            })
        t = pd.DataFrame(rows)
        t["boxes_total"] = t.boxes_train + t.boxes_val
        t["box_share"] = t.boxes_total / t.boxes_total.sum()
        t["val_fraction"] = t.boxes_val / t.boxes_total
        tot = self.yolo.groupby(["prefix_class", "split"]).size().unstack().fillna(0).astype(int)
        empty = self.yolo[self.yolo.empty_label].groupby("prefix_class").size()
        t["files_in_folder"] = [int(tot.loc[c].sum()) if c in tot.index else 0 for c in range(NUM_CLASSES)]
        t["empty_label_files"] = [int(empty.get(c, 0)) for c in range(NUM_CLASSES)]
        share = 1.0 / NUM_CLASSES
        rare = t[t.box_share < 0.5 * share]["class"].tolist()
        dominant = t[t.box_share > 1.5 * share]["class"].tolist()
        out = {"per_class": rows, "imbalance_ratio_boxes": float(t.boxes_total.max() / max(1, t.boxes_total.min())),
               "imbalance_ratio_images": float((t.images_train + t.images_val).max() / max(1, (t.images_train + t.images_val).min())),
               "rare_classes(<5% of boxes)": rare, "dominant_classes(>15% of boxes)": dominant,
               "images_with_multiple_classes": int((per_img.groupby(["split", "stem"]).cls.nunique() > 1).sum())}
        self.summary["imbalance"] = out
        t.to_csv(self.run_dir / "results" / "class_balance.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(NUM_CLASSES)
        for ax, (a, bb, lab) in zip(axes, [("boxes_train", "boxes_val", "boxes"), ("images_train", "images_val", "images containing class")]):
            ax.bar(x, t[a], label="train", color="#1f77b4")
            ax.bar(x, t[bb], bottom=t[a], label="val", color="#ff7f0e")
            ax.set_xticks(x, DISPLAY_NAMES, rotation=35, ha="right")
            ax.set_ylabel(lab)
            ax.legend()
        axes[0].set_title("Boxes per class")
        axes[1].set_title("Images per class")
        savefig(fig, self.run_dir / "figures" / "fig_class_distribution")
        self.add("## 6. Class imbalance\n")
        self.add(md_table(t[["class", "images_train", "images_val", "boxes_train", "boxes_val", "box_share", "val_fraction",
                             "mean_inst_per_img", "median_inst_per_img", "files_in_folder", "empty_label_files"]]))
        self.add(f"\n* Imbalance ratio (max/min): boxes {out['imbalance_ratio_boxes']:.2f}, images {out['imbalance_ratio_images']:.2f}. "
                 f"Rare classes (< 5% of boxes): {rare or 'none'}; dominant (> 15%): {dominant or 'none'}. "
                 f"Images containing more than one class: {out['images_with_multiple_classes']} — consistent with `clean_labels.py` having "
                 "removed every off-folder box (real scenes with two weapon types end up single-class).")
        self.add("* **Effect on metrics:** box counts are fairly even because each class came from a ~3,000-image folder, so classic imbalance "
                 "is *not* the main threat. The real per-class differences come from (a) missing labels (empty files are concentrated in some "
                 "folders), (b) source/style confounds (§7) and (c) leakage (§9) — all of which distort per-class AP more than count imbalance.")
        self.add("* **Mitigation without synthetic data:** report macro *and* per-class AP with confidence intervals; use repeat-factor or "
                 "class-balanced sampling only if imbalance appears after cleaning; collect more *real* images for any class that falls below "
                 "the others after de-duplication; stratify the new split by class *and* lineage group.\n")

    # ─────────────────────── 7. original vs augmented, sources ───────────────────────
    def check_lineage(self):
        f = self.files.dropna(subset=["content_id"]).copy()
        f["content_id"] = f.content_id.astype(int)
        first = f.sort_values(["root", "relpath"]).groupby("content_id").first()
        names = f.groupby("content_id").name.apply(lambda s: ";".join(sorted(set(s))))
        c = self.contents.set_index("content_id").join(first[["prefix_class", "corners_dark_flat", "letterbox_fraction",
                                                               "is_grayscale", "mtime_date", "jpeg_quality", "jpeg_qsig"]])
        c["names"] = names
        c["aug_name"] = c.names.str.contains("aug", case=False)
        c["rotation_fill"] = c.corners_dark_flat.fillna(0) >= 2
        c["letterboxed"] = c.letterbox_fraction.fillna(0) >= 0.05
        c["fingerprint_score"] = (3 * c.aug_name + 2 * c.rotation_fill + 1 * c.letterboxed + 1 * c.is_grayscale.fillna(False).astype(int))
        c["area"] = c.width * c.height
        strong = self.edges[self.edges.confidence.isin(["HIGH", "MEDIUM"])]
        rel = {}
        for r in strong.itertuples():
            rel.setdefault(r.a, {})[r.b] = (r.relation, r.confidence)
            rel.setdefault(r.b, {})[r.a] = (r.relation, r.confidence)
        role, rel_to_src, conf = {}, {}, {}
        sizes = c.groupby("lineage_id").size()
        for lid, grp in c[c.lineage_id.isin(sizes[sizes > 1].index)].groupby("lineage_id"):
            order = grp.sort_values(["fingerprint_score", "area", "n_bytes"], ascending=[True, False, False])
            src = order.index[0]
            role[src] = "source_candidate"
            for m in order.index[1:]:
                role[m] = "derived"
                rc = rel.get(src, {}).get(m)
                rel_to_src[m] = rc[0] if rc else "transitive (via other group members)"
                conf[m] = rc[1] if rc else "MEDIUM"
        c["role"] = [role.get(i, "singleton") for i in c.index]
        c["relation_to_source"] = [rel_to_src.get(i, "") for i in c.index]
        c["relation_confidence"] = [conf.get(i, "") for i in c.index]
        self.contents_annot = c
        c.reset_index()[["content_id", "lineage_id", "role", "relation_to_source", "relation_confidence", "names", "prefix_class",
                         "width", "height", "n_bytes", "aug_name", "rotation_fill", "letterboxed", "is_grayscale", "jpeg_quality",
                         "mtime_date"]].to_csv(self.run_dir / "results" / "lineage_groups.csv", index=False)
        gs = c.groupby("lineage_id").size()
        by_class = c.groupby("prefix_class").agg(
            unique_images=("role", "size"), singletons=("role", lambda s: (s == "singleton").sum()),
            group_sources=("role", lambda s: (s == "source_candidate").sum()), derived=("role", lambda s: (s == "derived").sum()),
            aug_named=("aug_name", "sum"), rotation_fill=("rotation_fill", "sum"), letterboxed=("letterboxed", "sum"),
            grayscale=("is_grayscale", "sum"), median_jpeg_quality=("jpeg_quality", "median"))
        by_class.index = [DISPLAY_NAMES[int(i)] if 0 <= i < NUM_CLASSES else "?" for i in by_class.index]
        by_class["derived_share"] = by_class.derived / by_class.unique_images
        rel_counts = Counter(r for r in rel_to_src.values())
        # resolution & source confound
        rr = self.raw.dropna(subset=["width"]).copy()
        rr["res"] = rr.width.astype(int).astype(str) + "x" + rr.height.astype(int).astype(str)
        res_share = []
        for cl in range(NUM_CLASSES):
            sub = rr[rr.folder_class == cl]
            vc = sub.res.value_counts(normalize=True)
            dates = sub.mtime_date.value_counts()
            res_share.append({"class": DISPLAY_NAMES[cl], "top_resolution": vc.index[0], "top_res_share": float(vc.iloc[0]),
                              "n_distinct_resolutions": int(sub.res.nunique()),
                              "download_dates": "; ".join(f"{d}×{n}" for d, n in dates.items())})
        res_df = pd.DataFrame(res_share)
        confound = self._confound_test(rr)
        out = {"unique_contents": int(len(c)), "lineage_groups_size_ge2": int((gs > 1).sum()),
               "largest_group": int(gs.max()), "group_size_histogram": gs.value_counts().sort_index().to_dict(),
               "singletons(likely original)": int((c.role == "singleton").sum()),
               "group_sources(likely original)": int((c.role == "source_candidate").sum()),
               "derived(likely augmented/duplicate)": int((c.role == "derived").sum()),
               "derived_by_relation": dict(rel_counts), "by_class": by_class.reset_index().to_dict(orient="records"),
               "resolution_by_class": res_share, "source_confound_test": confound,
               "giant_groups(>50, review for chaining)": gs[gs > 50].to_dict()}
        self.summary["original_vs_augmented"] = out
        self._fig_lineage(gs, rr)
        self.add("## 7. Original vs augmented data, and source fingerprints\n")
        self.add("Lineage groups = connected components of HIGH/MEDIUM near-duplicate links (exact copies, pixel copies, flips/rotations, "
                 "resize/recompression, photometric variants, and embedding-confirmed crops/views). Within a group the *source candidate* "
                 "is the member with the fewest augmentation fingerprints (no 'aug' in name, no dark rotation-fill corners, no letterbox, "
                 "colour), then the largest resolution and file size. **Direction for flips is inherently ambiguous** (A=flip(B) ⇔ B=flip(A)), "
                 "so source/derived roles are heuristic; group membership is the robust part.\n")
        self.add(f"* Unique images (byte-level): {out['unique_contents']}; lineage groups with ≥ 2 members: {out['lineage_groups_size_ge2']} "
                 f"(largest: {out['largest_group']}). Group-size histogram: {out['group_size_histogram']}.")
        self.add(f"* Likely original: {out['singletons(likely original)']} singletons + {out['group_sources(likely original)']} group sources. "
                 f"**Likely derived (augmented or duplicated): {out['derived(likely augmented/duplicate)']}.** By relation to source: {dict(rel_counts)}.")
        self.add(f"\n{md_table(by_class.reset_index().rename(columns={'index': 'class'}), '{:.2f}')}\n")
        self.add("* **Source / resolution confound** (raw folders):\n")
        self.add(md_table(res_df, "{:.2f}"))
        self.add(f"\n* **Can the class be predicted without looking at the weapon?** A random forest given only file metadata "
                 f"(resolution, aspect, bytes/pixel, JPEG quality, progressive flag, EXIF/ICC presence) predicts the class with "
                 f"**{confound['metadata_only']['accuracy_mean']:.1%} ± {confound['metadata_only']['accuracy_std']:.1%}** accuracy "
                 f"(chance = {confound['chance']:.1%}); with global image statistics only (brightness, contrast, sharpness, noise, borders): "
                 f"**{confound['image_stats_only']['accuracy_mean']:.1%}**. Folds are grouped by lineage so near-duplicates cannot leak across folds. "
                 "High values mean the classes are separable by *source style*, so a detector can partly learn the dataset of origin "
                 "instead of the weapon — expect a drop on real CCTV footage.")
        # images whose copies sit in DIFFERENT class folders = conflicting labels for the same picture
        pcls = c.prefix_class
        strong = self.edges[self.edges.confidence.isin(["HIGH", "MEDIUM"])]
        xc = strong[strong.a.map(pcls) != strong.b.map(pcls)].copy()
        xc["class_a"] = xc.a.map(lambda i: DISPLAY_NAMES[int(pcls[i])] if 0 <= pcls[i] < NUM_CLASSES else "?")
        xc["class_b"] = xc.b.map(lambda i: DISPLAY_NAMES[int(pcls[i])] if 0 <= pcls[i] < NUM_CLASSES else "?")
        xc["file_a"], xc["file_b"] = xc.a.map(c.names), xc.b.map(c.names)
        xc[["file_a", "class_a", "file_b", "class_b", "relation", "confidence", "n_inliers", "cos"]].to_csv(
            self.run_dir / "results" / "cross_class_duplicates.csv", index=False)
        pairs = (xc.assign(p=[" / ".join(sorted(t)) for t in zip(xc.class_a, xc.class_b)]).p.value_counts().head(8))
        n_img = len(set(xc.a) | set(xc.b))
        out["cross_class_duplicate_links"] = int(len(xc))
        out["cross_class_duplicate_images"] = int(n_img)
        out["cross_class_pairs"] = pairs.to_dict()
        self.add(f"* **The same picture filed under two classes (label conflicts):** {len(xc)} verified links involving {n_img} unique "
                 f"images; most common class pairs: {pairs.to_dict()}. Sniper Rifle vs Shotgun / Assault Rifle / Pistol dominate, i.e. "
                 "photos of one gun were collected into several class folders. Each such image teaches the model two different labels "
                 "for one object. List: `results/cross_class_duplicates.csv`.")
        self.add("* Files: `results/lineage_groups.csv`, `figures/fig_lineage.png`, `samples/class_overview.jpg`.\n")

    def _confound_test(self, rr: pd.DataFrame) -> dict:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

        d = rr.dropna(subset=["width", "height", "n_bytes"]).copy()
        d["aspect"] = d.width / d.height
        d["bpp"] = 8 * d.n_bytes / (d.width * d.height)
        d["jq"] = d.jpeg_quality.fillna(-1)
        d["prog"] = d.progressive.fillna(False).astype(int)
        d["exif"] = d.n_exif_tags.fillna(0)
        d["icc"] = d.has_icc.fillna(False).astype(int)
        d["log_lap"] = np.log1p(d.laplacian_var.fillna(0))
        d["gray"] = d.is_grayscale.fillna(False).astype(int)
        meta = ["width", "height", "aspect", "bpp", "jq", "prog", "exif", "icc"]
        stats = ["mean_luma", "std_luma", "log_lap", "noise_sigma", "gray", "corners_flat", "letterbox_fraction"]
        y = d.folder_class.astype(int).to_numpy()
        groups = d.lineage_id.to_numpy()
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        res = {"n_images": int(len(d)), "chance": float(pd.Series(y).value_counts(normalize=True).max())}
        for name, cols in (("metadata_only", meta), ("image_stats_only", stats)):
            X = d[cols].fillna(0).to_numpy(dtype=float)
            clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=42)
            s = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="accuracy")
            res[name] = {"features": cols, "accuracy_mean": float(s.mean()), "accuracy_std": float(s.std())}
        return res

    def _fig_lineage(self, gs: pd.Series, rr: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        h = gs[gs > 1].value_counts().sort_index()
        axes[0].bar(h.index.astype(str), h.values, color="#6a3d9a")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("lineage group size (unique images)")
        axes[0].set_ylabel("number of groups (log)")
        axes[0].set_title("Near-duplicate lineage group sizes")
        top = rr.assign(res=rr.width.astype(int).astype(str) + "×" + rr.height.astype(int).astype(str))
        cats = top.res.value_counts().index[:5].tolist()
        top["res_cat"] = np.where(top.res.isin(cats), top.res, "other")
        ct = pd.crosstab(top.folder_class, top.res_cat, normalize="index")[cats + ["other"]]
        bottom = np.zeros(len(ct))
        for col in ct.columns:
            axes[1].bar(range(len(ct)), ct[col], bottom=bottom, label=col)
            bottom += ct[col].to_numpy()
        axes[1].set_xticks(range(len(ct)), [DISPLAY_NAMES[int(i)] for i in ct.index], rotation=35, ha="right")
        axes[1].set_ylabel("share of images")
        axes[1].set_title("Image resolution by class (source confound)")
        axes[1].legend(fontsize=8, loc="lower right")
        savefig(fig, self.run_dir / "figures" / "fig_lineage")
        rng = np.random.default_rng(1)
        tiles = []
        for cl in range(NUM_CLASSES):
            sub = rr[rr.folder_class == cl]
            for r in sub.iloc[rng.choice(len(sub), min(8, len(sub)), replace=False)].itertuples():
                tiles.append(tile(r.abspath, size=180, caption=f"{DISPLAY_NAMES[cl]}\n{r.name} {int(r.width)}x{int(r.height)}"))
        sheet(tiles, 8, self.run_dir / "samples" / "class_overview.jpg", "8 random raw images per class (rows = classes)")

    # ─────────────────────── 8 & 9. split-before-augmentation and leakage ───────────────────────
    def check_leakage(self):
        y = self.yolo.dropna(subset=["content_id"]).copy()
        y["content_id"] = y.content_id.astype(int)
        e = self.edges
        train_c, val_c = self.c_train, self.c_val
        best = {}  # val content -> best (conf, relation, ph_min, cos, partner)
        for r in e[e.confidence != "LOW"].itertuples():
            for v, t in ((r.a, r.b), (r.b, r.a)):
                if v in val_c and t in train_c:
                    cand = (CONF_ORDER[r.confidence], r.relation, r.ph_min, r.cos, t)
                    if v not in best or (cand[0], -cand[2], cand[3]) > (best[v][0], -best[v][2], best[v][3]):
                        best[v] = cand
        lineage_has_train = set(self.contents.loc[self.contents.content_id.isin(train_c), "lineage_id"])
        val = y[y.split == "val"].copy()
        val["exact_leak"] = val.content_id.isin(train_c)
        val["best_conf"] = val.content_id.map(lambda c: best[c][0] if c in best else -1)
        val["best_relation"] = val.content_id.map(lambda c: best[c][1] if c in best else "")
        val["best_ph_min"] = val.content_id.map(lambda c: best[c][2] if c in best else np.nan)
        val["best_cos"] = val.content_id.map(lambda c: best[c][3] if c in best else np.nan)
        val.loc[val.exact_leak, ["best_conf", "best_relation"]] = [2, "exact_copy"]
        val["leak_high"] = val.best_conf >= 2
        val["leak_highmed"] = val.best_conf >= 1
        val["leak_group"] = val.lineage_id.isin(lineage_has_train) | val.exact_leak
        n_val = len(val)
        per_class = val.groupby("prefix_class").agg(
            val_images=("relpath", "size"), exact=("exact_leak", "sum"), high=("leak_high", "sum"),
            high_or_medium=("leak_highmed", "sum"), group_level=("leak_group", "sum"))
        per_class.index = [DISPLAY_NAMES[int(i)] if 0 <= i < NUM_CLASSES else "?" for i in per_class.index]
        for col in ("exact", "high", "high_or_medium", "group_level"):
            per_class[col + "_%"] = 100 * per_class[col] / per_class.val_images
        rel_counts = val[val.leak_highmed].best_relation.value_counts().to_dict()
        # cross-split lineage groups
        rows = []
        for lid, grp in y.groupby("lineage_id"):
            sp = set(grp.split)
            if sp != {"train", "val"}:
                continue
            gc = set(grp.content_id)
            ge = e[(e.a.isin(gc)) & (e.b.isin(gc)) & (e.confidence != "LOW")]
            cross = ge[(ge.a.isin(train_c) & ge.b.isin(val_c)) | (ge.b.isin(train_c) & ge.a.isin(val_c))]
            exact_cross = bool(set(grp[grp.split == "train"].content_id) & set(grp[grp.split == "val"].content_id))
            tr = sorted(grp[grp.split == "train"].relpath)
            va = sorted(grp[grp.split == "val"].relpath)
            rows.append({
                "lineage_id": int(lid), "n_train": len(tr), "n_val": len(va), "n_test": 0,
                "train_files": "; ".join(Path(p).name for p in tr[:15]) + (f" (+{len(tr) - 15})" if len(tr) > 15 else ""),
                "val_files": "; ".join(Path(p).name for p in va[:15]) + (f" (+{len(va) - 15})" if len(va) > 15 else ""),
                "test_files": "",
                "relations": "; ".join(sorted(set(cross.relation))) + ("; exact_copy" if exact_cross else ""),
                "min_phash_distance": int(cross.ph_min.min()) if len(cross) else (0 if exact_cross else None),
                "max_embedding_cos": float(cross.cos.max()) if len(cross) else (1.0 if exact_cross else None),
                "confidence": "HIGH" if exact_cross or (len(cross) and (cross.confidence == "HIGH").any())
                else ("MEDIUM" if len(cross) else "TRANSITIVE"),
                "classes": ",".join(sorted({DISPLAY_NAMES[int(c)] for c in grp.prefix_class if 0 <= c < NUM_CLASSES})),
            })
        leak_groups = pd.DataFrame(rows).sort_values(["confidence", "n_val"], ascending=[True, False]) if rows else pd.DataFrame()
        leak_groups.to_csv(self.run_dir / "results" / "cross_split_leakage_groups.csv", index=False)
        val.to_csv(self.run_dir / "results" / "val_leakage_per_image.csv", index=False,
                   columns=["relpath", "prefix", "prefix_class", "lineage_id", "exact_leak", "best_relation", "best_ph_min",
                            "best_cos", "leak_high", "leak_highmed", "leak_group"])
        clean = val[~val.leak_group]
        (self.run_dir / "proposed" / "clean_val_subset.txt").write_text("\n".join(sorted(clean.abspath)) + "\n")
        # split-before-augmentation: observed vs expected spanning of lineage groups under random image-level assignment
        p_train = (y.split == "train").mean()
        grp = y.groupby("lineage_id").agg(k=("relpath", "size"), splits=("split", lambda s: frozenset(s)))
        grp = grp[grp.k >= 2]
        grp["kk"] = grp.k.clip(upper=6)
        grp["expected"] = grp.k.map(lambda k: sim.expected_spanning_rate(int(k), p_train))  # per group, its own size
        span = grp.assign(spans=grp.splits.map(lambda s: s == frozenset({"train", "val"}))).groupby("kk").agg(
            groups=("spans", "size"), observed_spanning=("spans", "mean"), expected_if_random_image_split=("expected", "mean"))
        span["expected_if_split_before_augmentation"] = 0.0
        span.index = [f"{k}" if k < 6 else "6+" for k in span.index]
        self.span_table = span
        out = {
            "val_images": int(n_val), "train_images": int((y.split == "train").sum()), "test_images": 0,
            "val_leaked_exact": int(val.exact_leak.sum()),
            "val_leaked_high": int(val.leak_high.sum()), "val_leaked_high_or_medium": int(val.leak_highmed.sum()),
            "val_leaked_group_level": int(val.leak_group.sum()),
            "val_leak_rate_high": float(val.leak_high.mean()), "val_leak_rate_high_or_medium": float(val.leak_highmed.mean()),
            "val_leak_rate_group": float(val.leak_group.mean()),
            "leak_relations": rel_counts, "cross_split_lineage_groups": int(len(leak_groups)),
            "cross_split_groups_by_confidence": leak_groups.confidence.value_counts().to_dict() if len(leak_groups) else {},
            "per_class": per_class.reset_index().to_dict(orient="records"),
            "clean_val_subset_size": int(len(clean)),
            "clean_val_by_class": clean.prefix_class.map(lambda c: DISPLAY_NAMES[int(c)] if 0 <= c < NUM_CLASSES else "?").value_counts().to_dict(),
            "spanning_test": span.reset_index().rename(columns={"index": "group_size"}).to_dict(orient="records"),
            "p_train": float(p_train),
        }
        self.summary["leakage"] = out
        self.val_leak = val
        self._fig_leakage(per_class, span)
        self._sheets_leakage(e)
        self.add("## 8. Split-before-augmentation\n")
        verdict = ("**NOT FOLLOWED (provably violated)**" if out["cross_split_lineage_groups"] > 0 else "**NOT VERIFIABLE**")
        self.add(f"Verdict: {verdict}. Evidence:\n")
        self.add("1. **The split script exists and splits images, not groups.** `archive/train_yolo.py` collects every (image, label) pair across "
                 "all class folders, `random.shuffle`s them with seed 42 (line 64) and cuts 80/20 (line 11). `int(28391 × 0.8) = 22712` equals "
                 "the current train count exactly, so the current split is this script's output. There is no test split.")
        self.add("2. **The source folders already contained augmented images before the split.** The SMG folder is 100% `smg_aug_*` files, and "
                 "flip/rotation/near-identical copies exist inside single raw folders (§7).")
        for sc in self.scripts:
            if sc["uses_albumentations"]:
                self.add(f"   * The augmentation script itself sits in the image folder (`{sc['relpath']}`): it pads each folder to "
                         f"{sc['target_total']} images with randomly chosen sources transformed by {', '.join(sc['transforms'][1:]) or sc['transforms']}. "
                         "Every class folder holds exactly 3,000 images, matching that target.")
                break
        self.add(f"3. **Lineage groups cross the split at the rate random image-level assignment predicts** (p_train = {p_train:.3f}). If the "
                 "split had been done before augmentation, no group would span train and val:\n")
        self.add(md_table(span.reset_index().rename(columns={"index": "group size (files)"}), "{:.3f}"))
        self.add(f"\n4. Timestamps: every label file was written on 2026-06-18 (when `clean_labels.py` ran); the split zip "
                 "`dataset/raw/sword_v2_full_dataset.zip` was written two minutes later; image timestamps keep their original per-source "
                 "download dates. Order of operations: source datasets (already augmented) → pseudo-labelling → label cleaning → image-level split.")
        self.add("\n**Correct procedure:** SOURCE IMAGES → GROUP/DEDUPLICATE (lineage groups from §7) → TRAIN/VAL/TEST SPLIT by group, "
                 "stratified by class → AUGMENT TRAIN ONLY (on the fly, in the training pipeline). A proposed group-aware split is in "
                 "`proposed/split_v3_grouped/`.\n")
        self.add("## 9. Transformed-image leakage (critical)\n")
        self.add(f"* **{out['val_leaked_high']} of {n_val} val images ({100 * out['val_leak_rate_high']:.1f}%) have a HIGH-confidence copy in train** "
                 f"(exact: {out['val_leaked_exact']}); {out['val_leaked_high_or_medium']} ({100 * out['val_leak_rate_high_or_medium']:.1f}%) with "
                 f"HIGH or MEDIUM; {out['val_leaked_group_level']} ({100 * out['val_leak_rate_group']:.1f}%) share a lineage group with train.")
        self.add(f"* Leaked val images by relation to their closest train copy: {rel_counts}")
        self.add(f"* Cross-split lineage groups: {out['cross_split_lineage_groups']} ({out['cross_split_groups_by_confidence']}). Every group "
                 "is listed with lineage ID, train files, val files, test files (none — no test split), similarity evidence and confidence in "
                 "`results/cross_split_leakage_groups.csv`; per-image flags in `results/val_leakage_per_image.csv`.\n")
        self.add(md_table(per_class.reset_index().rename(columns={"index": "class"})[
            ["class", "val_images", "exact", "high", "high_or_medium", "group_level", "high_%", "high_or_medium_%", "group_level_%"]], "{:.1f}"))
        if len(leak_groups):
            self.add("\nTop cross-split groups (largest val side first):\n")
            self.add(md_table(leak_groups.sort_values("n_val", ascending=False).head(12)[
                ["lineage_id", "n_train", "n_val", "relations", "min_phash_distance", "max_embedding_cos", "confidence", "classes"]], "{:.3f}"))
        self.add(f"\n* A **leakage-free val subset** (no lineage link to train at any confidence ≥ MEDIUM) of {out['clean_val_subset_size']} images "
                 f"is listed in `proposed/clean_val_subset.txt` ({out['clean_val_by_class']}). Re-scoring the models on it vs the full val set "
                 "measures how much leakage inflated the reported metrics. Visual evidence: `samples/leakage_*.jpg`.\n")

    def _fig_leakage(self, per_class: pd.DataFrame, span: pd.DataFrame):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        x = np.arange(len(per_class))
        w = 0.27
        for i, (col, lab, colr) in enumerate([("high_%", "HIGH", "#d62728"), ("high_or_medium_%", "HIGH+MEDIUM", "#ff7f0e"),
                                              ("group_level_%", "same lineage group", "#9467bd")]):
            axes[0].bar(x + (i - 1) * w, per_class[col], w, label=lab, color=colr)
        axes[0].set_xticks(x, per_class.index, rotation=35, ha="right")
        axes[0].set_ylabel("% of val images with a near-copy in train")
        axes[0].set_title("Train→val leakage by class")
        axes[0].set_ylim(0, 118)
        axes[0].legend(ncol=3, loc="upper center", fontsize=8)
        xs = np.arange(len(span))
        axes[1].bar(xs - 0.2, span.observed_spanning, 0.4, label="observed", color="#1f77b4")
        axes[1].bar(xs + 0.2, span.expected_if_random_image_split, 0.4, label="expected: random image-level split", color="#aec7e8")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_xticks(xs, span.index)
        axes[1].set_xlabel("lineage group size (files)")
        axes[1].set_ylabel("share of groups spanning train and val")
        axes[1].set_title("Split-before-augmentation test (0 if split was group-wise)")
        axes[1].legend(fontsize=8)
        savefig(fig, self.run_dir / "figures" / "fig_leakage")

    def _sheets_leakage(self, e: pd.DataFrame):
        path_of = self.contents.set_index("content_id").abspath
        train_rel = self.yolo[self.yolo.split == "train"].dropna(subset=["content_id"])
        train_path = dict(zip(train_rel.content_id.astype(int), train_rel.abspath))
        cross = e[(e.confidence != "LOW") & ((e.a.isin(self.c_train) & e.b.isin(self.c_val)) | (e.b.isin(self.c_train) & e.a.isin(self.c_val)))]
        rng = np.random.default_rng(2)
        for relation, sub in cross.groupby("relation"):
            sub = sub.iloc[rng.permutation(len(sub))[:10]]
            tiles = []
            for r in sub.itertuples():
                t, v = (r.a, r.b) if r.a in self.c_train else (r.b, r.a)
                tiles.append(tile(train_path.get(t, path_of[t]), caption=f"TRAIN {Path(train_path.get(t, path_of[t])).name}\nph={r.ph_min} T={r.T} cos={r.cos:.3f}"))
                tiles.append(tile(path_of[v], caption=f"VAL {Path(path_of[v]).name}\n{r.confidence}"))
            sheet(tiles, 6, self.run_dir / "samples" / f"leakage_{relation}.jpg", f"Train | Val pairs — {relation} ({len(cross[cross.relation == relation])} pairs)")
        # calibration sheets for every confidence/relation bucket (all pairs, not only cross-split)
        for (cf, relation), sub in e.groupby(["confidence", "relation"]):
            if cf == "LOW":
                sub = sub.sort_values(["cos", "ph_min"], ascending=[False, True]).head(12)  # borderline: strongest LOW pairs
            else:
                sub = sub.iloc[rng.permutation(len(sub))[:12]]
            tiles = []
            for r in sub.itertuples():
                tiles.append(tile(path_of[r.a], caption=f"{Path(path_of[r.a]).name}\nph_id={r.ph_id} ph_min={r.ph_min} {r.T}"))
                tiles.append(tile(path_of[r.b], caption=f"{Path(path_of[r.b]).name}\ncos={r.cos:.3f} dh={r.dh_T}"))
            sheet(tiles, 6, self.run_dir / "samples" / "calibration" / f"{cf}_{relation}.jpg", f"Calibration — {cf} / {relation} (n={len(e[(e.confidence == cf) & (e.relation == relation)])})")

    # ─────────────────────────── proposals ───────────────────────────
    def write_proposals(self):
        p = self.run_dir / "proposed"
        y = self.yolo.copy()
        lab_ok = y.has_label & ~y.empty_label & (y.n_invalid_boxes.fillna(0) == 0) & y.status.eq("ok")
        excluded = y[~lab_ok]
        (p / "needs_review_empty_labels.txt").write_text("\n".join(sorted(y[y.empty_label].abspath)) + "\n")
        pool = y[lab_ok].drop_duplicates("sha256")  # one copy per byte-identical image
        rng = np.random.default_rng(42)
        target = {"train": 0.70, "val": 0.15, "test": 0.15}
        assign = {}
        for cl, sub in pool.groupby("prefix_class"):
            groups = sub.groupby("lineage_id").size()
            # largest groups first (random tie-break), each to the split furthest below its target share
            order = sorted(groups.index, key=lambda g: (-groups[g], rng.random()))
            total = groups.sum()
            filled = {k: 0 for k in target}
            for g in order:
                if g in assign:  # group already placed via another class (cross-class duplicates)
                    continue
                split = max(target, key=lambda k: target[k] * total - filled[k])
                assign[g] = split
                filled[split] += groups[g]
        pool = pool.assign(new_split=pool.lineage_id.map(assign))
        # a lineage group can contain several prefixes; force the whole group into one split (first assignment wins)
        first = pool.groupby("lineage_id").new_split.first()
        pool["new_split"] = pool.lineage_id.map(first)
        d = p / "split_v3_grouped"
        d.mkdir(exist_ok=True)
        for s in ("train", "val", "test"):
            (d / f"{s}.txt").write_text("\n".join(sorted(pool[pool.new_split == s].abspath)) + "\n")
        (d / "sword_v3_grouped.yaml").write_text(
            "# PROPOSED dataset config — lineage-grouped, class-stratified 70/15/15 split (groups never cross splits;\n"
            "# largest groups placed first into the split furthest below target). Not used by any script yet.\n"
            "# Paths point into the existing dataset/yolo_dataset (labels resolve via /images/ -> /labels/). Nothing was copied or moved.\n"
            f"path: {d}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnc: {NUM_CLASSES}\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(CANONICAL_NAMES)))
        stats = pool.groupby(["prefix_class", "new_split"]).size().unstack().fillna(0).astype(int)
        stats.index = [DISPLAY_NAMES[int(i)] if 0 <= i < NUM_CLASSES else "?" for i in stats.index]
        cross = pool.groupby("lineage_id").new_split.nunique().gt(1).sum()
        split_stats = {"images": pool.new_split.value_counts().to_dict(), "per_class": stats.to_dict(orient="index"),
                       "excluded_images": int(len(excluded) + (len(y[lab_ok]) - len(pool))),
                       "excluded_reasons": {"empty_label": int(y.empty_label.sum()), "no_label": int((~y.has_label).sum()),
                                            "invalid_boxes": int((y.n_invalid_boxes.fillna(0) > 0).sum()),
                                            "integrity_not_ok": int((~y.status.eq("ok")).sum()),
                                            "byte_duplicate_copies": int(len(y[lab_ok]) - len(pool))},
                       "lineage_groups_spanning_splits": int(cross)}
        write_json(d / "split_stats.json", split_stats)
        self.summary["proposed_split"] = split_stats
        # manifest of the current dataset (one row per image file)
        ca = self.contents_annot
        man = self.files.dropna(subset=["content_id"]).copy()
        man["content_id"] = man.content_id.astype(int)
        man = man.join(ca[["role", "relation_to_source", "relation_confidence"]], on="content_id")
        lk = self.val_leak.set_index("relpath")[["leak_high", "leak_highmed", "leak_group"]]
        man = man.join(lk, on="relpath")
        man["proposed_split"] = man.relpath.map(dict(zip(pool.relpath, pool.new_split)))
        man["class_id_from_folder"] = man.prefix_class
        man["canonical_class_from_folder"] = man.prefix_class.map(lambda c: CANONICAL_NAMES[int(c)] if 0 <= c < NUM_CLASSES else "")
        man["image_id"] = man.sha256.str[:16]
        cols = ["image_id", "sha256", "root", "split", "relpath", "raw_folder", "class_id_from_folder", "canonical_class_from_folder",
                "width", "height", "n_bytes", "format", "status", "exif_orientation", "jpeg_quality", "mtime_date", "has_label",
                "empty_label", "n_boxes", "classes_present", "lineage_id", "role", "relation_to_source", "relation_confidence",
                "leak_high", "leak_highmed", "leak_group", "proposed_split"]
        man[cols].to_csv(p / "dataset_manifest_current.csv", index=False)
        write_json(p / "manifest.schema.json", MANIFEST_SCHEMA)
        (p / "directory_structure.md").write_text(DIRECTORY_STRUCTURE)
        q = self.suspicious.copy()
        if len(q):
            q["proposed_action"] = "exclude from dataset lists (do not delete)"
        q.to_csv(p / "quarantine_list.csv", index=False)

    # ─────────────────────────── verdict & report ───────────────────────────
    def verdict(self):
        s = self.summary
        lk, integ, fmt = s["leakage"], s["integrity"], s["annotation_format"]
        blocking = []
        if lk["val_leak_rate_high"] >= 0.01:
            blocking.append(f"Train→val leakage: {100 * lk['val_leak_rate_high']:.1f}% of val images have a HIGH-confidence copy in train "
                            f"({100 * lk['val_leak_rate_group']:.1f}% at group level). Split was image-level after augmentation.")
        blocking.append("No held-out test split; val was used for checkpoint selection (best.pt) and for ensemble tuning.")
        blocking.append("Label provenance unknown: an unrecorded share of train AND val labels were generated by a YOLOv8s model "
                        "(auto_label.py, auto_label_hard_images.py at conf 0.05 + TTA); file metadata cannot separate human from model labels.")
        er = fmt["yolo_empty_label_rate"]
        blocking.append(f"Empty label files: {100 * er['train']:.1f}% of train and {100 * er['val']:.1f}% of val — many likely unlabelled weapons "
                        "(clean_labels.py removed off-folder boxes), which trains the model to ignore weapons and corrupts val precision.")
        conf = s["original_vs_augmented"]["source_confound_test"]
        if conf["metadata_only"]["accuracy_mean"] > 3 * conf["chance"]:
            blocking.append(f"Class ↔ source confound: metadata alone predicts the class with {100 * conf['metadata_only']['accuracy_mean']:.0f}% "
                            f"accuracy (chance {100 * conf['chance']:.0f}%); results will not transfer to CCTV without an external test set.")
        minor = [f"{integ['suspicious_files']} stray/suspicious files (incl. .py/.ini/.txt inside images/train)",
                 "Inconsistent class names across yaml, folders, code and checkpoints ('Sniper Dataset', 'assault rifle', …)",
                 f"{integ['exif_orientation_not_1']} images with EXIF orientation ≠ 1"]
        grade = "D" if lk["val_leak_rate_high"] >= 0.01 or lk["val_leak_rate_group"] >= 0.05 else ("C" if len(blocking) > 1 else "B")
        s["verdict"] = {"grade": grade, "blocking_issues": blocking, "minor_issues": minor}
        return grade, blocking, minor

    def finish(self):
        grade, blocking, minor = self.verdict()
        s = self.summary
        lk, du = s["leakage"], s["duplicates"]
        bx, im = s["boxes"], s["imbalance"]
        ova = s["original_vs_augmented"]
        stats = [
            ("Total images (YOLO set: train + val)", f"{lk['train_images'] + lk['val_images']} (train {lk['train_images']}, val {lk['val_images']}, test 0)"),
            ("Raw source images", f"{int((self.files.root == 'raw').sum())} ({du['raw_images_not_in_yolo_dataset']} never entered the YOLO set)"),
            ("Total annotations (YOLO boxes)", f"{bx['boxes_total_yolo']} (valid {bx['boxes_valid']}, invalid {bx['boxes_invalid']})"),
            ("Valid / invalid images", f"{s['integrity']['status_counts']}"),
            ("Exact duplicates (YOLO)", f"{du['exact_duplicate_groups_yolo']} groups / {du['exact_duplicate_redundant_copies_yolo']} redundant copies"),
            ("Near-duplicate clusters (YOLO, ≥2 images)", f"{du['near_dup_clusters_yolo_size_ge2']} covering {du['images_in_near_dup_clusters_yolo']} images"),
            ("Corrupted images", f"{s['integrity']['status_counts'].get('corrupt', 0)} corrupt, {s['integrity']['jpeg_missing_eoi_truncated']} truncated JPEGs"),
            ("Class counts (boxes)", ", ".join(f"{r['class']} {r['boxes_train'] + r['boxes_val']}" for r in im["per_class"])),
            ("Invalid boxes", f"{bx['boxes_invalid']}"),
            ("Missing labels / images", f"empty label files: train {s['annotation_format']['yolo_empty_label_files']['train']}, val {s['annotation_format']['yolo_empty_label_files']['val']}; "
                                        f"images without label file / labels without image: see §4"),
            ("Original / augmented (heuristic)", f"{ova['singletons(likely original)'] + ova['group_sources(likely original)']} likely original, {ova['derived(likely augmented/duplicate)']} likely derived"),
            ("Suspected lineage groups (≥2)", f"{ova['lineage_groups_size_ge2']}"),
            ("Cross-split leakage", f"{lk['cross_split_lineage_groups']} groups; {lk['val_leaked_high']} val images HIGH ({100 * lk['val_leak_rate_high']:.1f}%), {lk['val_leaked_group_level']} group-level ({100 * lk['val_leak_rate_group']:.1f}%)"),
            ("Split proportions", f"train {lk['p_train']:.3f} / val {1 - lk['p_train']:.3f} / test 0"),
        ]
        head = [
            "# S.W.O.R.D. v2 — Dataset Research-Readiness Audit\n",
            f"*Run `{self.run_dir.name}` · git `{self.manifest['git']['commit'][:10]}` · {self.manifest['created_at']} · "
            f"{self.manifest['hardware'].get('cpu', '')} · read-only (dataset fingerprints verified unchanged — see Appendix)*\n",
            f"## 10. Verdict: **{grade}** — "
            + {"A": "research-ready", "B": "usable after minor cleanup", "C": "significant cleanup required",
               "D": "serious leakage/design problems; do not use for final paper results yet"}[grade] + "\n",
            md_table(pd.DataFrame(stats, columns=["Item", "Value"])),
            "\n**Blocking issues**\n",
            "\n".join(f"{i}. {b}" for i, b in enumerate(blocking, 1)),
            "\n**Minor issues**\n",
            "\n".join(f"* {m}" for m in minor),
            "\n**Prioritised remediation plan** (nothing below has been applied — each step needs your approval)\n",
            REMEDIATION,
            "\nThe sections below follow the order of the audit brief (checks 1–9).\n",
        ]
        figs = ["\n## Figures\n"] + [f"![{p.stem}](figures/{p.name})" for p in sorted((self.run_dir / "figures").glob("*.png"))]
        appendix = ["\n## Appendix A — Methods and thresholds\n", METHODS.format(
            selftest=json.dumps(self.selftest["vs_imagehash_reference_hamming"]),
            d4=json.dumps(self.selftest["d4_argmin_recovers_true_transform_rate"]),
            rules=json.dumps(sim.RULES), cand_ph=sim.CANDIDATE_PH_MAX, cand_cos=sim.CANDIDATE_COS_MIN, knn=sim.KNN_K,
            v_hi=sim.VERIFY["strong_inliers"], v_med=sim.VERIFY["weak_inliers"], v_smin=sim.VERIFY["scale_min"],
            v_smax=sim.VERIFY["scale_max"], v_span=sim.VERIFY["min_inlier_span"], v_cos=sim.VERIFY["emb_override_cos"],
            cal_n=(self.sensitivity or {}).get("n_sources", 0), cal_recall=(self.sensitivity or {}).get("recall_final_high_or_medium", float("nan")),
            cal_by=", ".join(f"{t['transform']} {t['detected_final']:.0%}" for t in (self.sensitivity or {}).get("by_transform", [])),
            rand_cos=(f"median {np.median(self.rand_cos):.3f}, 99.9th pct {np.percentile(self.rand_cos, 99.9):.3f}" if len(self.rand_cos) else "n/a"))]
        appendix += ["\n## Appendix B — Provenance (zip snapshots vs current files)\n", "```json", json.dumps(self.zip_prov, indent=1)[:4000], "```"]
        after = {"yolo_dataset": dataset_fingerprint(YOLO_ROOT), "raw": dataset_fingerprint(RAW_ROOT)}
        before = self.manifest["dataset_fingerprints_before"]
        unchanged = all(after[k]["listing_sha256"] == before[k]["listing_sha256"] for k in after)
        write_json(self.run_dir / "results" / "read_only_verification.json", {"before": before, "after": after, "unchanged": unchanged})
        appendix += ["\n## Appendix C — Read-only verification\n",
                     f"Dataset listing fingerprints (path, size, mtime of every file) before and after the audit are "
                     f"{'**identical**' if unchanged else '**DIFFERENT — investigate**'} (`results/read_only_verification.json`).",
                     "\n## Appendix D — Reproduce\n", "```bash",
                     "venv/bin/python research/tools/audit/run_audit.py --run-id <new_id>",
                     "venv/bin/python research/tools/audit/report.py --run-id <new_id>", "```",
                     "The exact code used for this run is in `code_snapshot/` (hashes in `manifest.json`)."]
        (self.run_dir / "REPORT.md").write_text("\n".join(head + self.md + figs + appendix) + "\n")
        s["read_only_unchanged"] = unchanged
        write_json(self.run_dir / "results" / "summary.json", s)
        self._latex()
        append_index(self.run_dir, "dataset_audit", {
            "grade": grade, "val_leak_rate_high": round(lk["val_leak_rate_high"], 4),
            "val_leak_rate_group": round(lk["val_leak_rate_group"], 4), "cross_split_groups": lk["cross_split_lineage_groups"],
            "exact_dup_groups": du["exact_duplicate_groups_yolo"], "invalid_boxes": bx["boxes_invalid"],
            "empty_label_val": s["annotation_format"]["yolo_empty_label_files"]["val"], "read_only_unchanged": unchanged})
        return grade

    def _latex(self):
        im = pd.DataFrame(self.summary["imbalance"]["per_class"])
        pc = pd.DataFrame(self.summary["leakage"]["per_class"]).rename(columns={"index": "class"})
        t = im.merge(pc, on="class", how="left").fillna(0)
        lines = [r"\begin{table}[t]", r"\centering", r"\caption{SWORD v2 dataset composition and train$\to$val near-duplicate leakage.}",
                 r"\label{tab:dataset_audit}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
                 r"Class & Img$_{tr}$ & Img$_{val}$ & Box$_{tr}$ & Box$_{val}$ & Leak$_{HIGH}$ (\%) & Leak$_{group}$ (\%) \\", r"\midrule"]
        for r in t.to_dict(orient="records"):
            n = r["val_images"]
            hi = 100.0 * r["high"] / n if n else 0.0
            grp = 100.0 * r["group_level"] / n if n else 0.0
            lines.append(f"{r['class']} & {int(r['images_train'])} & {int(r['images_val'])} & {int(r['boxes_train'])} & "
                         f"{int(r['boxes_val'])} & {hi:.1f} & {grp:.1f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (self.run_dir / "results" / "table_dataset_audit.tex").write_text("\n".join(lines) + "\n")


REMEDIATION = """1. **Freeze the evidence.** Keep this audit run; record the original Kaggle/Roboflow source URLs per class and, if possible,
   re-download the pristine sources to recover which labels were human-made (only way to establish label provenance).
2. **Rebuild the split by lineage group** (`proposed/split_v3_grouped/`: 70/15/15, class-stratified, one copy per byte-identical image,
   empty/invalid-label images held out for review). Never place two members of a lineage group in different splits.
3. **Fix the labels before retraining.** Review `proposed/needs_review_empty_labels.txt` and the model-probe sheets; re-annotate images
   whose weapons were deleted by `clean_labels.py` (multi-class scenes) and spot-audit a random sample of pseudo-labels per class.
   Annotate the test split by hand (100% human-verified) — it is the only set the paper's numbers may come from.
4. **Add an external test set** of real CCTV-style footage (different sources/resolutions) plus weapon-free footage for false-alarm rates;
   this is the only defence against the source confound in §7.
5. **Augment train only, on the fly** (Ultralytics mosaic/flip/HSV), never as stored files that can cross splits.
6. **Retrain and re-evaluate** all detectors with identical budgets on the new split; select checkpoints on val, tune ensembles on val,
   report once on test (with bootstrap confidence intervals), and additionally on the external set.
7. Housekeeping: exclude the files in `proposed/quarantine_list.csv` from dataset lists (do not delete), adopt the canonical class names
   (`proposed/canonical_classes.yaml`), and generate YOLO labels + COCO JSON from one manifest (`proposed/manifest.schema.json`)."""

METHODS = """* **Exact duplicates:** SHA-256 of file bytes; pixel-identical copies via SHA-256 of decoded RGB pixels.
* **Perceptual hashes:** each image → 64×64 Lanczos greyscale thumbnail → pHash (32×32 DCT, top-left 8×8 > median), dHash (9×8 gradient),
  aHash (8×8 mean), each computed for all 8 flips/rotations (D4). Self-test on 200 images: Hamming distance to the `imagehash` reference
  {selftest}; true-transform recovery rate {d4}.
* **Candidate pairs:** all pairs with pHash distance ≤ {cand_ph} under any transform (exhaustive, vectorised), plus {knn} nearest neighbours
  in DINOv2 ViT-S/14 embedding space with cosine ≥ {cand_cos}. Random-pair cosine distribution: {rand_cos}.
* **Rules → verification:** hash/embedding rules ({rules}) only propose a relation type. Every candidate except pixel-identical
  copies must pass **geometric verification**: SIFT (300 keypoints, 480-px greyscale; image b mirrored when the hash transform is a
  reflection) + Lowe ratio 0.75 + RANSAC similarity transform; HIGH ≥ {v_hi} inliers, MEDIUM ≥ {v_med}, with a plausible scale
  ({v_smin}–{v_smax}) and inliers spanning ≥ {v_span:.0%} of the image. Pairs with DINOv2 cosine ≥ {v_cos} are accepted without
  verification (zoom/crop variants of low-texture product shots). Why: thresholds alone could not separate "same photo, augmented"
  from "different gun, same studio style"; 10–15 inliers admitted degenerate RANSAC fits that chained 20,577 images of 8 classes
  into one group; hash-only rules linked different thin objects on white backgrounds. Precision was checked visually on random
  samples of every bucket (`samples/calibration/`). LOW pairs are never used for grouping or leakage.
* **Measured sensitivity:** re-running the dataset's own Albumentations script on {cal_n} random images (2 copies each), the full
  chain detects **{cal_recall:.1%}** of known copies as HIGH/MEDIUM (per transform: {cal_by}). Leakage figures are therefore lower
  bounds (true leakage is somewhat higher).
* **Lineage groups:** union–find over HIGH and MEDIUM links among all unique images (YOLO and raw).
* **Leakage:** a val image is leaked if it is byte-identical to a train image or has a HIGH (or HIGH/MEDIUM) link to one; group-level
  leakage additionally counts transitive links through other group members.
* **Split-before-augmentation test:** for groups of k files, the probability of spanning both splits under random image-level assignment
  is 1 − p^k − (1 − p)^k (p = train share); a group-wise split gives 0.
* **Source confound:** 5-fold StratifiedGroupKFold (groups = lineage) random forest on metadata-only and image-statistics-only features.
* **Model probe:** RT-DETR-L and YOLO26s (this project's checkpoints) at conf ≥ 0.5; a detection is 'unexplained' if its IoU with every GT box
  is < 0.3, and 'consensus' if both models agree (same class, IoU ≥ 0.5). Indicative only — not ground truth.
* **Limitations:** embedding similarity can link different photos of the same product shot (counted as MEDIUM/HIGH 'embedding' relations —
  inspect `samples/calibration/`); flip direction cannot be determined; label provenance (human vs model) is not recoverable from files."""

MANIFEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SWORD dataset manifest row (one per image)",
    "type": "object",
    "required": ["image_id", "sha256", "source_dataset", "source_url", "license", "file_path", "width", "height", "lineage_id",
                 "split", "is_augmented", "label_path", "label_provenance", "annotator", "verified"],
    "properties": {
        "image_id": {"type": "string", "description": "stable ID, e.g. first 16 hex chars of sha256"},
        "sha256": {"type": "string"},
        "source_dataset": {"type": "string", "description": "e.g. 'kaggle:<owner>/<slug>' or 'roboflow:<workspace>/<project>/<version>'"},
        "source_url": {"type": "string"}, "license": {"type": "string"},
        "source_file_name": {"type": "string", "description": "name inside the original download"},
        "downloaded_at": {"type": "string", "format": "date"},
        "file_path": {"type": "string", "description": "path relative to dataset/source/"},
        "width": {"type": "integer"}, "height": {"type": "integer"},
        "lineage_id": {"type": "string", "description": "near-duplicate group; the split unit"},
        "parent_image_id": {"type": ["string", "null"], "description": "set for stored derivatives (should not exist in a clean dataset)"},
        "is_augmented": {"type": "boolean"}, "augmentation": {"type": ["string", "null"]},
        "split": {"enum": ["train", "val", "test", "external_test", "excluded"]},
        "exclusion_reason": {"type": ["string", "null"]},
        "label_path": {"type": "string"}, "n_boxes": {"type": "integer"},
        "classes": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 9}},
        "label_provenance": {"enum": ["human", "model_pseudo_label", "model_then_human_verified", "unknown"]},
        "pseudo_label_model": {"type": ["string", "null"]}, "annotator": {"type": ["string", "null"]},
        "verified": {"type": "boolean"}, "verified_by": {"type": ["string", "null"]}, "verified_at": {"type": ["string", "null"]},
        "scene_type": {"enum": ["cctv", "staged", "studio_product", "web_photo", "game_render", "other", "unknown"]},
        "is_negative": {"type": "boolean", "description": "verified weapon-free image"},
        "difficulty_tags": {"type": "array", "items": {"enum": ["small", "occluded", "low_light", "motion_blur", "crowded"]}},
        "notes": {"type": "string"},
    },
}

DIRECTORY_STRUCTURE = """# Proposed research-grade dataset layout

```
dataset/
  source/                      # immutable original downloads, exactly as obtained (read-only)
    <source_name>/...          # one folder per Kaggle/Roboflow source + SOURCE.md (URL, licence, date, version)
  cleaned/                     # de-duplicated, re-annotated images + YOLO labels (one copy per image)
    images/  labels/
  manifests/
    manifest.csv               # one row per image (schema: manifest.schema.json)
    lineage_groups.csv         # near-duplicate groups (the split unit)
    changes.log                # every exclusion/relabel with reason, who, when
  splits/
    v3_grouped/                # file lists only — no copies
      train.txt  val.txt  test.txt  external_test.txt
      dataset.yaml             # Ultralytics config pointing at the lists
  augmentation/                # augmentation *configs* only (applied on the fly to train)
  external_test/               # real CCTV clips/frames, weapon-free footage for false-alarm rates
  audit/                       # audit runs (this report), never edited by hand
```

Rules: `source/` is never modified; every derived file is reproducible from `source/` + `manifests/` + scripts; the test split is
human-verified and frozen before any model is trained on the new split.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-model-probe", action="store_true")
    ap.add_argument("--probe-labelled-sample", type=int, default=600)
    args = ap.parse_args()
    run_dir = RESEARCH_ROOT / "runs" / args.run_id
    logger = setup_logging(run_dir)
    import shutil
    for stale in ("samples", "figures"):  # sheets/figures from an earlier report run must not survive a re-run
        shutil.rmtree(run_dir / stale, ignore_errors=True)
        (run_dir / stale).mkdir()
    a = Audit(run_dir, logger)
    probe = None
    if not args.skip_model_probe:
        cp = run_dir / "cache" / "model_probe.pkl"
        if cp.exists():
            probe = pd.read_pickle(cp)
        else:
            import model_probe

            y = a.yolo[a.yolo.status.eq("ok") & a.yolo.has_label]
            emp = y[y.empty_label]
            lab = y[~y.empty_label]
            rng = np.random.default_rng(42)
            samp = pd.concat([lab[lab.split == s].iloc[rng.choice((lab.split == s).sum(), min(args.probe_labelled_sample, (lab.split == s).sum()), replace=False)]
                              for s in ("train", "val")])
            targets = pd.concat([emp, samp])
            gt = {}
            vb = a.boxes[(a.boxes.root == "yolo") & a.boxes.valid.fillna(False)]
            key_rel = dict(zip(a.yolo.split + "|" + a.yolo.stem, a.yolo.relpath))
            for (sp, stem), g in vb.groupby(["split", "stem"]):
                rp = key_rel.get(f"{sp}|{stem}")
                if rp:
                    gt[rp] = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            probe = model_probe.run_probe(targets, gt, args.device, logger=logger)
            probe.to_pickle(cp)
        probe.drop(columns=["consensus_boxes"]).to_csv(run_dir / "results" / "model_probe.csv", index=False)
    steps = [("1 duplicates", a.check_duplicates), ("2 integrity", a.check_integrity), ("3 class names", a.check_class_names),
             ("4 format", a.check_format), ("5 boxes", lambda: a.check_boxes(probe)), ("6 imbalance", a.check_imbalance),
             ("7 lineage", a.check_lineage), ("8-9 leakage", a.check_leakage), ("proposals", a.write_proposals)]
    for name, fn in steps:
        logger.info(f"[report] {name}")
        fn()
    grade = a.finish()
    logger.info(f"REPORT written: {run_dir / 'REPORT.md'} — verdict {grade}")


if __name__ == "__main__":
    main()
