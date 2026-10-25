"""Shared helpers for the SWORD dataset audit: paths, class maps, run folders, manifests.

Nothing in this module (or anywhere in the audit) writes to the dataset folders.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = PROJECT_ROOT / "research"
DATASET_ROOT = PROJECT_ROOT / "dataset"
YOLO_ROOT = DATASET_ROOT / "yolo_dataset"
RAW_ROOT = DATASET_ROOT / "raw" / "Weapons_Dataset_30k_10_Classes"
ZIP_SNAPSHOTS = {
    "yolo_dataset.zip (2026-09-22)": DATASET_ROOT / "yolo_dataset.zip",
    "raw/sword_v2_full_dataset.zip (2026-06-18)": DATASET_ROOT / "raw" / "sword_v2_full_dataset.zip",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
NUM_CLASSES = 10

# Class ID -> names. IDs are what the label files and trained checkpoints use; they are NOT changed.
CURRENT_YAML_NAMES = [
    "Pistol", "Shotgun", "assault rifle", "grenade", "knife",
    "sword", "Sniper Dataset", "SMG", "rocket launcher", "Machine gun",
]
CANONICAL_NAMES = [
    "pistol", "shotgun", "assault_rifle", "grenade", "knife",
    "sword", "sniper_rifle", "smg", "rocket_launcher", "machine_gun",
]
DISPLAY_NAMES = [
    "Pistol", "Shotgun", "Assault Rifle", "Grenade", "Knife",
    "Sword", "Sniper Rifle", "SMG", "Rocket Launcher", "Machine Gun",
]

# Filename prefix (lower-cased, trailing digits stripped) -> class ID implied by the source folder.
PREFIX_TO_CLASS = {
    "pistol": 0, "shotgun": 1, "rifle": 2, "grenade": 3, "knife": 4, "sword": 5,
    "sniper": 6, "smg_aug": 7, "rocket_launcher": 8, "machine_gun": 9,
}
RAW_FOLDER_TO_CLASS = {
    "1 Pistol": 0, "2 Shotgun": 1, "3 assault rifle": 2, "4 grenade": 3, "5 knife": 4,
    "6 sword": 5, "7 Sniper Dataset": 6, "8 SMG": 7, "9 rocket launcher": 8, "10 Machine gun": 9,
}

_PREFIX_RE = re.compile(r"[_\- ]*\d+$")


def filename_prefix(name: str) -> str:
    """'Pistol_10006.jpg' -> 'pistol', 'machine_gun1.jpg' -> 'machine_gun', 'smg_aug_7.jpg' -> 'smg_aug'."""
    return _PREFIX_RE.sub("", Path(name).stem).lower()


def prefix_class(name: str) -> int:
    return PREFIX_TO_CLASS.get(filename_prefix(name), -1)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            block = fp.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def git_info() -> dict:
    porcelain = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote": _run(["git", "remote", "get-url", "origin"]),
        "dirty": bool(porcelain),
        "dirty_files": porcelain.splitlines()[:50],
    }


def package_versions(names: list[str]) -> dict:
    from importlib import metadata

    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = _run(["sysctl", "-n", "hw.memsize"])
        info["ram_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
    try:
        import torch

        info["torch_mps"] = bool(torch.backends.mps.is_available())
        info["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return info


def dataset_fingerprint(root: Path) -> dict:
    """SHA-256 over the sorted (relative path, size, mtime) listing: changes if any file is added, removed or touched."""
    rows = []
    total = 0
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = Path(dirpath) / f
            st = p.stat()
            total += st.st_size
            rows.append(f"{p.relative_to(root).as_posix()}|{st.st_size}|{st.st_mtime_ns}")
    rows.sort()
    return {
        "root": str(root),
        "n_files": len(rows),
        "total_bytes": total,
        "listing_sha256": hashlib.sha256("\n".join(rows).encode()).hexdigest(),
    }


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2, cls=NumpyJSONEncoder)


def make_run_dir(test_name: str, run_id: str | None = None) -> Path:
    run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{test_name}"
    run_dir = RESEARCH_ROOT / "runs" / run_id
    for sub in ("cache", "results", "figures", "samples", "proposed", "logs", "code_snapshot"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("sword_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(run_dir / "logs" / "audit.log"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def snapshot_code(run_dir: Path) -> dict:
    """Copy the audit code into the run folder so the exact version that produced the results is kept."""
    src = Path(__file__).resolve().parent
    hashes = {}
    for f in sorted(src.glob("*.py")):
        shutil.copy2(f, run_dir / "code_snapshot" / f.name)
        hashes[f.name] = sha256_file(f)
    return hashes


def write_manifest(run_dir: Path, test_name: str, config: dict, extra: dict | None = None) -> dict:
    manifest = {
        "run_id": run_dir.name,
        "test": test_name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "git": git_info(),
        "hardware": hardware_info(),
        "packages": package_versions([
            "numpy", "torch", "torchvision", "ultralytics", "opencv-python", "pillow", "scipy",
            "pandas", "scikit-learn", "timm", "imagehash", "matplotlib",
        ]),
        "config": config,
        "code_sha256": snapshot_code(run_dir),
    }
    if extra:
        manifest.update(extra)
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def append_index(run_dir: Path, test_name: str, metrics: dict) -> None:
    """One row per run in research/INDEX.csv so runs can be compared over time."""
    index = RESEARCH_ROOT / "INDEX.csv"
    row = {"run_id": run_dir.name, "test": test_name, "finished_at": datetime.now().isoformat(timespec="seconds"),
           "git_commit": git_info()["commit"][:10], **metrics}
    existing = []
    if index.exists():
        with open(index) as fp:
            existing = list(csv.DictReader(fp))
    fields = list(dict.fromkeys([k for r in existing for k in r] + list(row)))
    with open(index, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in existing + [row]:
            w.writerow(r)


class Timer:
    def __init__(self, logger: logging.Logger, label: str):
        self.logger, self.label = logger, label

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"[start] {self.label}")
        return self

    def __exit__(self, *exc):
        self.logger.info(f"[done ] {self.label} in {time.time() - self.t0:.1f}s")
