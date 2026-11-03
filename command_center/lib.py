"""Shared helpers for the S.W.O.R.D. command centre: paths, model registry, system status, service client, run index."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS = PROJECT_ROOT / "research" / "runs"
DATA = PROJECT_ROOT / "research" / "data"
WEIGHTS = PROJECT_ROOT / "weights"
UPLOADS = PROJECT_ROOT / "service_data" / "command_center_uploads"   # gitignored runtime data
JOBS = PROJECT_ROOT / "service_data" / "command_center_jobs"
LEVEL_COLORS = {"NORMAL": "#2f9e55", "ELEVATED": "#e0a800", "SUSPICIOUS_ACTIVITY": "#e8590c", "WEAPON_RELATED_EVENT": "#d6336c"}
LEVELS = list(LEVEL_COLORS)


# ───────────────────────────── models ─────────────────────────────
def detector_registry() -> Dict[str, Path]:
    """Weapon detectors available locally: the original S.W.O.R.D. weights and every CCTV fine-tune run."""
    reg = {}
    for name, f in (("YOLO26s (original)", "yolo26s_best.pt"), ("YOLO26m (original)", "yolo26m_best.pt"),
                    ("RT-DETR-L (original)", "rtdetr_l_best.pt")):
        if (WEIGHTS / f).exists():
            reg[name] = WEIGHTS / f
    for best in sorted(RUNS.glob("*_detector-*/train/weights/best.pt")):
        run = best.parents[2].name
        if "SMOKE" in run or "ABORTED" in run:
            continue
        label = run.split("_detector-", 1)[1]
        reg[f"{label} ({run[:8]})"] = best
    return reg


def best_detector() -> Optional[Path]:
    """The CCTV fine-tuned detector (latest non-smoke run), else the original YOLO26s."""
    reg = detector_registry()
    tuned = [p for k, p in reg.items() if "cctv-ft" in k]
    return tuned[-1] if tuned else reg.get("YOLO26s (original)")


def weapon_head_status() -> dict:
    from action_recognition.weapon_head import DEFAULT_HEAD

    if not DEFAULT_HEAD.exists():
        return {"installed": False}
    d = json.loads(DEFAULT_HEAD.read_text())
    return {"installed": True, "path": str(DEFAULT_HEAD.relative_to(PROJECT_ROOT)), "created": d.get("created"),
            "C": d.get("C"), "calibration": d.get("calibration", {})}


# ───────────────────────────── system ─────────────────────────────
def system_status() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        out["torch"] = torch.__version__
        out["mps"] = bool(torch.backends.mps.is_available())
        out["cuda"] = bool(torch.cuda.is_available())
    except Exception as e:   # noqa: BLE001
        out["torch_error"] = str(e)
    try:
        import onnxruntime

        out["coreml"] = "CoreMLExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:   # noqa: BLE001
        out["coreml"] = False
    du = shutil.disk_usage(PROJECT_ROOT)
    out["disk_free_gb"] = round(du.free / 1e9, 1)
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"(\d+)%;\s*([a-z ]+);", batt)
        out["battery"] = {"percent": int(m.group(1)), "state": m.group(2).strip(), "ac": "AC Power" in batt} if m else None
    except Exception:   # noqa: BLE001
        out["battery"] = None
    out["ffmpeg"] = bool(shutil.which("ffmpeg"))
    return out


def active_jobs() -> List[str]:
    """Long-running research / training processes on this machine (for the monitor page)."""
    try:
        ps = subprocess.run(["ps", "-axo", "pid,etime,command"], capture_output=True, text=True, timeout=5).stdout
    except Exception:   # noqa: BLE001
        return []
    keys = ("finetune_cctv", "train_gpu", "run_full_pipeline", "run_video_set", "detector_replay", "train_eval_transformer",
            "person_zoom_eval", "eval_phase1_external", "pipeline.run", "uvicorn")
    return [l.strip() for l in ps.splitlines() if any(k in l for k in keys) and "grep" not in l]


# ───────────────────────────── research runs ─────────────────────────────
def list_runs() -> List[dict]:
    """Every run folder, newest first (folders without a YYYYMMDD- date prefix last)."""
    runs = []
    dirs = [d for d in RUNS.iterdir() if d.is_dir()] if RUNS.exists() else []
    dated = lambda d: bool(re.match(r"\d{8}-", d.name))  # noqa: E731
    for d in sorted(dirs, key=lambda d: (dated(d), d.name), reverse=True):
        kind = d.name.split("_", 1)[1] if "_" in d.name else d.name
        runs.append({"name": d.name, "path": d, "kind": kind,
                     "report": (d / "REPORT.md").exists(), "table": (d / "results_table.md").exists(),
                     "figures": sorted(str(p.relative_to(d)) for p in d.glob("*.png")),
                     "aborted": any(t in d.name.upper() for t in ("ABORTED", "FAILED", "SUPERSEDED", "SMOKE"))})
    return runs


def training_curves() -> Dict[str, Path]:
    """Ultralytics results.csv of every detector run (for live training monitoring)."""
    return {p.parents[1].name: p for p in sorted(RUNS.glob("*_detector-*/train/results.csv"))}


# ───────────────────────────── test sources ─────────────────────────────
def sample_sources() -> Dict[str, Path]:
    """Built-in test sources: the weapon-free street clip, mock-attack sequences, Mendeley and UCF-Crime clips."""
    src = {}
    if (DATA / "videos" / "vtest.avi").exists():
        src["vtest.avi — street, weapon-free (10 fps)"] = DATA / "videos" / "vtest.avi"
    for f in sorted((DATA / "mock_attack" / "sequences").glob("Cam*.json")):
        src[f"mock-attack {f.stem[:40]} (2 fps, ground truth)"] = f
    men = DATA / "mendeley_firearm" / "extracted" / "Gun_Action_Recognition_Dataset"
    for cls in ("Handgun", "Machine_Gun", "No_Gun"):
        for v in sorted((men / cls).glob("*/video.mp4"))[:6]:
            src[f"Mendeley {cls}/{v.parent.name}"] = v
    for v in sorted((DATA / "ucf_crime" / "extracted" / "UCF_Crimes" / "Videos").glob("*/*.mp4"))[:40]:
        src[f"UCF-Crime {v.parent.name}/{v.name}"] = v
    return src


# ───────────────────────────── service client ─────────────────────────────
class ServiceClient:
    def __init__(self, base: str = "http://127.0.0.1:8000"):
        self.base = base.rstrip("/")

    def _get(self, path, **params):
        import requests

        r = requests.get(self.base + path, params=params, timeout=5)
        r.raise_for_status()
        return r

    def health(self) -> Optional[dict]:
        try:
            return self._get("/api/health").json()
        except Exception:   # noqa: BLE001
            return None

    def runs(self) -> list:
        return self._get("/api/runs").json()

    def run(self, run_id: str) -> dict:
        return self._get(f"/api/runs/{run_id}").json()

    def incidents(self, **params) -> list:
        return self._get("/api/incidents", **params).json()

    def incident(self, iid: int) -> dict:
        return self._get(f"/api/incidents/{iid}").json()

    def audit(self, limit: int = 200) -> list:
        return self._get("/api/audit", limit=limit).json()

    def preview(self, run_id: str) -> Optional[bytes]:
        try:
            return self._get(f"/api/runs/{run_id}/preview.jpg").content
        except Exception:   # noqa: BLE001
            return None

    def media(self, path: str, **params) -> bytes:
        return self._get(path, **params).content

    def start_run(self, body: dict) -> dict:
        import requests

        r = requests.post(self.base + "/api/runs", json=body, timeout=10)
        if r.status_code >= 400:
            raise RuntimeError(r.json().get("detail", r.text))
        return r.json()

    def stop_run(self, run_id: str, actor: str) -> dict:
        import requests

        return requests.post(f"{self.base}/api/runs/{run_id}/stop", json={"actor": actor}, timeout=10).json()

    def ack(self, iid: int, actor: str, note: str) -> dict:
        import requests

        return requests.post(f"{self.base}/api/incidents/{iid}/ack", json={"actor": actor, "note": note}, timeout=10).json()


def start_service_background(port: int = 8000) -> subprocess.Popen:
    """Launch `python -m service` (127.0.0.1) as a background process; its log goes to service_data/."""
    (PROJECT_ROOT / "service_data").mkdir(exist_ok=True)
    log = open(PROJECT_ROOT / "service_data" / "service_from_command_center.log", "a")
    return subprocess.Popen([sys.executable, "-m", "service", "--port", str(port)], cwd=PROJECT_ROOT, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
