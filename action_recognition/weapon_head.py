"""
Learned weapon-action head on top of PoseC3D: P(the person is handling a weapon) from the 120 NTU class
probabilities of one window.

Why: zero-shot, the hand-made WEAPON_THREAT grouping of NTU classes carries no weapon signal on surveillance video
(AUC ~0.53), but a linear model on the same 120 outputs separates weapon handling from ordinary actions on unseen
locations (AUC ~0.80). Trained by research/tools/phase4/train_weapon_head.py (logistic regression on standardised
log-probabilities, cross-fitted by location); the JSON file holds the coefficients, the standardisation, the
calibrated thresholds and the provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEAD = PROJECT_ROOT / "weights" / "posec3d_weapon_head_v1.json"


class WeaponActionHead:
    def __init__(self, path: Path = DEFAULT_HEAD):
        d = json.loads(Path(path).read_text())
        self.path = Path(path)
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.std = np.asarray(d["std"], dtype=np.float64)
        self.coef = np.asarray(d["coef"], dtype=np.float64)
        self.intercept = float(d["intercept"])
        self.calibration = d.get("calibration", {})
        self.meta = {k: v for k, v in d.items() if k not in ("mean", "std", "coef")}
        if not (len(self.mean) == len(self.std) == len(self.coef) == 120):
            raise ValueError(f"{path}: expected 120-dimensional head")

    @staticmethod
    def features(probabilities_120: Sequence[float]) -> np.ndarray:
        return np.log(np.clip(np.asarray(probabilities_120, dtype=np.float64), 1e-6, 1.0))

    def prob(self, probabilities_120: Sequence[float]) -> float:
        z = (self.features(probabilities_120) - self.mean) / self.std
        return float(1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept))))

    def to_json(self) -> dict:
        return {**self.meta, "mean": self.mean.tolist(), "std": self.std.tolist(), "coef": self.coef.tolist()}


def load_default() -> Optional[WeaponActionHead]:
    """The deployed head if it has been trained (weights/ is not versioned), else None."""
    return WeaponActionHead(DEFAULT_HEAD) if DEFAULT_HEAD.exists() else None


def save_head(path: Path, mean, std, coef, intercept: float, **meta) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({**meta, "intercept": float(intercept), "mean": list(map(float, mean)),
                                      "std": list(map(float, std)), "coef": list(map(float, coef))}, indent=1))
