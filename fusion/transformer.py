"""
Phase 5 — Temporal Threat Transformer (model, data, training). NOT used for inference until trained and shown to
beat the rules engine (fusion/rules.py) on held-out videos.

Two input granularities, both sampled at a fixed rate (default 5 Hz; a window of 50 steps = 10 s):
    scene   per time step, each evidence channel's MAXIMUM over the people in view, plus the number of people, the
            unattended-weapon score and the frame's highest raw weapon-detection score (`export_scene_sequences`).
            Matches scene-level interval labels (fusion/labels.py), which is what public datasets provide.
    person  one sequence per tracked person with the channels of fusion/features.py (`export_sequences`); needs
            person-scope labels with anchors.
Output: NORMAL / ELEVATED / SUSPICIOUS_ACTIVITY / WEAPON_RELATED_EVENT for the last step of the window
(observable events, not identity).

    python -m fusion.transformer --runs <run_dir>... --labels labels.csv --out weights/fusion_transformer_scene.pt
(held-out evaluation against the rules baseline: research/tools/fusion/train_eval_transformer.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from fusion.features import CHANNELS, FeatureConfig, person_features, scene_features
from fusion.rules import LEVELS

EXTRA = ["speed_heights_s", "weapon_unattended"]
FEATURES = CHANNELS + EXTRA                                       # person-level
SCENE_FEATURES = [f"max_{c}" for c in CHANNELS] + ["log1p_persons", "weapon_unattended", "max_raw_detection"]


class TemporalThreatTransformer(nn.Module):
    def __init__(self, n_features: int = len(FEATURES), d_model: int = 64, n_heads: int = 4, n_layers: int = 3,
                 max_len: int = 256, n_classes: int = len(LEVELS), dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: [B, T, F]; pad_mask: [B, T] True where padded. Returns logits [B, n_classes]."""
        b, t, _ = x.shape
        h = torch.cat([self.cls.expand(b, -1, -1), self.inp(x)], dim=1) + self.pos[:, : t + 1]
        if pad_mask is not None:
            pad_mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), pad_mask], dim=1)
        return self.head(self.enc(h, src_key_padding_mask=pad_mask)[:, 0])


def _resample(ts: np.ndarray, xs: np.ndarray, rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Hold-last-value resampling onto a regular grid."""
    grid = np.arange(ts[0], ts[-1] + 1e-9, 1.0 / rate_hz)
    idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
    return grid, xs[idx]


def _load(frames_json) -> dict:
    return frames_json if isinstance(frames_json, dict) else json.loads(Path(frames_json).read_text())


# ───────────────────────────── person-level ─────────────────────────────
def export_sequences(frames_json, rate_hz: float = 5.0, cfg: Optional[FeatureConfig] = None) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Recorded run -> {track_id: (times[T], features[T, F])} resampled to rate_hz (last value held)."""
    cfg = cfg or FeatureConfig()
    doc = _load(frames_json)
    shape = (doc["height"], doc["width"])
    latest: Dict[int, dict] = {}
    raw: Dict[int, List[Tuple[float, List[float]]]] = {}
    for f in doc["frames"]:
        latest.update({int(k): v for k, v in (f.get("actions") or {}).items()})
        t = f["timestamp"]
        unatt = scene_features(f["weapons"], shape, cfg)["weapon_unattended"]  # same definition as the rules engine
        for p in f["persons"]:
            pf = person_features(p, f["weapons"], shape, t, latest.get(p["track_id"]), cfg)
            sp = (p.get("trajectory") or {}).get("speed_heights_s") or 0.0
            raw.setdefault(p["track_id"], []).append((t, [pf[c] for c in CHANNELS] + [sp, unatt]))
    return {tid: _resample(np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=np.float32), rate_hz)
            for tid, rows in raw.items()}


# ───────────────────────────── scene-level ─────────────────────────────
def scene_vector(frame: dict, shape, latest: Dict[int, dict], cfg: FeatureConfig) -> List[float]:
    t = frame["timestamp"]
    per = [person_features(p, frame["weapons"], shape, t, latest.get(p["track_id"]), cfg) for p in frame["persons"]]
    mx = [max((f[c] for f in per), default=0.0) for c in CHANNELS]
    raw = max([d["score"] for d in frame.get("weapon_detections", frame["weapons"])] + [0.0])
    return mx + [float(np.log1p(len(per))), scene_features(frame["weapons"], shape, cfg)["weapon_unattended"], raw]


def export_scene_sequences(frames_json, rate_hz: float = 5.0, cfg: Optional[FeatureConfig] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Recorded run -> (times[T], scene features[T, len(SCENE_FEATURES)]) on a regular grid."""
    cfg = cfg or FeatureConfig()
    doc = _load(frames_json)
    shape = (doc["height"], doc["width"])
    latest: Dict[int, dict] = {}
    ts, xs = [], []
    for f in doc["frames"]:
        latest.update({int(k): v for k, v in (f.get("actions") or {}).items()})
        ts.append(f["timestamp"])
        xs.append(scene_vector(f, shape, latest, cfg))
    return _resample(np.array(ts), np.array(xs, dtype=np.float32), rate_hz)


def windows(x: np.ndarray, end: int, win: int) -> Tuple[np.ndarray, np.ndarray]:
    """Window ending at step `end` (inclusive), left-padded to `win` steps; returns (x[win, F], pad_mask[win])."""
    start = max(0, end - win + 1)
    seg = x[start:end + 1]
    pad = win - len(seg)
    mask = np.zeros(win, dtype=bool)
    if pad:
        seg = np.concatenate([np.zeros((pad, x.shape[1]), dtype=x.dtype), seg])
        mask[:pad] = True
    return seg, mask


def scene_windows(x: np.ndarray, levels: np.ndarray, win: int = 50, stride: int = 5) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Training windows (x, pad_mask, level of the last step); steps labelled -1 (unknown) are skipped."""
    return [(*windows(x, end, win), int(levels[end])) for end in range(0, len(x), stride) if levels[end] >= 0]


# ───────────────────────────── training / inference ─────────────────────────────
class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[tuple]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if len(it) == 2:   # (x, y) without padding
            x, y = it
            return torch.from_numpy(x), torch.zeros(len(x), dtype=torch.bool), torch.tensor(y)
        x, m, y = it
        return torch.from_numpy(x), torch.from_numpy(m), torch.tensor(y)


def windows_with_labels(seqs: Dict[int, Tuple[np.ndarray, np.ndarray]], labels: List[dict], video: str,
                        win: int = 50, stride: int = 5) -> List[Tuple[np.ndarray, int]]:
    """Person-level windows from {video, track_id, start_s, end_s, label} dicts (legacy person-level path)."""
    items = []
    for tid, (ts, xs) in seqs.items():
        for end in range(win, len(ts) + 1, stride):
            t_end = ts[end - 1]
            y = 0
            for lb in labels:
                if lb["video"] == video and (not lb.get("track_id") or int(lb["track_id"]) == tid) and lb["start_s"] <= t_end <= lb["end_s"]:
                    y = max(y, LEVELS.index(lb["label"]))
            items.append((xs[end - win:end], y))
    return items


def train(items: List[tuple], epochs: int = 30, lr: float = 1e-3, seed: int = 0, device: str = "cpu",
          batch_size: int = 64, weight_decay: float = 0.01, class_weights: bool = True) -> Tuple[TemporalThreatTransformer, List[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_features = items[0][0].shape[1]
    model = TemporalThreatTransformer(n_features=n_features).to(device)
    ys = [it[-1] for it in items]
    counts = np.bincount(ys, minlength=len(LEVELS)).astype(float)
    w = counts.sum() / np.maximum(counts, 1) / len(LEVELS)
    w[counts == 0] = 0.0
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device) if class_weights else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(WindowDataset(items), batch_size=batch_size, shuffle=True, generator=gen)
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for x, m, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device), m.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        history.append(total / len(items))
    return model, history


@torch.no_grad()
def predict_steps(model: TemporalThreatTransformer, x: np.ndarray, win: int = 50, device: str = "cpu",
                  batch: int = 256) -> np.ndarray:
    """Class probabilities for every step of a sequence (each from the window ending there). [T, n_classes]"""
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        ws = [windows(x, end, win) for end in range(s, min(len(x), s + batch))]
        xb = torch.from_numpy(np.stack([w[0] for w in ws])).to(device)
        mb = torch.from_numpy(np.stack([w[1] for w in ws])).to(device)
        out.append(torch.softmax(model(xb, mb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LEVELS)))


def save_model(model: TemporalThreatTransformer, path: Path, features: Sequence[str], meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "features": list(features), "levels": LEVELS, "meta": meta}, path)


def load_model(path: Path, device: str = "cpu") -> Tuple[TemporalThreatTransformer, dict]:
    ck = torch.load(path, map_location=device, weights_only=True)  # tensors + plain metadata only
    m = TemporalThreatTransformer(n_features=len(ck["features"])).to(device)
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def main():
    from fusion.labels import by_video, load_labels, scene_levels

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders with frames.json (or segments/*/frames.json)")
    ap.add_argument("--labels", required=True, help="interval labels CSV (fusion/labels.py)")
    ap.add_argument("--out", default="weights/fusion_transformer_scene.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--rate", type=float, default=5.0)
    ap.add_argument("--win", type=int, default=50)
    args = ap.parse_args()
    labels = by_video(load_labels(Path(args.labels)))
    items = []
    for run in args.runs:
        for fj in [Path(run) / "frames.json"] + sorted(Path(run).glob("segments/*/frames.json")):
            if not fj.exists():
                continue
            doc = json.loads(fj.read_text())
            if doc["video"] not in labels:
                continue
            ts, x = export_scene_sequences(doc, args.rate)
            items += scene_windows(x, scene_levels(labels[doc["video"]], ts), args.win)
    model, hist = train(items, epochs=args.epochs)
    save_model(model, Path(args.out), SCENE_FEATURES, {"rate_hz": args.rate, "win": args.win, "windows": len(items)})
    print(f"trained on {len(items)} windows; final loss {hist[-1]:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
