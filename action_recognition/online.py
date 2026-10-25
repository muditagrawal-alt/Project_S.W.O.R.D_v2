"""
Streaming Phase 4: per-track skeleton buffers -> PoseC3D every `stride_s` seconds of video time.

The offline clip builder (clips.py) needs the whole video's tracks/poses JSON first; a live system cannot wait for
that. This keeps the last `window_s` seconds of COCO-17 skeletons for every tracked person and, once per stride,
classifies each person with enough history: `clip_len` frames sampled uniformly over the window (as in PoseC3D
training), official test preprocessing, flip averaging, and — for NTU's two-person classes — the nearest
co-occurring person as a second skeleton. Clips are batched across people.
"""

from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from action_recognition.posec3d import (DEFAULT_CHECKPOINT, ResNet3dSlowOnly, default_action_map_path, load_action_map,
                                        load_slowonly_weights, map_predictions_to_threats, posec3d_inputs)


def resample_linear(ts: np.ndarray, k: np.ndarray, s: np.ndarray, targets: np.ndarray):
    """Skeletons at `targets` (seconds) by linear interpolation between the neighbouring recorded frames. A joint is
    interpolated only when visible (score > 0) in both neighbours; otherwise the nearer frame's value is used."""
    j = np.clip(np.searchsorted(ts, targets, side="right") - 1, 0, len(ts) - 1)
    j2 = np.minimum(j + 1, len(ts) - 1)
    w = np.clip((targets - ts[j]) / np.maximum(ts[j2] - ts[j], 1e-9), 0.0, 1.0)
    w = np.where(j2 == j, 0.0, w)[:, None]                                  # [T, 1]
    both = (s[j] > 0) & (s[j2] > 0)                                         # [T, 17]
    near = w < 0.5
    kk = np.where(both[..., None], k[j] * (1 - w[..., None]) + k[j2] * w[..., None],
                  np.where(near[..., None], k[j], k[j2]))
    ss = np.where(both, s[j] * (1 - w) + s[j2] * w, np.where(near, s[j], s[j2]))
    return kk.astype(np.float32), ss.astype(np.float32)


class OnlineActionRecognizer:
    def __init__(self, checkpoint: str = str(DEFAULT_CHECKPOINT), action_map_path: Optional[str] = None,
                 device: str = "cpu", window_s: float = 3.0, stride_s: float = 1.0, min_window_s: float = 1.5,
                 clip_len: int = 48, min_frames: int = 4, min_avg_conf: float = 0.3, pair_nearest: bool = True,
                 flip_test: bool = True, stale_s: float = 3.0, batch: int = 8, half: Optional[bool] = None,
                 temporal: str = "linear"):
        self.device = torch.device(device)
        self.model = load_slowonly_weights(ResNet3dSlowOnly(), checkpoint, device="cpu").to(self.device).eval()
        # fp16 on GPU: ~20% faster on Apple MPS; max probability difference vs fp32 0.0007, identical top-1 (tested)
        self.half = (self.device.type in ("mps", "cuda")) if half is None else half
        if self.half:
            self.model = self.model.half()
        self.action_map = load_action_map(action_map_path or default_action_map_path())
        self.names = self.action_map["all_ntu120_classes"]
        self.window_s, self.stride_s, self.min_window_s = window_s, stride_s, min_window_s
        self.clip_len, self.min_frames, self.min_avg_conf = clip_len, min_frames, min_avg_conf
        self.pair_nearest, self.flip_test, self.stale_s, self.batch = pair_nearest, flip_test, stale_s, batch
        # "nearest": clip_len frames picked uniformly by index (repeats frames when the window has fewer, as at 2 fps)
        # "linear" (default): clip_len skeletons interpolated uniformly in time over the window. Chosen on weapon-free
        # vtest: at 2 fps "nearest" collapses PoseC3D's semantics (walking -> 'step on foot', 'shoot at other person';
        # WEAPON_THREAT mean 0.093 vs 0.016 at 10 fps); "linear" restores them (0.016; top-1 JS divergence to the
        # 10-fps output 0.82 -> 0.26). research/runs/20261025-000436_phase4-framerate-vtest
        if temporal not in ("nearest", "linear"):
            raise ValueError(temporal)
        self.temporal = temporal
        self.buf: Dict[int, deque] = defaultdict(deque)   # tid -> (t, kpts[17,2], scores[17])
        self.last_run: Dict[int, float] = {}
        self.latest: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self.buf.clear()
        self.last_run.clear()
        self.latest.clear()

    # ───────────────────────────── buffering ─────────────────────────────
    def _push(self, t: float, persons: List[dict]) -> None:
        for p in persons:
            sc = np.asarray(p["scores"], dtype=np.float32)
            if sc.max() <= 0:  # no pose for this person in this frame
                continue
            q = self.buf[p["track_id"]]
            q.append((t, np.asarray(p["keypoints"], dtype=np.float32), sc))
            while q and q[0][0] < t - self.window_s - 1e-6:
                q.popleft()
        for tid in [k for k, q in self.buf.items() if not q or t - q[-1][0] > self.stale_s]:
            self.buf.pop(tid, None)
            self.last_run.pop(tid, None)
            self.latest.pop(tid, None)

    def _window(self, tid: int, t: float):
        frames = [e for e in self.buf[tid] if e[0] >= t - self.window_s - 1e-6]
        if len(frames) < self.min_frames or frames[-1][0] - frames[0][0] < self.min_window_s - 1e-6:
            return None
        ts = np.array([e[0] for e in frames])
        k = np.stack([e[1] for e in frames])
        s = np.stack([e[2] for e in frames])
        if float(s.mean()) < self.min_avg_conf:
            return None
        return ts, k, s

    def _partner(self, tid: int, ts: np.ndarray, kps: np.ndarray, scores: np.ndarray):
        """Nearest other track present for >= half of the window, aligned to `ts` by nearest timestamp."""
        centre = np.nanmean(np.where(scores[..., None] > 0, kps, np.nan), axis=(0, 1))
        best = None
        for other, q in self.buf.items():
            if other == tid or not q:
                continue
            ots = np.array([e[0] for e in q])
            inside = (ots >= ts[0] - 1e-6) & (ots <= ts[-1] + 1e-6)
            if inside.sum() < 0.5 * len(ts):
                continue
            ok = np.stack([q[i][1] for i in np.nonzero(inside)[0]])
            osc = np.stack([q[i][2] for i in np.nonzero(inside)[0]])
            oc = np.nanmean(np.where(osc[..., None] > 0, ok, np.nan), axis=(0, 1))
            d = float(np.hypot(*(oc - centre)))
            if np.isfinite(d) and (best is None or d < best[0]):
                best = (d, other, ots[inside], ok, osc)
        if best is None:
            return None
        _, other, ots, ok, osc = best
        nearest = np.abs(ots[None, :] - ts[:, None]).argmin(axis=1)
        gap = np.abs(ots[nearest] - ts)
        pk, ps = ok[nearest].copy(), osc[nearest].copy()
        ps[gap > 0.5] = 0.0  # partner not seen near this instant
        return other, pk, ps

    # ───────────────────────────── inference ─────────────────────────────
    def _due(self, t: float) -> list:
        """Tracks whose stride has elapsed and whose window is usable -> clip jobs (arrays are copies)."""
        due = []
        for tid in list(self.buf):
            if t - self.last_run.get(tid, -1e9) < self.stride_s - 1e-6:
                continue
            w = self._window(tid, t)
            if w is None:
                continue
            ts, kps, scores = w
            if self.temporal == "linear":
                targets = ts[0] + (np.arange(self.clip_len) + 0.5) * (ts[-1] - ts[0]) / self.clip_len
                sample = lambda kk, ss: resample_linear(ts, kk, ss, targets)  # noqa: E731
            else:
                idx = np.minimum(((np.arange(self.clip_len) + 0.5) * len(ts) / self.clip_len).astype(int), len(ts) - 1)
                sample = lambda kk, ss: (kk[idx], ss[idx])  # noqa: E731
            k0, s0 = sample(kps, scores)
            persons_k, persons_s, partner = [k0], [s0], None
            if self.pair_nearest:
                pr = self._partner(tid, ts, kps, scores)
                if pr is not None:
                    partner, pk, ps = pr
                    k1, s1 = sample(pk, ps)
                    persons_k.append(k1)
                    persons_s.append(s1)
            due.append((tid, ts, np.stack(persons_k), np.stack(persons_s), partner))
            self.last_run[tid] = t
        return due

    def _infer(self, due: list, img_shape, t: float) -> list:
        """Pure model step (touches no buffers), so it can run on a worker thread."""
        out = []
        for s in range(0, len(due), self.batch):
            chunk = due[s:s + self.batch]
            views = [posec3d_inputs(k, sc, tuple(img_shape[:2]), flip_test=self.flip_test) for _, _, k, sc, _ in chunk]
            n_views = views[0].shape[0]
            with torch.no_grad():
                x = torch.cat(views).to(self.device)
                logits = self.model(x.half() if self.half else x)
            probs = F.softmax(logits.float(), dim=1).reshape(len(chunk), n_views, -1).mean(1).cpu().numpy()
            for (tid, ts, k, sc, partner), p in zip(chunk, probs):
                r = map_predictions_to_threats(p, self.action_map, kps=k, scores=sc)
                r.update(t=round(float(t), 3), window_start=round(float(ts[0]), 3), window_end=round(float(ts[-1]), 3),
                         n_frames=int(len(ts)), partner_track_id=partner, probabilities_120=np.round(p, 4).tolist())
                out.append((tid, r))
        return out

    def _commit(self, results: list) -> Dict[int, Dict[str, Any]]:
        new = {}
        for tid, r in results:
            if tid in self.buf:  # track may have gone stale while a background job was running
                self.latest[tid] = r
                new[tid] = r
        return new

    def update(self, t: float, persons: List[dict], img_shape) -> Dict[int, Dict[str, Any]]:
        """Feed one frame of tracked people (track_id, keypoints[17], scores[17]); returns NEW results for this step."""
        self._push(t, persons)
        return self._commit(self._infer(self._due(t), img_shape, t))


class AsyncActionRecognizer(OnlineActionRecognizer):
    """Live variant: PoseC3D runs on a background thread, so Phase 4 is decoupled from the frame loop.

    Measured on an Apple M4 (research/runs/20261025-021521_timing-clean): with Phase 4 on the SAME GPU as the
    detectors, async gives no speed-up (the GPU serialises the work; the frame loop waits inside the next
    detector call instead). With action_device="cpu", the frame loop runs at 8.8 fps instead of 5.8 (vtest), but
    actions arrive about 3 s late and less often (about 23% of the synchronous updates).

    Buffering stays on the caller's thread. At most one batch is in flight; its results are returned by the first
    update() after it finishes, so they arrive late by the inference latency (reported as `latency_s`). Results are
    otherwise identical to OnlineActionRecognizer, but which windows get classified depends on timing, so offline
    research runs should use the synchronous class for reproducibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phase4")
        self._future: Optional[Future] = None

    def update(self, t: float, persons: List[dict], img_shape) -> Dict[int, Dict[str, Any]]:
        self._push(t, persons)
        new = {}
        if self._future is not None and self._future.done():
            results = self._future.result()
            for _, r in results:
                r["latency_s"] = round(float(t) - r["t"], 3)
            new = self._commit(results)
            self._future = None
        if self._future is None:
            due = self._due(t)
            if due:
                self._future = self._pool.submit(self._infer, due, tuple(img_shape[:2]), t)
        return new

    def reset(self) -> None:
        if self._future is not None:
            self._future.result()
            self._future = None
        super().reset()

    def close(self) -> None:
        self.reset()
        self._pool.shutdown(wait=True)
