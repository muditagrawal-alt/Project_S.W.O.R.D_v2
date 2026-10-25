"""Geometric verification of candidate near-duplicate pairs (SIFT + RANSAC similarity transform).

Perceptual hashes and embeddings cannot tell "same photo, augmented" apart from "different weapon shot in the
same studio style" (visually confirmed on the calibration sheets). Local-feature matching with a RANSAC-fitted
similarity transform can: augmented copies (flip, rotate ±20°, scale, shift, colour jitter, blur, noise) and frames
from the same fixed camera keep many geometrically consistent matches; different objects do not.

Flips: SIFT is rotation- but not mirror-invariant, so image b is mirrored whenever the hash transform involves a
reflection (hflip / vflip / transpose / antitranspose — each is a mirror plus a rotation), and the other orientation
is tried when the first attempt fails.

Speed: features (identity and mirrored) are extracted once per image into memory-mapped arrays shared by all
worker processes; a pair then costs only descriptor matching + RANSAC.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)
MAX_SIDE = 480
N_FEATURES = 300
RATIO = 0.75
RANSAC_PX = 6.0
PRE_BLUR_SIGMA = float(os.environ.get("SWORD_VERIFY_BLUR", "0"))
MIRRORING = {"hflip", "vflip", "transpose", "antitranspose"}

_sift = None
_store = None  # (desc, pts, count, shape) memmaps, opened lazily in each worker


def extract(path: str, mirror: bool):
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=N_FEATURES)
    im = Image.open(path).convert("L")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
    g = np.asarray(im)
    if mirror:
        g = np.ascontiguousarray(g[:, ::-1])
    if PRE_BLUR_SIGMA > 0:
        g = cv2.GaussianBlur(g, (0, 0), PRE_BLUR_SIGMA)
    kp, des = _sift.detectAndCompute(g, None)
    if des is None or not kp:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), g.shape
    order = np.argsort([-k.response for k in kp])[:N_FEATURES]
    pts = np.float32([kp[i].pt for i in order])
    return pts, np.clip(des[order], 0, 255).astype(np.uint8), g.shape


# ───────────────────────────── shared feature store ─────────────────────────────
def _open_store(folder: Path, n: int, mode: str):
    desc = np.lib.format.open_memmap(folder / "desc.npy", mode=mode, dtype=np.uint8, shape=(n, 2, N_FEATURES, 128))
    pts = np.lib.format.open_memmap(folder / "pts.npy", mode=mode, dtype=np.float32, shape=(n, 2, N_FEATURES, 2))
    cnt = np.lib.format.open_memmap(folder / "count.npy", mode=mode, dtype=np.int16, shape=(n, 2))
    shp = np.lib.format.open_memmap(folder / "shape.npy", mode=mode, dtype=np.int16, shape=(n, 2))
    return desc, pts, cnt, shp


def _extract_job(args):
    folder, n, i, path = args
    global _store
    if _store is None:
        _store = _open_store(Path(folder), n, "r+")
    desc, pts, cnt, shp = _store
    for m in (0, 1):
        try:
            p, d, s = extract(path, bool(m))
        except Exception:
            p, d, s = np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8), (1, 1)
        k = len(p)
        desc[i, m, :k], pts[i, m, :k], cnt[i, m] = d, p, k
        shp[i] = s[:2]
    return i


def build_store(paths: list[str], folder: Path, workers: int = 9, logger=None) -> None:
    from concurrent.futures import ProcessPoolExecutor

    folder.mkdir(parents=True, exist_ok=True)
    done_flag = folder / "COMPLETE"
    if done_flag.exists():
        return
    n = len(paths)
    store = _open_store(folder, n, "w+")
    del store
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, _ in enumerate(ex.map(_extract_job, [(str(folder), n, i, p) for i, p in enumerate(paths)], chunksize=64), 1):
            if logger and k % 5000 == 0:
                logger.info(f"  SIFT features {k}/{n}")
    done_flag.write_text(str(n))


def _get(i: int, mirror: bool):
    desc, pts, cnt, shp = _store
    k = int(cnt[i, int(mirror)])
    # shapes are stored as int16: convert to Python ints, otherwise h * w (e.g. 480 * 480) overflows int16
    return np.asarray(pts[i, int(mirror), :k]), np.asarray(desc[i, int(mirror), :k], dtype=np.float32), tuple(int(v) for v in shp[i])


# ───────────────────────────── matching ─────────────────────────────
def match(fa, fb) -> dict:
    pa, da, sa = fa
    pb, db, _ = fb
    out = {"n_kp_a": len(pa), "n_kp_b": len(pb), "n_good": 0, "n_inliers": 0, "scale": np.nan, "angle": np.nan,
           "inlier_span": 0.0}
    if len(pa) < 4 or len(pb) < 4:
        return out
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(np.ascontiguousarray(da, np.float32), np.ascontiguousarray(db, np.float32), k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    out["n_good"] = len(good)
    if len(good) < 6:
        return out
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX,
                                         maxIters=2000, confidence=0.995)
    if M is None or inl is None:
        return out
    out["n_inliers"] = int(inl.sum())
    out["scale"] = float(np.hypot(M[0, 0], M[1, 0]))
    out["angle"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    ip = src[inl.ravel().astype(bool)]
    out["inlier_span"] = float(np.ptp(ip[:, 0]) * np.ptp(ip[:, 1]) / (float(sa[0]) * float(sa[1]))) if len(ip) >= 3 else 0.0
    return out


def _orient_and_match(fa, get_b, T: str) -> dict:
    first = T in MIRRORING
    r = match(fa, get_b(first))
    r["mirrored"] = first
    if r["n_inliers"] < 15:
        r2 = match(fa, get_b(not first))
        if r2["n_inliers"] > r["n_inliers"]:
            r, r["mirrored"] = r2, not first
    return r


def _verify_job(args):
    folder, n, idx, ia, ib, T = args
    global _store
    if _store is None:
        _store = _open_store(Path(folder), n, "r")
    r = _orient_and_match(_get(ia, False), lambda mir: _get(ib, mir), T)
    r["idx"] = idx
    return r


def verify_pair(args: tuple) -> dict:
    """Stand-alone check from image paths: args = (pair_index, path_a, path_b, hash_transform)."""
    idx, pa, pb, T = args
    try:
        r = _orient_and_match(extract(pa, False), lambda mir: extract(pb, mir), T)
    except Exception as e:
        r = {"n_inliers": 0, "n_good": 0, "error": repr(e)[:120], "mirrored": False}
    r["idx"] = idx
    return r


def run_verification(pairs, path_of, workers: int = 9, logger=None, store_dir: Path | None = None):
    """pairs: DataFrame with a, b, T (content indices). path_of: content index -> path. Returns rows aligned to pairs.index."""
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor

    if store_dir is None:  # small jobs (calibration): extract on the fly
        jobs = [(i, path_of[a], path_of[b], t) for i, a, b, t in zip(pairs.index, pairs.a, pairs.b, pairs["T"])]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(verify_pair, jobs, chunksize=8))
        return pd.DataFrame(rows).set_index("idx").reindex(pairs.index)
    n = len(path_of)
    build_store([path_of[i] for i in range(n)], store_dir, workers, logger)
    jobs = [(str(store_dir), n, i, int(a), int(b), t) for i, a, b, t in zip(pairs.index, pairs.a, pairs.b, pairs["T"])]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(_verify_job, jobs, chunksize=512), 1):
            rows.append(r)
            if logger and k % 50000 == 0:
                logger.info(f"  verified {k}/{len(jobs)} pairs")
    return pd.DataFrame(rows).set_index("idx").reindex(pairs.index)
