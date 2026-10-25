"""Self-tests for the audit's hashing code, run before every audit and stored with its results.

1. Our pHash/dHash/aHash vs the `imagehash` reference implementation (should be near-identical).
2. D4 matching: hash(T(thumbnail of A)) must match hash(thumbnail of T applied to the real image A).
3. Detection of re-encoded / brightness-shifted copies at identity.
"""

from __future__ import annotations

import io

import imagehash
import numpy as np
from PIL import Image, ImageEnhance

from image_scan import D4_NAMES, ahash64, d4_transforms, dhash64, phash64

_PIL_D4 = {
    "identity": lambda im: im,
    "rot90": lambda im: im.transpose(Image.Transpose.ROTATE_90),
    "rot180": lambda im: im.transpose(Image.Transpose.ROTATE_180),
    "rot270": lambda im: im.transpose(Image.Transpose.ROTATE_270),
    "hflip": lambda im: im.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    "vflip": lambda im: im.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
    "transpose": lambda im: im.transpose(Image.Transpose.TRANSPOSE),
    "antitranspose": lambda im: im.transpose(Image.Transpose.TRANSVERSE),
}


def _thumb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L").resize((64, 64), Image.LANCZOS), dtype=np.float32)


def _ham(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def run_selftest(image_paths: list[str]) -> dict:
    ref = {"phash": [], "dhash": [], "ahash": []}
    d4_dist = {n: [] for n in D4_NAMES}
    d4_best_is_correct = {n: 0 for n in D4_NAMES}
    reencode, bright = [], []
    for p in image_paths:
        im = Image.open(p).convert("RGB")
        t = _thumb(im)
        # 1. Reference comparison at identity.
        ref["phash"].append(_ham(phash64(t), int(str(imagehash.phash(im)), 16)))
        ref["dhash"].append(_ham(dhash64(t), int(str(imagehash.dhash(im)), 16)))
        ref["ahash"].append(_ham(ahash64(t), int(str(imagehash.average_hash(im)), 16)))
        # 2. D4 consistency: physically transform the image, compare with transformed-thumbnail hashes.
        variants = [phash64(np.ascontiguousarray(v)) for v in d4_transforms(t)]
        for name in D4_NAMES:
            real = phash64(_thumb(_PIL_D4[name](im)))
            dists = [_ham(v, real) for v in variants]
            d4_dist[name].append(dists[D4_NAMES.index(name)])
            if int(np.argmin(dists)) == D4_NAMES.index(name):
                d4_best_is_correct[name] += 1
        # 3. JPEG re-encode at q=60 and +25% brightness should stay close at identity.
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=60)
        reencode.append(_ham(phash64(_thumb(Image.open(io.BytesIO(buf.getvalue())))), phash64(t)))
        bright.append(_ham(phash64(_thumb(ImageEnhance.Brightness(im).enhance(1.25))), phash64(t)))
    n = len(image_paths)
    summarize = lambda v: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "max": int(np.max(v))}
    result = {
        "n_images": n,
        "vs_imagehash_reference_hamming": {k: summarize(v) for k, v in ref.items()},
        "d4_true_transform_hamming": {k: summarize(v) for k, v in d4_dist.items()},
        "d4_argmin_recovers_true_transform_rate": {k: v / n for k, v in d4_best_is_correct.items()},
        "jpeg_q60_reencode_hamming": summarize(reencode),
        "brightness_x1.25_hamming": summarize(bright),
    }
    result["passed"] = bool(
        result["vs_imagehash_reference_hamming"]["phash"]["mean"] <= 4
        and all(v["p95"] <= 6 for v in result["d4_true_transform_hamming"].values())
        and min(result["d4_argmin_recovers_true_transform_rate"].values()) >= 0.9
    )
    return result
