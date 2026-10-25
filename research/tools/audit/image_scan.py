"""Stage 2 — per-image integrity, metadata, augmentation fingerprints and D4 perceptual hashes.

Every image is fully decoded twice (Pillow strict mode + OpenCV). Perceptual hashes are computed
for all 8 flips/rotations (the dihedral group D4) so that flipped/rotated copies can be matched.
"""

from __future__ import annotations

import hashlib
import io
import zlib

import cv2
import numpy as np
import scipy.fft
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False  # truncated files must fail, not be silently padded
Image.MAX_IMAGE_PIXELS = None            # we flag extreme sizes ourselves instead of raising
cv2.setNumThreads(1)

D4_NAMES = ["identity", "rot90", "rot180", "rot270", "hflip", "vflip", "transpose", "antitranspose"]
D4_INVERSE = {"identity": "identity", "rot90": "rot270", "rot180": "rot180", "rot270": "rot90",
              "hflip": "hflip", "vflip": "vflip", "transpose": "transpose", "antitranspose": "antitranspose"}

# Standard IJG luminance quantisation table (quality 50), used to estimate JPEG quality.
_IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def d4_transforms(t: np.ndarray) -> list[np.ndarray]:
    """Order matches D4_NAMES. np.rot90 rotates counter-clockwise."""
    return [t, np.rot90(t, 1), np.rot90(t, 2), np.rot90(t, 3), np.fliplr(t), np.flipud(t), t.T, np.rot90(t, 2).T]


def _bits_to_int(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8)).tobytes(), "big")


def phash64(t64: np.ndarray) -> int:
    """imagehash-style pHash: 32x32 grey, 2-D DCT-II, top-left 8x8 compared with its median."""
    s = t64.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    d = scipy.fft.dct(scipy.fft.dct(s, axis=0), axis=1)
    low = d[:8, :8]
    return _bits_to_int((low > np.median(low)).ravel())


def dhash64(t64: np.ndarray) -> int:
    s = cv2.resize(np.ascontiguousarray(t64), (9, 8), interpolation=cv2.INTER_AREA)
    return _bits_to_int((s[:, 1:] > s[:, :-1]).ravel())


def ahash64(t64: np.ndarray) -> int:
    s = t64.reshape(8, 8, 8, 8).mean(axis=(1, 3))
    return _bits_to_int((s > s.mean()).ravel())


def detect_magic(head: bytes) -> str:
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head[:4] in (b"GIF8",):
        return "gif"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    return "unknown"


def estimate_jpeg_quality(quant: dict | None) -> tuple[int | None, str | None]:
    """Order-independent estimate from the luminance table's sum versus the IJG table; plus a table signature."""
    if not quant:
        return None, None
    tables = {k: list(v) for k, v in sorted(quant.items())}
    sig = hashlib.sha1(repr(tables).encode()).hexdigest()[:10]
    luma = np.array(tables.get(0, next(iter(tables.values()))), dtype=np.float64)
    scale = 100.0 * luma.sum() / _IJG_LUMA.sum()
    quality = (200.0 - scale) / 2.0 if scale <= 100 else 5000.0 / scale
    return int(round(min(100, max(1, quality)))), sig


def _corner_stats(gray: np.ndarray) -> tuple[int, int]:
    """Count corners that are dark & flat (typical rotation fill) and corners that are flat in any colour."""
    h, w = gray.shape
    ph, pw = max(3, int(0.06 * h)), max(3, int(0.06 * w))
    dark = const = 0
    for patch in (gray[:ph, :pw], gray[:ph, -pw:], gray[-ph:, :pw], gray[-ph:, -pw:]):
        m, s = float(patch.mean()), float(patch.std())
        if s < 3.0:
            const += 1
            if m < 12.0:
                dark += 1
    return dark, const


def _bar_fraction(gray: np.ndarray) -> float:
    """Largest fraction of the image covered by uniform letterbox bars (top+bottom or left+right)."""
    def run(lines: np.ndarray) -> int:
        ref = lines[0].mean()
        n = 0
        for line in lines[: len(lines) // 4]:
            if line.std() < 2.0 and abs(line.mean() - ref) < 3.0:
                n += 1
            else:
                break
        return n

    h, w = gray.shape
    tb = (run(gray) + run(gray[::-1])) / h
    lr = (run(gray.T) + run(gray.T[::-1])) / w
    return float(max(tb, lr))


def scan_image(path: str) -> dict:
    out: dict = {"abspath": path}
    try:
        with open(path, "rb") as fp:
            data = fp.read()
    except OSError as e:
        out["status"] = "unreadable"
        out["error"] = repr(e)[:200]
        return out
    out["n_bytes"] = len(data)
    out["sha256"] = hashlib.sha256(data).hexdigest()
    out["crc32"] = zlib.crc32(data) & 0xFFFFFFFF
    if not data:
        out["status"] = "zero_byte"
        return out
    out["magic"] = detect_magic(data[:16])
    if out["magic"] == "jpeg":
        eoi = data.rfind(b"\xff\xd9")
        out["jpeg_has_eoi"] = eoi != -1
        out["jpeg_trailing_bytes"] = (len(data) - eoi - 2) if eoi != -1 else None

    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        out["pil_verify"] = True
    except Exception as e:
        out["pil_verify"] = False
        out["pil_verify_error"] = repr(e)[:200]

    try:
        im = Image.open(io.BytesIO(data))
        out["format"] = im.format
        out["mode"] = im.mode
        out["width"], out["height"] = im.size
        exif = im.getexif()
        out["n_exif_tags"] = len(exif)
        out["exif_orientation"] = exif.get(0x0112)
        out["exif_software"] = str(exif.get(0x0131))[:60] if exif.get(0x0131) else None
        out["exif_camera"] = " ".join(str(exif.get(t)) for t in (0x010F, 0x0110) if exif.get(t))[:60] or None
        out["progressive"] = bool(im.info.get("progressive") or im.info.get("progression"))
        out["has_icc"] = "icc_profile" in im.info
        out["jpeg_quality"], out["jpeg_qsig"] = estimate_jpeg_quality(getattr(im, "quantization", None))
        im.load()
        out["pil_decode"] = True
    except Exception as e:
        out["pil_decode"] = False
        out["pil_decode_error"] = repr(e)[:200]

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    out["cv2_decode"] = arr is not None
    if not out.get("pil_decode"):
        out["status"] = "corrupt" if not out["cv2_decode"] else "pil_failed_cv2_ok"
        return out
    out["status"] = "ok" if out["cv2_decode"] and out.get("pil_verify") else "partial"

    rgb = im.convert("RGB")
    rgb_arr = np.asarray(rgb)
    out["pixel_sha256"] = hashlib.sha256(f"{rgb_arr.shape}".encode() + rgb_arr.tobytes()).hexdigest()

    # Fingerprints on a bounded-size copy (max side 512) to keep cost flat across resolutions.
    small = rgb.copy()
    small.thumbnail((512, 512), Image.BILINEAR)
    s_arr = np.asarray(small).astype(np.int16)
    gray = np.asarray(small.convert("L"), dtype=np.float32)
    out["mean_luma"] = float(gray.mean())
    out["std_luma"] = float(gray.std())
    out["is_grayscale"] = bool(np.abs(s_arr[..., 0] - s_arr[..., 1]).mean() < 2.0
                               and np.abs(s_arr[..., 1] - s_arr[..., 2]).mean() < 2.0)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    out["laplacian_var"] = float(lap.var())
    out["noise_sigma"] = float(np.sqrt(np.pi / 2.0) * np.abs(lap).mean() / 6.0)
    out["corners_dark_flat"], out["corners_flat"] = _corner_stats(gray)
    out["letterbox_fraction"] = _bar_fraction(gray)

    # D4 perceptual hashes from one 64x64 Lanczos thumbnail.
    t64 = np.asarray(rgb.convert("L").resize((64, 64), Image.LANCZOS), dtype=np.float32)
    for i, t in enumerate(d4_transforms(t64)):
        t = np.ascontiguousarray(t)
        out[f"ph{i}"] = phash64(t)
        out[f"dh{i}"] = dhash64(t)
        out[f"ah{i}"] = ahash64(t)
    return out
