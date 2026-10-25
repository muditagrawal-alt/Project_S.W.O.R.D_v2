"""Stages 4–9 — exact duplicates, transform-aware near-duplicates, embeddings, lineage groups, leakage.

Evidence per candidate pair (a, b) of unique image contents:
  * ph_id      pHash Hamming distance at identity (0..64)
  * ph_min, T  smallest pHash distance over the 8 flips/rotations T, i.e. b ≈ T(a)
  * dh_T, ah_T dHash / aHash distance under that same T (independent confirmation)
  * cos        cosine similarity of DINOv2 ViT-S/14 embeddings (catches crops, small rotations, colour jitter)
  * pixel_eq   identical decoded pixels (bytes differ only in container / metadata)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from image_scan import D4_INVERSE, D4_NAMES

CANDIDATE_PH_MAX = 12      # pHash distance at which a pair is examined at all
CANDIDATE_COS_MIN = 0.70   # embedding cosine at which a pair is examined at all
KNN_K = 30

# Confidence rules — calibrated by visual inspection of contact sheets (see report, "Thresholds").
RULES = {
    "geo_ph_max": 8,        # transformed copy: pHash under T at most this ...
    "geo_dh_max": 16,       # ... confirmed by dHash under the same T ...
    "geo_margin": 8,        # ... and T must beat identity by at least this many bits
    "ident_ph_max": 4,      # near-identical at identity (resize / recompress)
    "ident_dh_max": 10,
    "photo_ph_max": 10,     # photometric / compression variant: hash close AND embedding close
    "photo_cos_min": 0.90,
    "emb_high_cos": 0.95,   # embedding alone, high
    "mixed_ph_max": 12,     # hash moderately close AND embedding fairly close
    "mixed_cos_min": 0.85,
    "emb_med_cos": 0.92,    # embedding alone, medium
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ───────────────────────────────── exact duplicates ─────────────────────────────────
def build_contents(files: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Collapse decodable image files with identical bytes into 'contents'. Returns (contents, content_id per file)."""
    dec = files[files.status.isin(["ok", "partial"])].copy()
    dec["content_id"] = dec.groupby("sha256", sort=True).ngroup()
    first = dec.sort_values(["root", "relpath"]).groupby("content_id").first()
    contents = first[["sha256", "pixel_sha256", "width", "height", "n_bytes", "abspath"]].copy()
    hash_cols = [c for c in dec.columns if c[:2] in ("ph", "dh", "ah") and c[2:].isdigit()]
    contents = contents.join(first[hash_cols])
    contents["n_files"] = dec.groupby("content_id").size()
    return contents.reset_index(), dec["content_id"]


def hash_matrix(contents: pd.DataFrame, kind: str) -> np.ndarray:
    return np.stack([contents[f"{kind}{t}"].astype("uint64").to_numpy() for t in range(8)])  # (8, N)


# ───────────────────────────── hash candidate search ─────────────────────────────
def hamming_candidates(P: np.ndarray, max_d: int, chunk: int = 1024) -> pd.DataFrame:
    """All (a, b, T) with popcount(P[T, a] ^ P[0, b]) <= max_d, canonicalised to a < b, min distance kept."""
    n = P.shape[1]
    base = P[0]
    parts = []
    for t in range(8):
        for s in range(0, n, chunk):
            q = P[t, s:s + chunk]
            dist = np.bitwise_count(q[:, None] ^ base[None, :])
            ii, jj = np.nonzero(dist <= max_d)
            d = dist[ii, jj]
            ii = ii + s
            keep = ii != jj
            if t == 0:
                keep &= ii < jj
            parts.append(np.stack([ii[keep], jj[keep], np.full(keep.sum(), t), d[keep]], axis=1))
    if not parts or sum(len(p) for p in parts) == 0:
        return pd.DataFrame(columns=["a", "b"])
    arr = np.concatenate(parts).astype(np.int64)
    i, j, t, d = arr.T
    inv = np.array([D4_NAMES.index(D4_INVERSE[name]) for name in D4_NAMES])
    a, b = np.minimum(i, j), np.maximum(i, j)
    T = np.where(i < j, t, inv[t])
    df = pd.DataFrame({"a": a, "b": b, "T_hash": T, "d": d}).sort_values("d").drop_duplicates(["a", "b"])
    return df[["a", "b"]].reset_index(drop=True)


# ─────────────────────────────────── embeddings ───────────────────────────────────
class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[str], size: int = 224):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - IMAGENET_MEAN) / IMAGENET_STD, i


def compute_embeddings(paths: list[str], device: str, batch: int = 64, workers: int = 6, logger=None) -> np.ndarray:
    import timm

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    model = model.eval().to(device)
    loader = torch.utils.data.DataLoader(ImageListDataset(paths), batch_size=batch, num_workers=workers,
                                         shuffle=False, persistent_workers=False)
    out = np.zeros((len(paths), model.num_features), dtype=np.float32)
    done = 0
    with torch.no_grad():
        for x, idx in loader:
            f = F.normalize(model(x.to(device)).float(), dim=1)
            out[idx.numpy()] = f.cpu().numpy()
            done += len(idx)
            if logger and done % (batch * 50) < batch:
                logger.info(f"  embeddings {done}/{len(paths)}")
    return out


def knn_candidates(E: np.ndarray, k: int, min_cos: float, device: str, chunk: int = 2048) -> pd.DataFrame:
    Et = torch.from_numpy(E).to(device)
    n = Et.shape[0]
    rows = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            S = Et[s:s + chunk] @ Et.T
            r = torch.arange(S.shape[0], device=device)
            S[r, r + s] = -1.0
            vals, idx = S.topk(min(k, n - 1), dim=1)
            vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
            qi = np.repeat(np.arange(s, s + S.shape[0]), vals.shape[1])
            m = vals.ravel() >= min_cos
            rows.append(np.stack([qi[m], idx.ravel()[m]], axis=1))
    arr = np.concatenate(rows) if rows else np.zeros((0, 2), dtype=np.int64)
    df = pd.DataFrame({"a": np.minimum(arr[:, 0], arr[:, 1]), "b": np.maximum(arr[:, 0], arr[:, 1])})
    return df.drop_duplicates().reset_index(drop=True)


def random_pair_cosines(E: np.ndarray, n: int = 200_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(E), n)
    b = rng.integers(0, len(E), n)
    keep = a != b
    return np.einsum("ij,ij->i", E[a[keep]], E[b[keep]])


# ─────────────────────────────────── evidence ───────────────────────────────────
def pair_evidence(pairs: pd.DataFrame, contents: pd.DataFrame, P: np.ndarray, D: np.ndarray, A: np.ndarray,
                  E: np.ndarray) -> pd.DataFrame:
    a = pairs.a.to_numpy()
    b = pairs.b.to_numpy()
    inv = np.array([D4_NAMES.index(D4_INVERSE[n]) for n in D4_NAMES])
    d_ab = np.bitwise_count(P[:, a] ^ P[0, b][None, :]).astype(np.int16)  # b ≈ T(a)
    d_ba = np.bitwise_count(P[:, b] ^ P[0, a][None, :]).astype(np.int16)  # a ≈ T(b) -> b ≈ T^-1(a)
    both = np.concatenate([d_ab, d_ba[inv]], axis=0)                       # 16 rows, both expressed as b ≈ T(a)
    best = both.argmin(axis=0)
    T = best % 8
    from_ba = best >= 8
    ph_min = both[best, np.arange(len(a))]

    def dist_under(H: np.ndarray) -> np.ndarray:
        fwd = np.bitwise_count(H[T, a] ^ H[0, b])
        bwd = np.bitwise_count(H[inv[T], b] ^ H[0, a])
        return np.where(from_ba, bwd, fwd).astype(np.int16)

    ev = pd.DataFrame({
        "a": a, "b": b,
        "ph_id": d_ab[0],
        "ph_min": ph_min,
        "T": np.array(D4_NAMES)[T],
        "dh_id": np.bitwise_count(D[0, a] ^ D[0, b]).astype(np.int16),
        "dh_T": dist_under(D),
        "ah_T": dist_under(A),
        "cos": np.einsum("ij,ij->i", E[a], E[b]).round(4),
        "pixel_eq": contents.pixel_sha256.to_numpy()[a] == contents.pixel_sha256.to_numpy()[b],
    })
    wa, ha = contents.width.to_numpy()[a], contents.height.to_numpy()[a]
    wb, hb = contents.width.to_numpy()[b], contents.height.to_numpy()[b]
    ev["same_dims"] = (wa == wb) & (ha == hb)
    ev["area_ratio"] = np.round(np.minimum(wa * ha, wb * hb) / np.maximum(wa * ha, wb * hb), 3)
    return ev


def classify_edges(ev: pd.DataFrame, rules: dict = RULES) -> pd.DataFrame:
    conf = np.full(len(ev), "LOW", dtype=object)
    rel = np.full(len(ev), "weak_similarity", dtype=object)

    def assign(mask, c, r):
        m = mask & (conf == "LOW") & (rel == "weak_similarity")
        conf[m] = c
        rel[m] = r

    assign(ev.pixel_eq.to_numpy(), "HIGH", "pixel_identical")
    geo = ((ev["T"] != "identity") & (ev.ph_min <= rules["geo_ph_max"]) & (ev.dh_T <= rules["geo_dh_max"])
           & ((ev.ph_id - ev.ph_min) >= rules["geo_margin"])).to_numpy()
    geo_rel = np.array(["geometric_" + t for t in ev["T"]], dtype=object)
    m = geo & (conf == "LOW") & (rel == "weak_similarity")
    conf[m], rel[m] = "HIGH", geo_rel[m]
    assign(((ev.ph_id <= rules["ident_ph_max"]) & (ev.dh_id <= rules["ident_dh_max"])).to_numpy(),
           "HIGH", "near_identical_resize_or_recompress")
    assign(((ev.ph_id <= rules["photo_ph_max"]) & (ev.cos >= rules["photo_cos_min"])).to_numpy(),
           "HIGH", "photometric_or_compression_variant")
    assign((ev.cos >= rules["emb_high_cos"]).to_numpy(), "HIGH", "embedding_near_duplicate")
    assign(((ev.ph_min <= rules["mixed_ph_max"]) & (ev.cos >= rules["mixed_cos_min"])).to_numpy(),
           "MEDIUM", "hash_and_embedding_similar")
    assign((ev.cos >= rules["emb_med_cos"]).to_numpy(), "MEDIUM", "embedding_similar_crop_or_view")
    out = ev.copy()
    out["confidence"] = conf
    out["relation"] = rel
    return out


# ───────────────────────────── geometric verification ─────────────────────────────
# Calibrated on visually labelled pairs (different weapons: 0–6 inliers; true copies/frames: mostly 33–346)
# and on known copies made with the dataset's own Albumentations pipeline (results/detector_sensitivity.json).
VERIFY = {"strong_inliers": 30, "weak_inliers": 20, "min_inlier_span": 0.04, "scale_min": 0.25, "scale_max": 4.0, "emb_override_cos": 0.96,
          "strict_geo_cos_min": 0.80, "verify_low_cos_min": 0.70, "verify_low_ph_max": 12}
# Why 30/20 + a scale check: at full-dataset scale, 10–15 inliers let through degenerate RANSAC fits (scale ≈ 0, all points
# collapsed onto one cluster), which chained 20,577 images of 8 classes into one "lineage group". With a plausible
# similarity scale and >= 20 inliers, cross-class links fall from 8.6% to 0.44% while 96.7% of known copies are kept.


def is_strict(ev: pd.DataFrame) -> pd.Series:
    """Only pixel-identical copies skip verification. Hash-only relations (near-identical, flips) looked clean on
    CCTV frames but produce false links between DIFFERENT thin objects on white backgrounds (e.g. a shotgun and a
    sabre with pHash distance <= 4, embedding cos 0.22), so they are verified like everything else."""
    return ev.relation.eq("pixel_identical")


def needs_verification(ev: pd.DataFrame) -> pd.Series:
    low = ev.confidence == "LOW"
    borderline = low & ((ev.cos >= VERIFY["verify_low_cos_min"]) | (ev.ph_min <= VERIFY["verify_low_ph_max"]))
    return (~low & ~is_strict(ev)) | borderline


def finalize_edges(ev: pd.DataFrame, ver: pd.DataFrame) -> pd.DataFrame:
    """Combine rule-based confidence with SIFT/RANSAC verification (only where needs_verification is True)."""
    out = ev.copy()
    out["confidence_base"], out["relation_base"] = out.confidence, out.relation
    for c in ("n_good", "n_inliers", "scale", "angle", "inlier_span", "mirrored"):
        out[c] = ver[c].reindex(out.index) if c in ver else np.nan
    strict = is_strict(out).to_numpy()
    check = needs_verification(out).to_numpy() & out.n_inliers.notna().to_numpy()
    plausible = ((out.inlier_span.fillna(0) >= VERIFY["min_inlier_span"])
                 & out.scale.between(VERIFY["scale_min"], VERIFY["scale_max"])).to_numpy()
    inl = np.where(plausible, out.n_inliers.fillna(-1).to_numpy(), -1)
    geo = (out.mirrored.fillna(False).astype(bool) | (out.angle.abs() > 2.0) | ((out.scale - 1.0).abs() > 0.04)).to_numpy()
    conf = np.where(strict, "HIGH", "LOW").astype(object)
    rel = np.where(strict, out.relation, out.relation).astype(object)
    ok_hi = check & ~strict & (inl >= VERIFY["strong_inliers"])
    ok_med = check & ~strict & (inl >= VERIFY["weak_inliers"]) & (inl < VERIFY["strong_inliers"])
    rejected = check & ~strict & (inl < VERIFY["weak_inliers"])
    conf[ok_hi], conf[ok_med] = "HIGH", "MEDIUM"
    verified = ok_hi | ok_med
    rel[verified & geo] = "verified_geometric_augmentation"
    rel[verified & ~geo] = "verified_same_view_photometric_or_same_camera"
    rel[rejected] = "rejected_by_verification"
    # Very high embedding similarity stands on its own: zoom/crop variants of low-texture product shots
    # often lack enough SIFT inliers, while every visually checked pair with cos >= 0.95 was a true copy.
    emb = (conf == "LOW") & (out.cos.to_numpy() >= VERIFY["emb_override_cos"])
    conf[emb], rel[emb] = "HIGH", "embedding_near_duplicate_unverified"
    out["confidence"], out["relation"] = conf, rel
    return out


# ─────────────────────────────────── grouping ───────────────────────────────────
class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def lineage_groups(n_contents: int, edges: pd.DataFrame) -> np.ndarray:
    uf = UnionFind(n_contents)
    for a, b in edges[["a", "b"]].itertuples(index=False):
        uf.union(int(a), int(b))
    roots = np.array([uf.find(i) for i in range(n_contents)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def expected_spanning_rate(k: int, p_train: float) -> float:
    """Probability that a group of k items, each assigned independently at random, lands in both splits."""
    return 1.0 - p_train ** k - (1.0 - p_train) ** k
