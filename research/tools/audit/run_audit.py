"""SWORD v2 — dataset research-readiness audit (read-only).

Implements the 10 checks of "Antigravity Prompt — S.W.O.R.D. Dataset Research-Readiness Audit".
The dataset is never modified; all outputs go to research/runs/<run_id>/.

Usage (from the project root):
    venv/bin/python research/tools/audit/run_audit.py                 # full audit
    venv/bin/python research/tools/audit/run_audit.py --limit 300     # quick smoke run
    venv/bin/python research/tools/audit/run_audit.py --run-id <id>   # resume / re-report from cache
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (RAW_ROOT, YOLO_ROOT, Timer, append_index, dataset_fingerprint, make_run_dir,  # noqa: E402
                    setup_logging, write_json, write_manifest)
import inventory as inv_mod  # noqa: E402
import labels as lab_mod  # noqa: E402
import similarity as sim  # noqa: E402
import verify  # noqa: E402
from image_scan import scan_image  # noqa: E402
from selftest import run_selftest  # noqa: E402

HASH_COLS = [f"{k}{t}" for k in ("ph", "dh", "ah") for t in range(8)]


def cached(run_dir: Path, name: str, fn, logger):
    path = run_dir / "cache" / f"{name}.pkl"
    if path.exists():
        logger.info(f"[cache] {name}")
        return pd.read_pickle(path)
    obj = fn()
    pd.to_pickle(obj, path)
    return obj


def limit_inventory(inv: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Debug mode: keep `limit` random images per split / raw folder, their labels, and all other files."""
    key = inv.root + "|" + inv.split + "|" + inv.raw_folder.fillna("") + "|"
    img = inv[inv.kind == "image"]
    kept = set()
    for _, g in img.groupby(key[img.index]):
        kept.update((key[g.index] + g.stem).sample(min(len(g), limit), random_state=seed))
    row_key = key + inv.stem
    return inv[~inv.kind.isin(["image", "label"]) | row_key.isin(kept)].reset_index(drop=True)


def stage_scan(inv: pd.DataFrame, workers: int, logger) -> pd.DataFrame:
    paths = inv[inv.kind == "image"].abspath.tolist()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(scan_image, paths, chunksize=32), 1):
            results.append(r)
            if i % 5000 == 0:
                logger.info(f"  scanned {i}/{len(paths)} images")
    df = pd.DataFrame(results)
    for c in HASH_COLS:
        if c in df:
            df[c] = np.array([int(v) if pd.notna(v) else 0 for v in df[c]], dtype=np.uint64)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default=None, help="Reuse an existing run folder (resume from its cache)")
    ap.add_argument("--limit", type=int, default=None, help="Debug: keep only N images per split/folder")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--skip-model-probe", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    run_dir = make_run_dir("dataset-audit" + ("-smoke" if args.limit else ""), args.run_id)
    logger = setup_logging(run_dir)
    logger.info(f"Run folder: {run_dir}")

    config = {
        "seed": args.seed, "limit": args.limit, "device": args.device,
        "candidate_ph_max": sim.CANDIDATE_PH_MAX, "candidate_cos_min": sim.CANDIDATE_COS_MIN, "knn_k": sim.KNN_K,
        "edge_rules": sim.RULES,
        "verification": {**sim.VERIFY, "sift_max_side": verify.MAX_SIDE, "sift_features": verify.N_FEATURES,
                         "lowe_ratio": verify.RATIO, "ransac_px": verify.RANSAC_PX, "pre_blur_sigma": verify.PRE_BLUR_SIGMA},
        "box_thresholds": {k: getattr(lab_mod, k) for k in ("TOL", "TOUCH_PX", "DEGENERATE_PX", "TINY_AREA_NORM",
                                                             "HUGE_AREA_NORM", "EXTREME_ASPECT", "DUP_IOU")},
        "embedding_model": "timm vit_small_patch14_dinov2.lvd142m @224px, CLS token, L2-normalised",
        "hash_definition": "64x64 Lanczos grey thumbnail; pHash = 32x32 DCT top-left 8x8 > median; "
                           "dHash = 9x8 INTER_AREA horizontal gradient; aHash = 8x8 mean; all under 8 D4 transforms",
    }
    with Timer(logger, "dataset fingerprints"):
        fingerprints = {"yolo_dataset": dataset_fingerprint(YOLO_ROOT), "raw": dataset_fingerprint(RAW_ROOT)}
    write_manifest(run_dir, "dataset_audit", config,
                   extra={"read_only": True, "dataset_fingerprints_before": fingerprints})

    # ── S1 inventory ──
    with Timer(logger, "S1 inventory"):
        inv = cached(run_dir, "inventory", inv_mod.build_inventory, logger)
        if args.limit:
            inv = limit_inventory(inv, args.limit, args.seed)
        inv.to_csv(run_dir / "results" / "inventory.csv", index=False)
        write_json(run_dir / "results" / "pairing.json", inv_mod.pairing_report(inv))
        inv_mod.suspicious_files(inv).to_csv(run_dir / "results" / "suspicious_files.csv", index=False)
        write_json(run_dir / "results" / "cross_split_name_overlap.json", inv_mod.cross_split_name_overlap(inv))

    # ── S0 self-test of the hashing code on 200 random images ──
    with Timer(logger, "S0 hash self-test"):
        pool = inv[(inv.kind == "image") & (inv.root == "yolo")].abspath.tolist()
        st = run_selftest(random.Random(args.seed).sample(pool, min(200, len(pool))))
        write_json(run_dir / "results" / "selftest.json", st)
        if not st["passed"]:
            logger.error("Hash self-test FAILED — aborting (see results/selftest.json)")
            sys.exit(1)

    # ── S2 image integrity + hashes ──
    with Timer(logger, "S2 image scan (decode, metadata, fingerprints, D4 hashes)"):
        scans = cached(run_dir, "image_scan", lambda: stage_scan(inv, args.workers, logger), logger)
    files = inv[inv.kind == "image"].merge(scans, on="abspath", how="left", suffixes=("", "_scan"))
    files.drop(columns=HASH_COLS).to_csv(run_dir / "results" / "image_files.csv", index=False)

    # ── S3 labels ──
    with Timer(logger, "S3 label parsing & box validity"):
        dims = {}
        for r in files.itertuples():
            key = (r.root, r.split if r.root == "yolo" else r.raw_folder, r.stem)
            if pd.notna(getattr(r, "width", np.nan)):
                dims.setdefault(key, (float(r.width), float(r.height)))
        label_rows = inv[inv.kind == "label"]
        lab_files, boxes = cached(run_dir, "labels", lambda: lab_mod.scan_labels(label_rows, dims), logger)
        lab_files.to_csv(run_dir / "results" / "label_files.csv", index=False)
        boxes.to_csv(run_dir / "results" / "boxes.csv", index=False)
        classes_txt = lab_mod.read_classes_txt(inv[inv.kind == "classes_txt"].abspath.tolist())
        write_json(run_dir / "results" / "classes_txt_contents.json", classes_txt)

    # ── S4 provenance: zip snapshots vs current files ──
    with Timer(logger, "S4 zip snapshot provenance"):
        crc = dict(zip(files[files.root == "yolo"].relpath, files[files.root == "yolo"].crc32))
        crc.update(dict(zip(lab_files[lab_files.root == "yolo"].relpath, lab_files[lab_files.root == "yolo"].label_crc32)))
        write_json(run_dir / "results" / "zip_provenance.json", inv_mod.compare_zip_to_current(inv, crc))

    # ── S5 exact duplicates & unique contents ──
    with Timer(logger, "S5 contents & exact duplicates"):
        contents, content_id = sim.build_contents(files)
        files["content_id"] = content_id
        pd.to_pickle(contents, run_dir / "cache" / "contents.pkl")
        files.drop(columns=HASH_COLS).to_pickle(run_dir / "cache" / "files.pkl")
        logger.info(f"  {len(files)} decodable image files -> {len(contents)} unique byte-contents")

    # ── S6 near-duplicate candidates: D4 hashes + DINOv2 embeddings ──
    P, D, A = (sim.hash_matrix(contents, k) for k in ("ph", "dh", "ah"))
    with Timer(logger, "S6a hash candidate search"):
        hash_pairs = cached(run_dir, "hash_pairs", lambda: sim.hamming_candidates(P, sim.CANDIDATE_PH_MAX), logger)
        logger.info(f"  {len(hash_pairs)} pHash candidate pairs (any of 8 transforms, d <= {sim.CANDIDATE_PH_MAX})")
    emb_path = run_dir / "cache" / "embeddings.npy"
    if args.skip_embeddings:
        E = np.zeros((len(contents), 1), dtype=np.float32)
        emb_pairs = pd.DataFrame(columns=["a", "b"])
    else:
        with Timer(logger, "S6b DINOv2 embeddings"):
            if emb_path.exists():
                E = np.load(emb_path)
            else:
                E = sim.compute_embeddings(contents.abspath.tolist(), args.device, logger=logger)
                np.save(emb_path, E)
        with Timer(logger, "S6c embedding kNN"):
            emb_pairs = cached(run_dir, "emb_pairs",
                               lambda: sim.knn_candidates(E, sim.KNN_K, sim.CANDIDATE_COS_MIN, args.device), logger)
            logger.info(f"  {len(emb_pairs)} embedding candidate pairs (cos >= {sim.CANDIDATE_COS_MIN})")
            np.save(run_dir / "cache" / "random_pair_cos.npy", sim.random_pair_cosines(E))
    with Timer(logger, "S6d pair evidence & classification"):
        cand = (pd.concat([hash_pairs, emb_pairs]).astype({"a": np.int64, "b": np.int64})
                .drop_duplicates().reset_index(drop=True))
        ev = sim.classify_edges(sim.pair_evidence(cand, contents, P, D, A, E))
        ev.to_pickle(run_dir / "cache" / "edges.pkl")
        logger.info("  edges by confidence (rules only): " + ev.confidence.value_counts().to_dict().__repr__())
    with Timer(logger, "S6e SIFT/RANSAC geometric verification"):
        vpath = run_dir / "cache" / "verification.pkl"
        todo = ev[sim.needs_verification(ev)]
        ver = pd.read_pickle(vpath) if vpath.exists() else pd.DataFrame()
        missing = todo.loc[todo.index.difference(ver.index)]  # incremental: only pairs not verified yet
        if len(missing):
            logger.info(f"  verifying {len(missing)} pairs ({len(ver)} cached)")
            new = verify.run_verification(missing, dict(zip(contents.content_id, contents.abspath)), args.workers,
                                          logger, store_dir=run_dir / "cache" / "sift_store")
            ver = pd.concat([ver, new])
            ver.to_pickle(vpath)
        final = sim.finalize_edges(ev, ver)
        logger.info("  edges by confidence (final): " + final.confidence.value_counts().to_dict().__repr__())

    logger.info("Core stages complete. Run report.py for analysis, figures and REPORT.md.")


if __name__ == "__main__":
    main()
