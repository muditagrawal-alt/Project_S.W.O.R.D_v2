"""Sensitivity of the near-duplicate detector on KNOWN augmented copies.

Re-runs the dataset's own Albumentations pipeline (albu_generate.py, isolated env) on randomly chosen
images, then pushes every (source, augmented) pair through exactly the same evidence → rules → SIFT
verification chain used by the audit, and reports the detection rate per rule and per applied transform.

    venv/bin/python research/tools/audit/aug_calibration.py --run-id <id> --albu-python <env>/bin/python
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import NUM_CLASSES, RESEARCH_ROOT, setup_logging, write_json  # noqa: E402
import similarity as sim  # noqa: E402
import verify  # noqa: E402
from image_scan import scan_image  # noqa: E402

HASH = [f"{k}{t}" for k in ("ph", "dh", "ah") for t in range(8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--albu-python", required=True)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run_dir = RESEARCH_ROOT / "runs" / args.run_id
    out = run_dir / "calibration_aug"
    out.mkdir(exist_ok=True)
    log = setup_logging(run_dir)

    files = pd.read_pickle(run_dir / "cache" / "files.pkl")
    y = files[(files.root == "yolo") & (files.status == "ok")]
    rng = np.random.default_rng(args.seed)
    src = pd.concat([g.iloc[rng.choice(len(g), min(args.per_class, len(g)), replace=False)]
                     for c, g in y.groupby("prefix_class") if 0 <= c < NUM_CLASSES])
    (out / "sources.txt").write_text("\n".join(src.abspath) + "\n")
    log.info(f"[calib] generating {args.copies} augmented copies for {len(src)} sources with the dataset's own pipeline")
    subprocess.run([args.albu_python, str(Path(__file__).with_name("albu_generate.py")), str(out / "sources.txt"),
                    str(out / "images"), str(args.copies), str(args.seed)], check=True)
    pairs = pd.read_csv(out / "images" / "pairs.csv")

    paths = list(dict.fromkeys(pairs.source.tolist() + pairs.augmented.tolist()))
    with ProcessPoolExecutor(max_workers=9) as ex:
        scans = pd.DataFrame(list(ex.map(scan_image, paths, chunksize=16)))
    for c in HASH:
        scans[c] = np.array([int(v) for v in scans[c]], dtype=np.uint64)
    scans["content_id"] = np.arange(len(scans))
    idx = dict(zip(scans.abspath, scans.content_id))
    E = sim.compute_embeddings(scans.abspath.tolist(), args.device, logger=log)
    P, D, A = (np.stack([scans[f"{k}{t}"].to_numpy() for t in range(8)]) for k in ("ph", "dh", "ah"))
    cand = pd.DataFrame({"a": pairs.source.map(idx), "b": pairs.augmented.map(idx)}).astype(np.int64)
    ev = sim.classify_edges(sim.pair_evidence(cand, scans, P, D, A, E))
    ver = verify.run_verification(ev, dict(zip(scans.content_id, scans.abspath)), logger=log)
    ev = sim.finalize_edges(ev, ver)
    ev["applied"] = pairs.applied.values
    ev.to_csv(out / "calibration_pairs.csv", index=False)

    detected = lambda df, col: float((df[col] != "LOW").mean())
    by_transform = []
    for t in ["none", "HorizontalFlip", "RandomBrightnessContrast", "Rotate", "Blur", "GaussNoise", "HueSaturationValue", "Affine"]:
        sub = ev[ev.applied == "none"] if t == "none" else ev[ev.applied.str.contains(t)]
        if len(sub):
            by_transform.append({"transform": t, "n": int(len(sub)), "detected_base_rules": detected(sub, "confidence_base"),
                                 "detected_final": detected(sub, "confidence"), "high_final": float((sub.confidence == "HIGH").mean())})
    res = {
        "n_sources": int(len(src)), "n_pairs": int(len(ev)),
        "recall_base_rules_high_or_medium": detected(ev, "confidence_base"),
        "recall_base_rules_high": float((ev.confidence_base == "HIGH").mean()),
        "recall_final_high_or_medium": detected(ev, "confidence"),
        "recall_final_high": float((ev.confidence == "HIGH").mean()),
        "by_transform": by_transform,
        "verification_inliers_quantiles": ev.n_inliers.quantile([0.01, 0.05, 0.1, 0.25, 0.5]).to_dict(),
        "missed_examples": ev[ev.confidence == "LOW"].head(20)[["a", "b", "ph_id", "ph_min", "T", "cos", "n_inliers", "applied"]].to_dict(orient="records"),
    }
    write_json(run_dir / "results" / "detector_sensitivity.json", res)
    log.info(f"[calib] recall (HIGH or MEDIUM): base rules {res['recall_base_rules_high_or_medium']:.3f} -> final {res['recall_final_high_or_medium']:.3f}")
    print(pd.DataFrame(by_transform).to_string())


if __name__ == "__main__":
    main()
