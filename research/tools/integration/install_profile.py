"""
Install a system profile (pipeline/profiles.py): copy the chosen detector into weights/ under a stable name, record
its provenance, and write the profile (detector, input size, zoom, thresholds, fusion settings) to
config/profiles.json. Optionally take the zoom threshold and rule parameters from a dev-tuning result
(research/tools/fusion/tune_rules.py results.json).

    python research/tools/integration/install_profile.py --name cctv-zoom \\
        --weights research/runs/<detector run>/train/weights/best.pt --weights-name yolo26s_cctv_zoom_best.pt \\
        --imgsz 1280 --zoom hands --tuning research/runs/<id>_tune-rules/results.json --default
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.profiles import PROFILES_FILE  # noqa: E402



def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--weights-name", required=True, help="file name under weights/")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tta-scales", nargs="*", type=int, default=[], help="extra full-frame sizes (multi-scale test-time "
                                                                         "augmentation, tracking/zoom.py)")
    ap.add_argument("--tta-flip", action="store_true", help="also detect the horizontally flipped full frame")
    ap.add_argument("--zoom-flip", action="store_true", help="also detect the horizontally flipped zoom crops")
    ap.add_argument("--zoom", default="", help="'', person, hands, person+hands")
    ap.add_argument("--zoom-min-score", type=float, default=None)
    ap.add_argument("--tuning", help="tune_rules.py results.json: take zoom_min_score and rule parameters from 'selected'")
    ap.add_argument("--operating-point", help="with --tuning: the false-alarm budget of the operating point to install "
                                              "(default: the first budget)")
    ap.add_argument("--verifier", help="second-stage verifier weights (copied to weights/); its fusion mode and minimum "
                                       "come from --tuning")
    ap.add_argument("--verifier-name", help="file name under weights/ for --verifier (default: its own file name)")
    ap.add_argument("--phase2", default=None, help="Phase 2 module (pipeline/profiles.py PHASE2_KINDS)")
    ap.add_argument("--track-seconds", type=float, default=None, help="Phase 2 lost-track time")
    ap.add_argument("--description", default="")
    ap.add_argument("--evaluated-in", default="")
    ap.add_argument("--default", action="store_true", help="make this the default profile")
    ap.add_argument("--weights-dir", default=str(PROJECT_ROOT / "weights"), help=argparse.SUPPRESS)   # tests
    ap.add_argument("--profiles-file", default=str(PROFILES_FILE), help=argparse.SUPPRESS)            # tests
    args = ap.parse_args()
    profiles_file = Path(args.profiles_file)
    src = Path(args.weights).resolve()
    dst = Path(args.weights_dir) / args.weights_name
    provenance = Path(args.weights_dir) / "provenance.json"
    if src != dst.resolve():
        shutil.copy2(src, dst)
    prov = json.loads(provenance.read_text()) if provenance.exists() else {}
    prov[args.weights_name] = {"source": str(src.relative_to(PROJECT_ROOT)) if src.is_relative_to(PROJECT_ROOT) else str(src),
                               "sha256": sha256(dst), "installed": datetime.now().isoformat(timespec="seconds")}
    provenance.write_text(json.dumps(prov, indent=2))
    fusion, zmin, max_area, min_window, verify_mode, verify_min = {}, args.zoom_min_score, 1.0, 0, "none", 0.0
    if args.tuning:
        tun = json.loads(Path(args.tuning).read_text())
        sel = dict((tun["operating_points"][args.operating_point] if args.operating_point else tun)["selected"]["params"])
        tuned_zmin = sel.pop("zoom_min_score", None)
        zmin = tuned_zmin if zmin is None else zmin
        max_area = sel.pop("max_area_frac", 1.0)
        min_window = sel.pop("zoom_min_window", 0)
        verify_mode, verify_min = sel.pop("verify_mode", "none"), sel.pop("verify_min", 0.0)
        fusion = sel
    profile = {"description": args.description, "weapon_model": f"weights/{args.weights_name}", "weapon_imgsz": args.imgsz,
               **({"weapon_tta_scales": args.tta_scales} if args.tta_scales else {}),
               **({"weapon_tta_flip": True} if args.tta_flip else {}), **({"zoom_flip": True} if args.zoom_flip else {}),
               "weapon_zoom": [m for m in args.zoom.split("+") if m], "zoom_min_score": zmin or 0.0,
               "zoom_min_window": int(min_window), "weapon_max_area": float(max_area), "fusion": fusion,
               **({"phase2": args.phase2} if args.phase2 else {}),
               **({"track_seconds": args.track_seconds} if args.track_seconds else {}),
               "evaluated_in": args.evaluated_in, "tuning": args.tuning or None, "operating_point": args.operating_point,
               "installed": datetime.now().isoformat(timespec="seconds")}
    if args.verifier and (verify_mode != "none" or verify_min > 0):
        vsrc = Path(args.verifier).resolve()
        vname = args.verifier_name or vsrc.name
        vdst = Path(args.weights_dir) / vname
        if vsrc != vdst.resolve():
            shutil.copy2(vsrc, vdst)
        prov[vname] = {"source": str(vsrc), "sha256": sha256(vdst), "installed": datetime.now().isoformat(timespec="seconds")}
        provenance.write_text(json.dumps(prov, indent=2))
        profile.update(weapon_verifier=f"weights/{vname}", verify_mode=verify_mode, verify_min=float(verify_min))
    profiles_file.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(profiles_file.read_text()) if profiles_file.exists() else {"profiles": {}, "default": "original"}
    doc["profiles"][args.name] = profile
    if args.default:
        doc["default"] = args.name
    profiles_file.write_text(json.dumps(doc, indent=2))
    print(json.dumps({args.name: profile, "default": doc["default"]}, indent=2))


if __name__ == "__main__":
    main()
