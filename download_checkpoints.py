"""
S.W.O.R.D. Phase 4 - Checkpoint Downloader
Downloads verified pretrained checkpoints from the official OpenMMLab / PYSKL model zoo.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

CHECKPOINTS = {
    "posec3d_ntu120": {
        "name": "PoseC3D NTU-120 XSub Keypoint (SlowOnly-R50)",
        "filename": "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth",
        "url": "https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth",
        "expected_bytes": 8314882,
        "default": True,
    },
    "stgcnpp_ntu120": {
        "name": "ST-GCN++ NTU-120 XSub HRNet 2D Keypoint (Joint stream)",
        "filename": "stgcn_pyskl_ntu120_xsub_hrnet_j.pth",
        "url": "http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/j.pth",
        "expected_bytes": 12536215,
        "default": True,
    },
    "posec3d_ntu60": {
        "name": "PoseC3D NTU-60 XSub Keypoint (MMAction2 v1.x)",
        "filename": "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth",
        "url": "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth",
        "expected_bytes": 8196656,
        "default": False,
    }
}


def download_file(url: str, dest_path: Path, expected_bytes: int = None):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_bytes is None or actual_size == expected_bytes:
            print(f"[SKIP] {dest_path.name} already exists ({actual_size:,} bytes).")
            return True
        else:
            print(f"[WARN] Existing file size {actual_size} != expected {expected_bytes}. Re-downloading...")

    print(f"[DOWNLOADING] {url}")
    print(f"         --> {dest_path}")
    
    start_time = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 64
            
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    bar = ("=" * int(percent // 4)).ljust(25)
                    sys.stdout.write(f"\r  [{bar}] {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)")
                    sys.stdout.flush()
        elapsed = time.time() - start_time
        print(f"\n[DONE] Saved to {dest_path} in {elapsed:.2f}s ({dest_path.stat().st_size:,} bytes).")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download official pretrained skeleton action checkpoints.")
    parser.add_argument("--model", type=str, default="all", choices=["all", "posec3d_ntu120", "stgcnpp_ntu120", "posec3d_ntu60"],
                        help="Model checkpoint to download")
    parser.add_argument("--dest_dir", type=str, default=str(Path(__file__).parent / "checkpoints"),
                        help="Destination directory for checkpoints")
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    targets = [args.model] if args.model != "all" else [k for k, v in CHECKPOINTS.items() if v["default"]]

    print(f"=== S.W.O.R.D. Phase 4 Pretrained Checkpoint Downloader ===")
    print(f"Destination: {dest_dir.resolve()}\n")

    success_count = 0
    for key in targets:
        info = CHECKPOINTS[key]
        print(f"Processing: {info['name']}")
        dest = dest_dir / info["filename"]
        ok = download_file(info["url"], dest, info.get("expected_bytes"))
        if ok:
            success_count += 1
        print("-" * 60)

    print(f"Finished: {success_count}/{len(targets)} checkpoints ready.")


if __name__ == "__main__":
    main()
