"""
Fetch the public CCTV datasets used for detector fine-tuning, Phase 4 evaluation and interval labels.
Everything lands under research/data/ (untracked). Downloaded content is treated as data only: no downloaded code
is executed. A provenance manifest (URLs, licenses, hashes, dates) is written next to each dataset.

    python research/tools/data/fetch_cctv_datasets.py --what mendeley ucf cctvgun

Datasets (academic use; cite them):
  mendeley  Ruiz-Santaquiteria J. et al., "Firearm-related action recognition and object detection dataset for video
            surveillance systems", Data in Brief 52 (2024) 110030, doi:10.1016/j.dib.2024.110030.
            CC BY-NC 3.0. https://data.mendeley.com/datasets/bbzpxhd22j/2  (SHA-256 published by Mendeley)
  ucf       Sultani W., Chen C., Shah M., "Real-world Anomaly Detection in Surveillance Videos", CVPR 2018.
            Research use. Bandwidth-limited selection, in priority order: the 73 videos annotated by CCTV-Gun,
            the Robbery/Shooting/Fighting/Assault test videos with official temporal annotations, a seeded sample
            of normal test videos, the splits and the temporal annotations (selective range reads of the 103 GB
            archive; per-file CRC-32 checked by zipfile). The server omits its intermediate certificate; it is
            fetched from the certificate's AIA URL and the full chain is verified against certifi's roots.
  cctvgun   Yellapragada S. et al., "CCTV-Gun: Benchmarking Handgun Detection in CCTV Images" (2023), Apache-2.0.
            Annotation files only (data/), from https://github.com/srikarym/CCTV-Gun at a pinned commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "research" / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import open_remote_zip  # noqa: E402

MENDELEY_URL = ("https://data.mendeley.com/public-files/datasets/bbzpxhd22j/files/"
                "f6b5b59a-27ad-4134-90fb-f56fee58bb75/file_downloaded")
MENDELEY_SHA256 = "b52fa773c20a22d7e93ccf776f22eb8c73184e346d3c62303d8c78cbc3cf3a6a"
UCF_ZIP = "https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip"
UCF_TEMPORAL = "https://www.crcv.ucf.edu/projects/real-world/Temporal_Anomaly_Annotation_For_Testing_Videos.zip"
UCF_DIRS = ["Robbery", "Shooting", "Fighting", "Assault", "Testing_Normal_Videos_Anomaly"]
UCF_TEST_CATEGORIES = ["Robbery", "Shooting", "Fighting", "Assault"]
CCTVGUN_REPO = "srikarym/CCTV-Gun"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def safe_extract(z: zipfile.ZipFile, member: zipfile.ZipInfo, dest: Path) -> Path:
    """Extract one member, refusing absolute paths and '..' (zip-slip)."""
    target = (dest / member.filename).resolve()
    if not str(target).startswith(str(dest.resolve()) + "/"):
        raise ValueError(f"unsafe path in archive: {member.filename}")
    if target.exists() and target.stat().st_size == member.file_size:
        return target
    return Path(z.extract(member, dest))


# ───────────────────────────── Mendeley ─────────────────────────────
def fetch_mendeley() -> None:
    out = DATA / "mendeley_firearm"
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / "Gun_Action_Recognition_Dataset.zip"
    if not (zpath.exists() and sha256(zpath) == MENDELEY_SHA256):
        log("mendeley: downloading 2.25 GB (resumable)")
        subprocess.run(["curl", "-fL", "-C", "-", "--retry", "5", "-o", str(zpath), MENDELEY_URL], check=True)
    digest = sha256(zpath)
    if digest != MENDELEY_SHA256:
        raise SystemExit(f"mendeley: SHA-256 mismatch {digest}")
    log("mendeley: SHA-256 verified; extracting")
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            safe_extract(z, m, out / "extracted")
    (out / "PROVENANCE.json").write_text(json.dumps({
        "dataset": "Firearm-related action recognition and object detection dataset for video surveillance systems",
        "citation": "Ruiz-Santaquiteria J, Muñoz JD, Maigler FJ, Deniz O, Bueno G. Data in Brief 52 (2024) 110030. "
                    "doi:10.1016/j.dib.2024.110030", "license": "CC BY-NC 3.0 (commercial use only with permission)",
        "source": "https://data.mendeley.com/datasets/bbzpxhd22j/2", "url": MENDELEY_URL,
        "sha256": digest, "fetched": now()}, indent=2))
    log("mendeley: done")


# ───────────────────────────── UCF-Crime ─────────────────────────────
def ucf_ca_bundle() -> str:
    """certifi roots + the intermediate the server omits (fetched from the leaf certificate's AIA URL)."""
    import certifi

    bundle = DATA / "ucf_crime" / ".ca_bundle.pem"
    if bundle.exists():
        return str(bundle)
    leaf = subprocess.run("echo | openssl s_client -connect www.crcv.ucf.edu:443 -servername www.crcv.ucf.edu 2>/dev/null "
                          "| openssl x509", shell=True, capture_output=True, text=True, check=True).stdout
    aia = subprocess.run(["openssl", "x509", "-noout", "-ext", "authorityInfoAccess"], input=leaf,
                         capture_output=True, text=True, check=True).stdout
    url = aia.split("CA Issuers - URI:")[1].split()[0]
    der = requests.get(url, timeout=60).content
    pem = subprocess.run(["openssl", "x509", "-inform", "der"], input=der, capture_output=True, check=True).stdout.decode()
    bundle.write_text(Path(certifi.where()).read_text() + "\n" + pem)
    return str(bundle)


def fetch_ucf(n_normals: int = 40) -> None:
    out = DATA / "ucf_crime"
    out.mkdir(parents=True, exist_ok=True)
    ca = ucf_ca_bundle()
    log("ucf: temporal annotations")
    tz = out / "Temporal_Anomaly_Annotation_For_Testing_Videos.zip"
    r = requests.get(UCF_TEMPORAL, verify=ca, timeout=120)
    r.raise_for_status()
    tz.write_bytes(r.content)
    with zipfile.ZipFile(tz) as z:
        for m in z.infolist():
            safe_extract(z, m, out / "annotations")
    log("ucf: opening remote archive index")
    z = open_remote_zip(UCF_ZIP, ca)
    by_name = {Path(m.filename).name: m for m in z.infolist() if not m.is_dir()}
    # priority order (bandwidth is limited): 1) videos annotated by CCTV-Gun (detector training),
    # 2) test videos of the selected categories with official temporal annotations, 3) a seeded sample of normals
    cg = json.loads((DATA / "cctv_gun" / "data" / "ucf" / "frames.json").read_text())
    p1 = [f"{v}_x264.mp4" for v in sorted(cg)]
    ann = (out / "annotations").rglob("Temporal_Anomaly_Annotation.txt")
    rows = [l.split() for l in next(ann).read_text().splitlines() if l.strip()]
    p2 = [r[0] for r in rows if r[1] in UCF_TEST_CATEGORIES]
    normals = sorted(r[0] for r in rows if r[1] == "Normal")
    import random
    p3 = sorted(random.Random(0).sample(normals, min(n_normals, len(normals))))
    splits = [m for m in z.infolist() if not m.is_dir() and ("/Anomaly_Detection_splits/" in m.filename
                                                               or "/Action_Regnition_splits/" in m.filename)]
    order, seen = [], set()
    for name in p1 + p2 + p3:
        if name in by_name and name not in seen:
            order.append(by_name[name])
            seen.add(name)
    missing = [n for n in p1 + p2 + p3 if n not in by_name]
    wanted = splits + order
    total = sum(m.file_size for m in wanted)
    log(f"ucf: {len(wanted)} files ({len(p1)} CCTV-Gun, {len(p2)} annotated test, {len(p3)} normal), {total / 1e9:.2f} GB; "
        f"missing in archive: {missing}")
    done = 0
    for i, m in enumerate(wanted, 1):
        safe_extract(z, m, out / "extracted")   # zipfile verifies each member's CRC-32 while reading
        done += m.file_size
        if i % 10 == 0 or i == len(wanted):
            log(f"ucf: {i}/{len(wanted)} files, {done / 1e9:.2f}/{total / 1e9:.2f} GB")
    (out / "PROVENANCE.json").write_text(json.dumps({
        "dataset": "UCF-Crime (selected categories)", "citation": "Sultani W, Chen C, Shah M. Real-world Anomaly Detection "
        "in Surveillance Videos. CVPR 2018.", "license": "research use; cite the paper",
        "archive": UCF_ZIP, "temporal_annotations": UCF_TEMPORAL,
        "selection": {"cctv_gun_videos": p1, "annotated_test_videos": p2, "normal_test_sample_seed0": p3},
        "files": [{"name": m.filename, "size": m.file_size, "crc32": f"{m.CRC:08x}"} for m in wanted],
        "temporal_annotations_sha256": sha256(tz), "fetched": now(),
        "tls": "server chain completed with the AIA-fetched intermediate and verified against certifi roots"}, indent=2))
    log("ucf: done")


# ───────────────────────────── CCTV-Gun ─────────────────────────────
def fetch_cctvgun() -> None:
    out = DATA / "cctv_gun"
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["gh", "api", f"repos/{CCTVGUN_REPO}/commits/master", "--jq", ".sha"],
                          capture_output=True, text=True, check=True).stdout.strip()
    tree = json.loads(subprocess.run(["gh", "api", f"repos/{CCTVGUN_REPO}/git/trees/{head}?recursive=1"],
                                     capture_output=True, text=True, check=True).stdout)
    files = [t["path"] for t in tree["tree"] if t["type"] == "blob" and (t["path"].startswith("data/")
                                                                       or t["path"] in ("LICENSE", "README.md", "CITATION.cff"))]
    for f in files:
        dest = (out / f).resolve()
        if not str(dest).startswith(str(out.resolve()) + "/"):
            raise ValueError(f)
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"https://raw.githubusercontent.com/{CCTVGUN_REPO}/{head}/{f}", timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
    (out / "PROVENANCE.json").write_text(json.dumps({
        "dataset": "CCTV-Gun annotations", "citation": "Yellapragada S, et al. CCTV-Gun: Benchmarking Handgun Detection "
        "in CCTV Images. 2023. arXiv:2303.10703", "license": "Apache-2.0", "repo": f"https://github.com/{CCTVGUN_REPO}",
        "commit": head, "files": {f: sha256(out / f) for f in files}, "fetched": now(),
        "note": "annotation files only; USRT (mock-attack) annotations are never used for training here"}, indent=2))
    log(f"cctvgun: {len(files)} files at {head[:10]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", nargs="+", choices=["mendeley", "ucf", "cctvgun"], required=True)
    ap.add_argument("--ucf-normals", type=int, default=40, help="seeded sample of normal test videos")
    args = ap.parse_args()
    for w in args.what:
        fetch_ucf(args.ucf_normals) if w == "ucf" else {"mendeley": fetch_mendeley, "cctvgun": fetch_cctvgun}[w]()


if __name__ == "__main__":
    main()
