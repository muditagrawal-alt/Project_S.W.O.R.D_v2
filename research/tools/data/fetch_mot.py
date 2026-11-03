"""
Fetch the multi-object-tracking benchmark data used to evaluate Phase 2 (person tracking), reading only the needed
members of the official MOTChallenge archives over HTTP range requests (remote_zip.py).

  MOT17 train, FRCNN copies (images are identical in the DPM/FRCNN/SDP copies): ground truth, public detections,
  seqinfo and the images of the VALIDATION HALF (second half of every sequence: the protocol of the ByteTrack,
  OC-SORT and BoT-SORT ablations, so results can be set beside theirs). --full also fetches the first halves.
  MOT15 PETS09-S2L1: ground truth, detections, seqinfo, all 795 images (the scene of OpenCV's vtest.avi street clip).

Licence: MOTChallenge data, CC BY-NC-SA 3.0 (research use; cite Milan et al. 2016 and Leal-Taixé et al. 2015).
Nothing is executed from the archives; only .jpg, .txt and .ini members are written.

    python research/tools/data/fetch_mot.py --out research/data/mot
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import struct
import sys
import time
import zipfile
import zlib
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import open_remote_zip  # noqa: E402

MOT17 = "https://motchallenge.net/data/MOT17.zip"
MOT17_LABELS = "https://motchallenge.net/data/MOT17Labels.zip"
MOT15 = "https://motchallenge.net/data/MOT15.zip"
TRAIN = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
SAFE = (".jpg", ".txt", ".ini")


class Archive:
    """A remote zip that reopens its connection and retries with backoff when the server drops it."""

    def __init__(self, url: str):
        self.url, self.z = url, open_remote_zip(url)

    def infolist(self):
        return self.z.infolist()

    def read(self, info, tries: int = 8) -> bytes:
        for k in range(tries):
            try:
                return self.z.read(info)
            except (OSError, zipfile.BadZipFile, requests.exceptions.RequestException) as e:
                if k == tries - 1:
                    raise
                wait = min(60, 2 ** k)
                print(f"  retry {k + 1} for {info.filename} in {wait} s ({type(e).__name__})", flush=True)
                time.sleep(wait)
                self.z = open_remote_zip(self.url)
        raise RuntimeError("unreachable")


def write(z: Archive, info, dest: Path) -> int:
    if not info.filename.endswith(SAFE) or ".." in info.filename:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == info.file_size:
        return 0
    dest.write_bytes(z.read(info))
    return info.file_size


def fetch_span(url: str, infos: list, dest_of, tries: int = 8) -> int:
    """Download the byte span holding `infos` (members stored next to each other in the archive) with ONE range
    request, then unpack each member locally and check its CRC: far faster than one request per member."""
    todo = [i for i in infos if not (dest_of(i).exists() and dest_of(i).stat().st_size == i.file_size)]
    if not todo:
        return 0
    todo.sort(key=lambda i: i.header_offset)
    start = todo[0].header_offset
    end = todo[-1].header_offset + 30 + len(todo[-1].filename.encode()) + 1024 + todo[-1].compress_size  # header slack
    for k in range(tries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=600)
            r.raise_for_status()
            buf = r.content
            break
        except requests.exceptions.RequestException as e:
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} ({type(e).__name__})", flush=True)
            time.sleep(min(60, 2 ** k))
    n = 0
    for i in todo:
        off = i.header_offset - start
        assert buf[off:off + 4] == b"PK\x03\x04", f"bad local header for {i.filename}"
        fnlen, exlen = struct.unpack("<HH", buf[off + 26:off + 30])
        data = buf[off + 30 + fnlen + exlen: off + 30 + fnlen + exlen + i.compress_size]
        if i.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompressobj(-15).decompress(data)
        elif i.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"unsupported compression {i.compress_type} for {i.filename}")
        assert zlib.crc32(data) & 0xFFFFFFFF == i.CRC, f"CRC mismatch for {i.filename}"
        dest = dest_of(i)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        n += len(data)
    return n


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full", action="store_true", help="MOT17: also the first half of every sequence")
    args = ap.parse_args()
    out = Path(args.out)
    t0, nbytes, prov = time.time(), 0, {"created": datetime.now().isoformat(timespec="seconds"),
                                        "licence": "CC BY-NC-SA 3.0 (MOTChallenge)", "sources": {}, "sequences": {}}
    # 1) MOT17 labels: gt, det, seqinfo (10 MB archive)
    zl = Archive(MOT17_LABELS)
    for i in zl.infolist():
        p = i.filename.split("/")
        if len(p) >= 3 and p[0] == "train" and p[1].endswith("-FRCNN"):
            nbytes += write(zl, i, out / "MOT17" / "train" / "/".join(p[1:]))
    prov["sources"]["MOT17Labels"] = MOT17_LABELS
    # 2) MOT17 images: validation half (or all)
    z = Archive(MOT17)
    prov["sources"]["MOT17"] = MOT17
    members = {i.filename: i for i in z.infolist()}
    for seq in TRAIN:
        sdir = out / "MOT17" / "train" / f"{seq}-FRCNN"
        ini = configparser.ConfigParser()
        ini.read(sdir / "seqinfo.ini")
        n = int(ini["Sequence"]["seqLength"])
        first = 1 if args.full else n // 2 + 1
        infos = [members[f"MOT17/train/{seq}-FRCNN/img1/{f:06d}.jpg"] for f in range(first, n + 1)]
        nbytes += fetch_span(MOT17, infos, lambda i: sdir / "img1" / i.filename.rsplit("/", 1)[1])
        prov["sequences"][seq] = {"frames": n, "fetched_frames": [first, n], "val_half_first_frame": n // 2 + 1,
                                  "gt_sha256": sha(sdir / "gt" / "gt.txt")}
        print(f"{seq}: frames {first}-{n} ({nbytes / 1e6:.0f} MB so far, {time.time() - t0:.0f} s)", flush=True)
    # 3) MOT15 PETS09-S2L1 (complete)
    z15 = Archive(MOT15)
    prov["sources"]["MOT15"] = MOT15
    pets = [i for i in z15.infolist() if i.filename.startswith("MOT15/train/PETS09-S2L1/") and i.filename.endswith(SAFE)]
    nbytes += fetch_span(MOT15, pets, lambda i: out / "MOT15" / "train" / i.filename.split("MOT15/train/", 1)[1])
    prov["sequences"]["PETS09-S2L1"] = {"gt_sha256": sha(out / "MOT15/train/PETS09-S2L1/gt/gt.txt")}
    prov["bytes_fetched"], prov["minutes"] = nbytes, round((time.time() - t0) / 60, 1)
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2))
    print(f"done: {nbytes / 1e6:.0f} MB in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
