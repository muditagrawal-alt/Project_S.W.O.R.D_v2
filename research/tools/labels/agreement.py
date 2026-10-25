"""
Inter-annotator agreement for interval labels (two CSVs in fusion/labels.py format, e.g. two exports of
/api/label/export?annotator=...). For every video labelled in both files:
  - Cohen's kappa over 10 Hz samples of the scene level (4 classes) and of the binary alarm (>= SUSPICIOUS_ACTIVITY)
  - boundary agreement: for each event (>= SUSPICIOUS) in A, the |start| and |end| offsets to the best-overlapping
    event in B

    python research/tools/labels/agreement.py a.csv b.csv [--rate 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.evaluate import segments  # noqa: E402
from fusion.labels import by_video, load_labels, scene_levels  # noqa: E402


def kappa(a: np.ndarray, b: np.ndarray, k: int) -> float | None:
    if len(a) == 0:
        return None
    cm = np.zeros((k, k))
    np.add.at(cm, (a, b), 1)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = float((cm.sum(0) * cm.sum(1)).sum()) / n ** 2
    return None if pe >= 1 else (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rate", type=float, default=10.0)
    args = ap.parse_args()
    A, B = by_video(load_labels(Path(args.a))), by_video(load_labels(Path(args.b)))
    common = sorted(set(A) & set(B))
    la, lb, offsets = [], [], []
    for v in common:
        end = max(iv.end_s for iv in A[v] + B[v])
        t = np.arange(0, end + 1e-9, 1 / args.rate)
        ya, yb = scene_levels(A[v], t), scene_levels(B[v], t)
        la.append(ya)
        lb.append(yb)
        ea, eb = segments(t, ya >= 2, 1.0), segments(t, yb >= 2, 1.0)
        for s, e in ea:
            best = max(eb, key=lambda x: min(e, x[1]) - max(s, x[0]), default=None)
            if best and min(e, best[1]) - max(s, best[0]) >= 0:
                offsets.append((abs(s - best[0]), abs(e - best[1])))
    ya, yb = (np.concatenate(la), np.concatenate(lb)) if la else (np.zeros(0, int), np.zeros(0, int))
    res = {"videos_in_both": len(common), "samples": int(len(ya)),
           "kappa_4_levels": None if not len(ya) else round(kappa(ya, yb, 4), 3),
           "kappa_alarm": None if not len(ya) else round(kappa((ya >= 2).astype(int), (yb >= 2).astype(int), 2), 3),
           "event_start_offset_s_median": round(float(np.median([o[0] for o in offsets])), 2) if offsets else None,
           "event_end_offset_s_median": round(float(np.median([o[1] for o in offsets])), 2) if offsets else None,
           "matched_events": len(offsets)}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
