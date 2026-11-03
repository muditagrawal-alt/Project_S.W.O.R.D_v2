"""Person-zoom weapon detection (tracking/zoom.py): crop windows, coordinate mapping, border filter, merging."""

import copy
from types import SimpleNamespace

import numpy as np

from tracking.zoom import ZoomConfig, detect_full_frame, detect_in_windows, merge_detections, zoom_windows


def person(bbox, wrists=((150, 200), (170, 200)), score=0.9):
    kp = [[0.0, 0.0]] * 17
    sc = [0.0] * 17
    kp[9], kp[10] = list(wrists[0]), list(wrists[1])
    sc[9] = sc[10] = score
    return {"bbox": list(bbox), "keypoints": kp, "scores": sc}


def test_windows_are_square_clipped_and_follow_modes():
    p = person((100, 100, 200, 300))
    (w,) = zoom_windows([p], (1080, 1920), ZoomConfig(modes=("person",), person_margin=0.25))
    assert w[2] - w[0] == w[3] - w[1] == 300           # max(w, h) * 1.5
    hands = zoom_windows([p], (1080, 1920), ZoomConfig(modes=("hands",), hand_frac=0.6))
    assert len(hands) == 2 and all(x2 - x1 == 120 for x1, _, x2, _ in hands)
    edge = zoom_windows([person((0, 0, 50, 900))], (1080, 1920), ZoomConfig(modes=("person",)))
    assert edge[0][0] == 0 and edge[0][1] == 0 and edge[0][3] <= 1080
    low = person((100, 100, 200, 300), score=0.1)          # wrists below kpt_thr: no hand crops
    assert zoom_windows([low], (1080, 1920), ZoomConfig(modes=("hands",))) == []


class FakeModel:
    """Returns one box per crop at a fixed position in crop coordinates."""

    def __init__(self, box):
        self.box = np.array([box], dtype=float)

    def predict(self, crops, **kw):
        t = lambda a: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: a))  # noqa: E731
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=t(self.box), conf=t(np.array([0.8])), cls=t(np.array([0]))))
                for _ in crops]


def test_detections_map_back_and_border_cuts_are_dropped():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    b, s, c = detect_in_windows(FakeModel((10, 20, 40, 60)), frame, [(500, 300, 800, 600)])
    assert b.tolist() == [[510, 320, 540, 360]] and s.tolist() == [0.8]
    # a box touching the crop's right edge (inside the frame) is a cut object -> dropped
    b, _, _ = detect_in_windows(FakeModel((250, 20, 300, 60)), frame, [(500, 300, 800, 600)])
    assert len(b) == 0
    # ...but kept when that crop edge is the frame edge
    b, _, _ = detect_in_windows(FakeModel((250, 20, 300, 60)), frame, [(1620, 300, 1920, 600)])
    assert len(b) == 1
    # flip: the same box in the flipped 300 px crop (x 10..40) is x 260..290 of the crop, i.e. 760..790 in the frame
    b, s, _ = detect_in_windows(FakeModel((10, 20, 40, 60)), frame, [(500, 300, 800, 600)], flip=True)
    assert b.tolist() == [[510, 320, 540, 360], [760, 320, 790, 360]] and s.tolist() == [0.8, 0.8]


def test_full_frame_tta_unflips_and_merges_scales():
    class Model:   # a box on the left third of whatever image it sees; the score depends on the input size
        sizes = []

        def predict(self, img, imgsz, **kw):
            self.sizes.append(imgsz)
            box = np.array([[10.0, 20.0, 40.0, 60.0]])
            if img[0, 0, 0] == 7:          # the flipped frame (marker pixel moved to the left edge)
                box = np.array([[60.0, 20.0, 90.0, 60.0]])
            t = lambda a: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: a))  # noqa: E731
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=t(box), conf=t(np.array([imgsz / 2000])), cls=t(np.array([0]))))]

    frame = np.zeros((100, 100, 3), np.uint8)
    frame[0, -1] = 7
    m = Model()
    b, s, _ = detect_full_frame(m, frame, 1280)
    assert m.sizes == [1280] and b.tolist() == [[10, 20, 40, 60]]
    m.sizes = []
    b, s, _ = detect_full_frame(m, frame, 1280, extra_sizes=(960,), flip=True)
    assert m.sizes == [1280, 1280, 960, 960]
    # the flipped frame's box (60..90) maps back to 10..40 and merges; the highest score (1280 px) is kept
    assert b.tolist() == [[10, 20, 40, 60]] and s.tolist() == [0.64]


def test_merge_keeps_highest_and_separate_classes():
    full = (np.array([[0, 0, 10, 10]]), np.array([0.3]), np.array([0]))
    crop = (np.array([[1, 1, 10, 10], [0, 0, 10, 10]]), np.array([0.9, 0.5]), np.array([0, 4]))
    b, s, c = merge_detections([full, crop])
    assert sorted(zip(s.round(2).tolist(), c.tolist())) == [(0.5, 4), (0.9, 0)]
    b, s, c = merge_detections([full, crop], class_agnostic=True)
    assert s.tolist() == [0.9]
    assert len(merge_detections([])[0]) == 0


def test_merge_reports_source_and_rederive_drops_weak_zoom_detections():
    full = (np.array([[0, 0, 10, 10]]), np.array([0.3]), np.array([0]))
    crop = (np.array([[100, 100, 120, 120], [1, 1, 10, 10]]), np.array([0.2, 0.9]), np.array([0, 0]))
    b, s, c, src = merge_detections([full, crop], return_source=True)
    assert sorted(zip(s.round(2).tolist(), src.tolist())) == [(0.2, 1), (0.9, 1)]   # crop box won the overlap

    from tracking.pipeline import rederive_weapons

    person = {"track_id": 1, "bbox": [90, 50, 150, 250], "keypoints": [[0.0, 0.0]] * 17, "scores": [0.0] * 17}
    frames = [{"timestamp": 0.0, "persons": [person], "weapons": [],
               "weapon_detections": [{"bbox": [100, 100, 120, 120], "cls": 0, "score": 0.2, "src": "zoom"},
                                     {"bbox": [0, 0, 10, 10], "cls": 0, "score": 0.3, "src": "full"}]}]
    fresh = copy.deepcopy(frames)
    rederive_weapons(frames, fps=2.0, zoom_min_score=0.25)
    assert [(w["score"], w["src"]) for w in frames[0]["weapons"]] == [(0.3, "full")]
    assert len(frames[0]["weapon_detections"]) == 1   # dropped from the raw list too, as live; re-score from a fresh copy
    rederive_weapons(fresh, fps=2.0, zoom_min_score=0.0)
    assert len(fresh[0]["weapons"]) == 2


def test_min_window_skips_strong_magnification_and_rescoring_matches():
    from tracking.zoom import from_allowed_window

    big, small = person((100, 100, 200, 300)), person((600, 100, 640, 180), wrists=((610, 150), (630, 150)))
    cfg = ZoomConfig(modes=("hands",), min_window=96)
    wins = zoom_windows([big, small], (1080, 1920), cfg, with_natural_side=True)
    assert len(wins) == 2 and all(n >= 96 for _, n in wins)          # small person: 0.6 * 80 = 48 px hand windows skipped
    assert len(zoom_windows([big, small], (1080, 1920), ZoomConfig(modes=("hands",)))) == 4
    assert from_allowed_window([145, 195, 155, 205], [big, small], (1080, 1920), cfg)
    assert not from_allowed_window([605, 145, 615, 155], [big, small], (1080, 1920), cfg)


def test_rederive_applies_box_size_limit_and_min_window():
    from tracking.pipeline import rederive_weapons

    small = person((600, 100, 640, 180), wrists=((610, 150), (630, 150)))
    small["track_id"] = 1
    dets = [{"bbox": [10, 10, 900, 500], "cls": 5, "score": 0.9, "src": "full"},        # 21% of a 1920x1080 frame
            {"bbox": [605, 145, 615, 155], "cls": 0, "score": 0.9, "src": "zoom"},      # from a 48 px hand window
            {"bbox": [300, 300, 330, 340], "cls": 0, "score": 0.9, "src": "full"}]
    frames = [{"weapon_detections": [dict(d) for d in dets], "persons": [small]}]
    rederive_weapons(frames, 2.0)
    assert len(frames[0]["weapons"]) == 3
    rederive_weapons(frames, 2.0, max_area_frac=0.125, zoom_min_window=96, frame_shape=(1080, 1920))
    assert [w["bbox"] for w in frames[0]["weapons"]] == [[300, 300, 330, 340]]
    assert [w["bbox"] for w in frames[0]["weapon_detections"]] == [[300, 300, 330, 340]]   # B0 reads the raw list


def test_live_step_applies_box_size_limit_and_min_window():
    from tracking.pipeline import SwordPipeline
    from tracking.tracker import ByteTrackWrapper

    class Phase2:
        def update(self, frame, t):
            return [{"track_id": 1, "bbox": [600, 100, 620, 150], "score": 0.9}], []

    class Model:   # full frame: one frame-filling box and one small box; crops: one box each
        def predict(self, src, **kw):
            imgs = src if isinstance(src, list) else [src]
            out = []
            for im in imgs:
                full = im.shape[0] == 1080
                b = np.array([[10, 10, 900, 500], [300, 300, 330, 340]], float) if full else np.array([[20, 20, 30, 30]], float)
                out.append(SimpleNamespace(boxes=SimpleNamespace(
                    xyxy=SimpleNamespace(cpu=lambda b=b: SimpleNamespace(numpy=lambda: b)),
                    conf=SimpleNamespace(cpu=lambda n=len(b): SimpleNamespace(numpy=lambda: np.full(n, 0.9))),
                    cls=SimpleNamespace(cpu=lambda n=len(b): SimpleNamespace(numpy=lambda: np.zeros(n))))))
            return out

    p = SwordPipeline.__new__(SwordPipeline)
    p.phase2, p.pose, p.weapon_det, p.use_ensemble, p.device, p.det_conf = Phase2(), None, Model(), False, "cpu", 0.1
    p.weapon_imgsz, p.zoom_imgsz, p.zoom_min_score, p.weapon_trk = 1280, 640, 0.0, ByteTrackWrapper(track_buffer=5)
    p.weapon_zoom, p.zoom_min_window, p.weapon_max_area, p.verifier, p.weapon_tta_scales = ("person",), 0, 1.0, None, ()
    p.weapon_tta_flip = p.zoom_flip = False
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert len(p.step(frame, 0, 0.0)["weapon_detections"]) == 3        # big + small + one zoom box
    p.zoom_min_window, p.weapon_max_area = 96, 0.125                       # person window 75 px < 96: no crops
    rec = p.step(frame, 1, 0.5)
    assert [d["bbox"] for d in rec["weapon_detections"]] == [[300.0, 300.0, 330.0, 340.0]]


def test_rederive_fuses_verifier_scores_and_gates():
    from tracking.pipeline import rederive_weapons

    dets = [{"bbox": [300, 300, 330, 340], "cls": 0, "score": 0.81, "src": "full", "verify": 0.25},
            {"bbox": [500, 300, 530, 340], "cls": 0, "score": 0.64, "src": "full", "verify": 0.04}]
    fresh = lambda: [{"weapon_detections": [dict(d) for d in dets], "persons": []}]  # noqa: E731
    f = fresh()
    rederive_weapons(f, 2.0, verify_mode="geo")
    assert [w["score"] for w in f[0]["weapon_detections"]] == [0.45, 0.16]          # sqrt(0.81*0.25), sqrt(0.64*0.04)
    assert f[0]["weapon_detections"][0]["det_score"] == 0.81
    f = fresh()
    rederive_weapons(f, 2.0, verify_mode="product", verify_min=0.1)
    assert [w["score"] for w in f[0]["weapon_detections"]] == [0.203]                 # second box gated out
