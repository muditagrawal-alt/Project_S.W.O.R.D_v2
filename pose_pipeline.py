"""
Project S.W.O.R.D v2 — Human Pose Extraction & Biomechanical Telemetry Pipeline
Powered by pretrained RTMPose-m (OpenMMLab).

Features:
- Extracts 12 key functional joints:
  * Wrists (left, right)
  * Elbows (left, right)
  * Shoulders (left, right)
  * Hips (left, right)
  * Knees (left, right)
  * Ankles (left, right)
- Biomechanical & Threat Analytics:
  * Hand-to-waist distance (pixel & torso-normalized)
  * Wrist velocity between consecutive frames (px/sec with EMA smoothing)
  * Left and right arm interior angles (at elbows)
  * Body orientation & posture classification (Frontal, Profile Left/Right, Back, Tilt)
- HUD Visualizer:
  * Color-coded skeleton bones & joint nodes
  * Interior elbow angle badges
  * Hand-to-waist reach lines
  * Wrist velocity vectors
  * Body orientation HUD badge
- Standalone CLI & callable library interface for integration into app.py
"""

import os
import sys
import time
import math
import argparse
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# COCO 17 KEYPOINT TOPOLOGY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
COCO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# The 12 target keypoints requested for extraction
TARGET_JOINTS = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Skeleton connection pairs (COCO topology)
SKELETON_PAIRS = [
    # Arms
    (5, 7),   # Left shoulder -> Left elbow
    (7, 9),   # Left elbow -> Left wrist
    (6, 8),   # Right shoulder -> Right elbow
    (8, 10),  # Right elbow -> Right wrist
    # Torso
    (5, 6),   # Shoulder line
    (5, 11),  # Left shoulder -> Left hip
    (6, 12),  # Right shoulder -> Right hip
    (11, 12), # Hip line
    # Legs
    (11, 13), # Left hip -> Left knee
    (13, 15), # Left knee -> Left ankle
    (12, 14), # Right hip -> Right knee
    (14, 16), # Right knee -> Right ankle
]

# Heuristic threshold for "hand near waist" (torso lengths). Not validated on labelled data.
WAIST_REACH_NORM = 0.65

# Visual Palette (BGR)
COLOR_LEFT_ARM = (0, 255, 255)     # Yellow
COLOR_RIGHT_ARM = (255, 180, 0)    # Cyan-Blue
COLOR_TORSO = (50, 205, 50)        # Lime Green
COLOR_LEFT_LEG = (255, 105, 180)   # Hot Pink
COLOR_RIGHT_LEG = (147, 112, 219)  # Purple
COLOR_WAIST_REACH = (0, 140, 255)  # Orange reach line
COLOR_VELOCITY = (0, 0, 255)       # Red velocity arrow
COLOR_JOINT = (0, 255, 255)        # Bright Yellow joint circle


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRIC & BIOMECHANICAL ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def calculate_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """Compute 2D Euclidean distance between two points."""
    return float(math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))


def calculate_angle_3pt(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    Compute interior angle at vertex b formed by rays ba and bc.
    Returns angle in degrees [0.0, 180.0].
    """
    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]
    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    mag_ba = math.hypot(ba_x, ba_y)
    mag_bc = math.hypot(bc_x, bc_y)

    if mag_ba < 1e-4 or mag_bc < 1e-4:
        return 0.0

    dot_prod = (ba_x * bc_x) + (ba_y * bc_y)
    cosine = dot_prod / (mag_ba * mag_bc)
    # Clip to valid arccos domain to avoid NaN from floating-point drift
    cosine = max(-1.0, min(1.0, cosine))
    angle_rad = math.acos(cosine)
    return float(math.degrees(angle_rad))


def analyze_hand_to_waist_distance(
    kpts: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.3
) -> Dict[str, Any]:
    """
    Calculate hand-to-waist metrics:
    - Waist center point (midpoint of left & right hip)
    - Left and right wrist distance to waist (pixels)
    - Normalized distance relative to torso length
    """
    res = {
        "waist_center": None,
        "torso_length": None,
        "left_wrist_dist_px": None,
        "left_wrist_dist_norm": None,
        "right_wrist_dist_px": None,
        "right_wrist_dist_norm": None,
        "hands_near_waist": False,
    }

    # Hips: 11 (left), 12 (right)
    l_hip_ok = scores[11] >= kpt_thr
    r_hip_ok = scores[12] >= kpt_thr

    if l_hip_ok and r_hip_ok:
        waist_x = (kpts[11, 0] + kpts[12, 0]) / 2.0
        waist_y = (kpts[11, 1] + kpts[12, 1]) / 2.0
        res["waist_center"] = (float(waist_x), float(waist_y))
    elif l_hip_ok:
        res["waist_center"] = (float(kpts[11, 0]), float(kpts[11, 1]))
    elif r_hip_ok:
        res["waist_center"] = (float(kpts[12, 0]), float(kpts[12, 1]))

    # Compute torso reference length (shoulder center to waist center)
    l_sh_ok = scores[5] >= kpt_thr
    r_sh_ok = scores[6] >= kpt_thr
    if res["waist_center"] is not None and (l_sh_ok or r_sh_ok):
        if l_sh_ok and r_sh_ok:
            sh_center = ((kpts[5, 0] + kpts[6, 0]) / 2.0, (kpts[5, 1] + kpts[6, 1]) / 2.0)
        elif l_sh_ok:
            sh_center = (float(kpts[5, 0]), float(kpts[5, 1]))
        else:
            sh_center = (float(kpts[6, 0]), float(kpts[6, 1]))
        res["torso_length"] = calculate_euclidean_distance(sh_center, res["waist_center"])

    # Wrists: 9 (left), 10 (right)
    if res["waist_center"] is not None:
        waist = res["waist_center"]
        torso_l = res["torso_length"]

        if scores[9] >= kpt_thr:
            d_l = calculate_euclidean_distance((kpts[9, 0], kpts[9, 1]), waist)
            res["left_wrist_dist_px"] = d_l
            if torso_l and torso_l > 5.0:
                res["left_wrist_dist_norm"] = d_l / torso_l

        if scores[10] >= kpt_thr:
            d_r = calculate_euclidean_distance((kpts[10, 0], kpts[10, 1]), waist)
            res["right_wrist_dist_px"] = d_r
            if torso_l and torso_l > 5.0:
                res["right_wrist_dist_norm"] = d_r / torso_l

        # Heuristic indicator (not validated): a wrist within 0.65 torso lengths of the waist centre.
        # Only the torso-normalised distance is used — a fixed pixel radius (the old 55 px fallback) depends
        # on image resolution and camera distance. Unknown torso length -> indicator stays False.
        for dist_norm in (res["left_wrist_dist_norm"], res["right_wrist_dist_norm"]):
            if dist_norm is not None and dist_norm < WAIST_REACH_NORM:
                res["hands_near_waist"] = True

    return res


def analyze_arm_angles(
    kpts: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.3
) -> Dict[str, Optional[float]]:
    """
    Calculate interior arm angles formed at the elbows:
    - Left arm angle: Left shoulder (5) -> Left elbow (7) -> Left wrist (9)
    - Right arm angle: Right shoulder (6) -> Right elbow (8) -> Right wrist (10)
    """
    res = {
        "left_arm_angle_deg": None,
        "right_arm_angle_deg": None,
    }

    # Left arm: 5 -> 7 -> 9
    if scores[5] >= kpt_thr and scores[7] >= kpt_thr and scores[9] >= kpt_thr:
        p_sh = (float(kpts[5, 0]), float(kpts[5, 1]))
        p_el = (float(kpts[7, 0]), float(kpts[7, 1]))
        p_wr = (float(kpts[9, 0]), float(kpts[9, 1]))
        res["left_arm_angle_deg"] = calculate_angle_3pt(p_sh, p_el, p_wr)

    # Right arm: 6 -> 8 -> 10
    if scores[6] >= kpt_thr and scores[8] >= kpt_thr and scores[10] >= kpt_thr:
        p_sh = (float(kpts[6, 0]), float(kpts[6, 1]))
        p_el = (float(kpts[8, 0]), float(kpts[8, 1]))
        p_wr = (float(kpts[10, 0]), float(kpts[10, 1]))
        res["right_arm_angle_deg"] = calculate_angle_3pt(p_sh, p_el, p_wr)

    return res


def analyze_body_orientation(
    kpts: np.ndarray, scores: np.ndarray, kpt_thr: float = 0.3
) -> Dict[str, Any]:
    """
    Determine body orientation in 2D surveillance imagery:
    - Facing direction: 'Frontal', 'Profile (Facing Left)', 'Profile (Facing Right)', 'Back', or 'Angled'
    - Shoulder tilt angle relative to horizontal plane
    - Torso coronal aspect ratio (shoulder width / torso height)
    """
    res = {
        "facing_direction": "Unknown",
        "shoulder_tilt_deg": 0.0,
        "torso_tilt_deg": 0.0,
        "coronal_ratio": None,
        "confidence": 0.0,
    }

    l_sh_ok = scores[5] >= kpt_thr
    r_sh_ok = scores[6] >= kpt_thr
    l_hip_ok = scores[11] >= kpt_thr
    r_hip_ok = scores[12] >= kpt_thr

    if l_sh_ok and r_sh_ok:
        dx = float(kpts[6, 0] - kpts[5, 0])
        dy = float(kpts[6, 1] - kpts[5, 1])
        sh_width = math.hypot(dx, dy)
        tilt_deg = math.degrees(math.atan2(dy, dx))
        res["shoulder_tilt_deg"] = round(tilt_deg, 1)

        # Torso height
        if l_hip_ok and r_hip_ok:
            sh_center_y = (kpts[5, 1] + kpts[6, 1]) / 2.0
            hip_center_y = (kpts[11, 1] + kpts[12, 1]) / 2.0
            torso_h = abs(hip_center_y - sh_center_y)
            if torso_h > 10.0:
                res["coronal_ratio"] = round(sh_width / torso_h, 2)

        # Facial clues: Nose (0), Left eye (1), Right eye (2), Left ear (3), Right ear (4)
        nose_score = float(scores[0])
        l_eye_score = float(scores[1])
        r_eye_score = float(scores[2])
        face_visible = (nose_score >= kpt_thr) or (l_eye_score >= kpt_thr) or (r_eye_score >= kpt_thr)

        c_ratio = res["coronal_ratio"]
        if c_ratio is not None and c_ratio < 0.40:
            # Significant shoulder foreshortening indicates side profile
            if nose_score >= kpt_thr:
                nose_x = kpts[0, 0]
                sh_mid_x = (kpts[5, 0] + kpts[6, 0]) / 2.0
                if nose_x < sh_mid_x:
                    res["facing_direction"] = "Profile (Facing Left)"
                else:
                    res["facing_direction"] = "Profile (Facing Right)"
            else:
                res["facing_direction"] = "Side Profile"
        elif face_visible:
            if c_ratio is not None and c_ratio >= 0.70:
                res["facing_direction"] = "Frontal"
            else:
                res["facing_direction"] = "Angled Front"
        else:
            # Face not visible but shoulders and hips detected
            res["facing_direction"] = "Back / Turned Away"

        res["confidence"] = float((scores[5] + scores[6]) / 2.0)
    elif l_sh_ok and not r_sh_ok:
        res["facing_direction"] = "Profile (Facing Left)"
        res["confidence"] = float(scores[5])
    elif r_sh_ok and not l_sh_ok:
        res["facing_direction"] = "Profile (Facing Right)"
        res["confidence"] = float(scores[6])

    return res


# ─────────────────────────────────────────────────────────────────────────────
# RTMPose TRACKER & PIPELINE CLASS
# ─────────────────────────────────────────────────────────────────────────────
class RTMPosePipeline:
    """
    Unified RTMPose-m human pose extraction and telemetry pipeline.

    Encapsulates:
    1. Pretrained RTMPose-m inference (via OpenMMLab rtmlib ONNX engine)
    2. Extraction of 12 requested key joints
    3. Biometric telemetry calculation:
       - Hand-to-waist distance
       - Left & right arm elbow angles
       - Temporal wrist velocity tracking
       - Body orientation & tilt
    4. HUD visualization
    """

    def __init__(
        self,
        mode: str = "balanced",  # 'balanced' uses RTMPose-m!
        backend: str = "onnxruntime",
        device: str = "cpu",
        kpt_thr: float = 0.35,
        velocity_ema_alpha: float = 0.60,
        track_people: bool = True,
        max_gap_s: float = 0.5,
        stale_s: float = 2.0,
        load_detector: bool = True,
    ):
        """
        Initialize the RTMPose pipeline with pretrained RTMPose-m model.

        Args:
            mode: RTMPose model tier ('balanced' = RTMPose-m).
            backend: Inference engine ('onnxruntime' or 'opencv').
            device: 'cpu', 'cuda', or 'mps'.
            kpt_thr: Confidence threshold to consider a joint detected.
            velocity_ema_alpha: Smoothing factor for wrist velocity.
            track_people: Give people persistent IDs with ByteTrack (required for meaningful wrist velocity;
                without it person_id is just the detection order in each frame and velocity is not computed).
            max_gap_s: A wrist unseen for longer than this restarts its velocity estimate.
            stale_s: Per-person state is dropped after this long without a detection.
            load_detector: Load rtmlib's YOLOX person detector. Set False when people come from an external
                tracker (Phase 2 pipeline) and only process_tracked() is used — saves memory and a ~100 MB download.
        """
        self.mode = mode
        self.backend = backend
        self.device = device
        self.kpt_thr = kpt_thr
        self.velocity_ema_alpha = velocity_ema_alpha

        print(f"[RTMPosePipeline] Initializing RTMPose-m (mode='{mode}', backend='{backend}', device='{device}')...")
        try:
            import rtmlib
            if load_detector:
                self.model = rtmlib.Body(
                    mode=mode,
                    backend=backend,
                    device=device,
                    to_openpose=False,
                )
            else:  # pose model only; process_tracked() supplies the person boxes
                from types import SimpleNamespace
                cfg = rtmlib.Body.MODE[mode]
                self.model = SimpleNamespace(det_model=None, pose_model=rtmlib.RTMPose(
                    cfg["pose"], model_input_size=cfg["pose_input_size"], backend=backend, device=device))
                track_people = False
            self._ready = True
            print("[RTMPosePipeline] Pretrained RTMPose-m loaded successfully.")
        except Exception as e:
            print(f"[RTMPosePipeline] Warning: Could not initialize RTMPose model: {e}")
            self.model = None
            self._ready = False

        # Per-track state for inter-frame velocity: {track_id: {"t": last time, "l"/"r": (x, y, t), "ema_l"/"ema_r": px/s}}
        self.max_gap_s = max_gap_s
        self.stale_s = stale_s
        self.track_state: Dict[int, Dict[str, Any]] = {}
        self.tracker = None
        if track_people:
            from tracking.tracker import ByteTrackWrapper
            self.tracker = ByteTrackWrapper()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def reset_tracking(self):
        """Reset inter-frame tracking state (e.g. at start of new video)."""
        self.track_state.clear()
        if self.tracker is not None:
            self.tracker.reset()

    def extract_target_joints(self, kpts: np.ndarray, scores: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Extract the 12 target joints requested:
        wrists, elbows, shoulders, hips, knees, ankles.
        """
        joints = {}
        for name, idx in TARGET_JOINTS.items():
            conf = float(scores[idx])
            detected = conf >= self.kpt_thr
            joints[name] = {
                "idx": idx,
                "x": float(kpts[idx, 0]),
                "y": float(kpts[idx, 1]),
                "confidence": conf,
                "detected": detected,
            }
        return joints

    def compute_wrist_velocity(
        self,
        person_idx: int,
        kpts: np.ndarray,
        scores: np.ndarray,
        current_time: Optional[float] = None,
        id_persistent: bool = True,
        torso_length: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Wrist speed between observations of the SAME tracked person (px/s, EMA-smoothed), plus a scale-free
        version in torso lengths per second.

        `person_idx` must be a persistent track ID. The previous implementation keyed this state by the
        person's position in each frame's detection list, so with two or more people the "velocity" was often
        measured between different people. When IDs are not persistent, velocity is reported as None.
        """
        if current_time is None:
            current_time = time.time()
        res: Dict[str, Any] = {
            "left_wrist_speed_px_s": None, "right_wrist_speed_px_s": None,
            "left_wrist_speed_norm": None, "right_wrist_speed_norm": None,
            "left_wrist_disp_px": None, "right_wrist_disp_px": None,
            "left_wrist_vector": None, "right_wrist_vector": None,
            "id_persistent": id_persistent,
        }
        if not id_persistent:
            return res
        st = self.track_state.setdefault(person_idx, {})
        for side, idx in (("left", 9), ("right", 10)):
            key = side[0]
            if scores[idx] < self.kpt_thr:
                continue  # unseen this frame: keep the last observation, no new estimate
            x, y = float(kpts[idx, 0]), float(kpts[idx, 1])
            prev = st.get(key)
            if prev is not None and 0.0 < current_time - prev[2] <= self.max_gap_s:
                dt = current_time - prev[2]
                dx, dy = x - prev[0], y - prev[1]
                speed = math.hypot(dx, dy) / dt
                ema = st.get("ema_" + key)
                speed = speed if ema is None else self.velocity_ema_alpha * speed + (1.0 - self.velocity_ema_alpha) * ema
                st["ema_" + key] = speed
                res[f"{side}_wrist_disp_px"] = round(math.hypot(dx, dy), 1)
                res[f"{side}_wrist_speed_px_s"] = round(speed, 1)
                res[f"{side}_wrist_vector"] = (round(dx, 1), round(dy, 1))
                if torso_length and torso_length > 5.0:
                    res[f"{side}_wrist_speed_norm"] = round(speed / torso_length, 3)
            else:
                st.pop("ema_" + key, None)  # first sighting or long gap: restart the estimate
            st[key] = (x, y, current_time)
        st["t"] = current_time
        return res

    def _purge_stale(self, now: float) -> None:
        for tid in [t for t, st in self.track_state.items() if now - st.get("t", now) > self.stale_s]:
            del self.track_state[tid]

    def _telemetry(self, pid: int, kpts: np.ndarray, scs: np.ndarray, t: float, id_persistent: bool,
                   bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        hand_waist = analyze_hand_to_waist_distance(kpts, scs, self.kpt_thr)
        valid_mask = scs >= self.kpt_thr
        if bbox is None:
            if np.any(valid_mask):
                v = kpts[valid_mask]
                bbox = [int(v[:, 0].min()), int(v[:, 1].min()), int(v[:, 0].max()), int(v[:, 1].max())]
            else:
                bbox = [0, 0, 0, 0]
        return {
            "person_id": pid,
            "id_persistent": id_persistent,
            "bbox": [int(b) for b in bbox],
            "keypoints_raw": kpts,
            "scores_raw": scs,
            "joints": self.extract_target_joints(kpts, scs),
            "hand_to_waist": hand_waist,
            "arm_angles": analyze_arm_angles(kpts, scs, self.kpt_thr),
            "wrist_velocity": self.compute_wrist_velocity(pid, kpts, scs, t, id_persistent, hand_waist["torso_length"]),
            "body_orientation": analyze_body_orientation(kpts, scs, self.kpt_thr),
        }

    def process_tracked(
        self,
        img_bgr: np.ndarray,
        tracks: List[Dict[str, Any]],
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Top-down pose for people that are already tracked (Phase 2): tracks = [{"track_id": int, "bbox": [x1, y1, x2, y2]}].
        Telemetry is keyed by track_id, so wrist velocity follows the same person across frames.
        """
        if not self._ready or self.model is None or img_bgr is None:
            return []
        t = timestamp if timestamp is not None else time.time()
        self._purge_stale(t)
        if not tracks:  # rtmlib would otherwise run the pose model on the whole image
            return []
        try:
            keypoints_all, scores_all = self.model.pose_model(img_bgr, bboxes=[tr["bbox"] for tr in tracks])
        except Exception as e:
            print(f"[RTMPosePipeline] Inference error: {e}")
            return []
        return [self._telemetry(tr["track_id"], keypoints_all[i], scores_all[i], t, True, tr["bbox"])
                for i, tr in enumerate(tracks)]

    def process_frame(
        self,
        img_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process a single image or video frame:
        1. Detect people (RTMPose's YOLOX detector) and, if tracking is enabled, give them persistent IDs (ByteTrack)
        2. Run RTMPose-m on each person box
        3. Compute hand-to-waist distance, arm angles, wrist velocity (tracked people only) and body orientation

        Returns list of telemetry dictionaries (one per detected person). `person_id` is the track ID when
        tracking is enabled (`id_persistent` True); otherwise it is just the detection order in this frame.
        """
        if not self._ready or self.model is None or img_bgr is None:
            return []
        t = timestamp if timestamp is not None else time.time()
        if getattr(self.model, "det_model", None) is None:
            print("[RTMPosePipeline] No person detector loaded (load_detector=False): use process_tracked().")
            return []
        if self.tracker is not None:
            try:
                bboxes = np.asarray(self.model.det_model(img_bgr), dtype=np.float32).reshape(-1, 4)
            except Exception as e:
                print(f"[RTMPosePipeline] Detection error: {e}")
                return []
            tracks = self.tracker.update(bboxes, np.ones(len(bboxes), dtype=np.float32))
            return self.process_tracked(img_bgr, tracks, t)
        try:
            keypoints_all, scores_all = self.model(img_bgr)
        except Exception as e:
            print(f"[RTMPosePipeline] Inference error: {e}")
            return []
        if keypoints_all is None or len(keypoints_all) == 0:
            return []
        return [self._telemetry(i, keypoints_all[i], scores_all[i], t, False) for i in range(len(keypoints_all))]

    def annotate(
        self,
        img: np.ndarray,
        telemetry_list: List[Dict[str, Any]],
        draw_angles: bool = True,
        draw_waist_reach: bool = True,
        draw_velocities: bool = True,
        draw_orientation: bool = True,
        show_hud_card: bool = True,
    ) -> np.ndarray:
        """
        Draw high-visibility HUD overlay onto the image:
        - Colored skeleton bones & joint nodes
        - Arm angles at elbows
        - Dotted reach lines from wrists to waist
        - Wrist velocity arrows & speed labels
        - Posture & orientation HUD badge
        """
        annotated = img.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Official OpenMMLab / RTMPose Complete 17-Keypoint Skeleton
        try:
            import rtmlib
            all_kpts = np.array([p["keypoints_raw"] for p in telemetry_list])
            all_scs = np.array([p["scores_raw"] for p in telemetry_list])
            annotated = rtmlib.draw_skeleton(
                annotated, all_kpts, all_scs,
                kpt_thr=self.kpt_thr, line_width=3, radius=5
            )
        except Exception:
            # Fallback to internal custom bone drawer
            for person in telemetry_list:
                kpts = person["keypoints_raw"]
                scs = person["scores_raw"]
                for idx1, idx2 in SKELETON_PAIRS:
                    if scs[idx1] >= self.kpt_thr and scs[idx2] >= self.kpt_thr:
                        pt1 = (int(kpts[idx1, 0]), int(kpts[idx1, 1]))
                        pt2 = (int(kpts[idx2, 0]), int(kpts[idx2, 1]))
                        cv2.line(annotated, pt1, pt2, (0, 255, 255), 3, cv2.LINE_AA)

        for person in telemetry_list:
            kpts = person["keypoints_raw"]
            scs = person["scores_raw"]
            p_id = person["person_id"]

            # 2. Draw Target Joint Nodes (Wrists, Elbows, Shoulders, Hips, Knees, Ankles)
            for j_name, j_data in person["joints"].items():
                if j_data["detected"]:
                    px, py = int(j_data["x"]), int(j_data["y"])
                    # Special styling for wrists (key threat indicator)
                    if "wrist" in j_name:
                        cv2.circle(annotated, (px, py), 7, (0, 0, 255), -1, cv2.LINE_AA)
                        cv2.circle(annotated, (px, py), 9, (255, 255, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.circle(annotated, (px, py), 5, COLOR_JOINT, -1, cv2.LINE_AA)
                        cv2.circle(annotated, (px, py), 6, (0, 0, 0), 1, cv2.LINE_AA)

            # 3. Draw Hand-to-Waist Reach Line
            if draw_waist_reach:
                hw = person["hand_to_waist"]
                waist = hw["waist_center"]
                if waist is not None:
                    wx, wy = int(waist[0]), int(waist[1])
                    cv2.circle(annotated, (wx, wy), 5, (0, 165, 255), -1, cv2.LINE_AA)

                    # Line to left wrist
                    if hw["left_wrist_dist_px"] is not None and scs[9] >= self.kpt_thr:
                        lx, ly = int(kpts[9, 0]), int(kpts[9, 1])
                        cv2.line(annotated, (lx, ly), (wx, wy), COLOR_WAIST_REACH, 1, cv2.LINE_AA)
                        dist_lbl = f"L: {hw['left_wrist_dist_px']:.0f}px"
                        mid_pt = ((lx + wx) // 2, (ly + wy) // 2)
                        cv2.putText(annotated, dist_lbl, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                    # Line to right wrist
                    if hw["right_wrist_dist_px"] is not None and scs[10] >= self.kpt_thr:
                        rx, ry = int(kpts[10, 0]), int(kpts[10, 1])
                        cv2.line(annotated, (rx, ry), (wx, wy), COLOR_WAIST_REACH, 1, cv2.LINE_AA)
                        dist_lbl = f"R: {hw['right_wrist_dist_px']:.0f}px"
                        mid_pt = ((rx + wx) // 2, (ry + wy) // 2)
                        cv2.putText(annotated, dist_lbl, mid_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # 4. Draw Arm Angles (Elbow interior angle)
            if draw_angles:
                angles = person["arm_angles"]
                # Left elbow angle badge
                if angles["left_arm_angle_deg"] is not None:
                    ex, ey = int(kpts[7, 0]), int(kpts[7, 1])
                    ang_txt = f"{angles['left_arm_angle_deg']:.0f}°"
                    cv2.putText(annotated, ang_txt, (ex - 28, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

                # Right elbow angle badge
                if angles["right_arm_angle_deg"] is not None:
                    ex, ey = int(kpts[8, 0]), int(kpts[8, 1])
                    ang_txt = f"{angles['right_arm_angle_deg']:.0f}°"
                    cv2.putText(annotated, ang_txt, (ex + 10, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2, cv2.LINE_AA)

            # 5. Draw Wrist Velocity Indicators
            if draw_velocities:
                vel = person["wrist_velocity"]
                if (vel["left_wrist_speed_px_s"] or 0.0) > 25.0 and vel["left_wrist_vector"] and scs[9] >= self.kpt_thr:
                    lx, ly = int(kpts[9, 0]), int(kpts[9, 1])
                    vx, vy = int(vel["left_wrist_vector"][0]), int(vel["left_wrist_vector"][1])
                    cv2.arrowedLine(annotated, (lx, ly), (lx + vx * 2, ly + vy * 2), COLOR_VELOCITY, 2, cv2.LINE_AA)
                    cv2.putText(annotated, f"{vel['left_wrist_speed_px_s']:.0f}px/s", (lx - 30, ly + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

                if (vel["right_wrist_speed_px_s"] or 0.0) > 25.0 and vel["right_wrist_vector"] and scs[10] >= self.kpt_thr:
                    rx, ry = int(kpts[10, 0]), int(kpts[10, 1])
                    vx, vy = int(vel["right_wrist_vector"][0]), int(vel["right_wrist_vector"][1])
                    cv2.arrowedLine(annotated, (rx, ry), (rx + vx * 2, ry + vy * 2), COLOR_VELOCITY, 2, cv2.LINE_AA)
                    cv2.putText(annotated, f"{vel['right_wrist_speed_px_s']:.0f}px/s", (rx + 10, ry + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            # 6. Draw Body Orientation Badge
            if draw_orientation:
                orient = person["body_orientation"]
                bbox = person["bbox"]
                if bbox[3] > bbox[1]:
                    bx = max(10, bbox[0])
                    by = max(25, bbox[1] - 8)
                    facing_str = f"P{p_id}: {orient['facing_direction']} (Tilt: {orient['shoulder_tilt_deg']}°)"
                    cv2.putText(annotated, facing_str, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 127), 2, cv2.LINE_AA)

        # 7. Overall Corner Biometric HUD Card
        if show_hud_card and len(telemetry_list) > 0:
            p0 = telemetry_list[0]
            orient = p0["body_orientation"]
            angles = p0["arm_angles"]
            vel = p0["wrist_velocity"]
            hw = p0["hand_to_waist"]

            # Transparent background overlay
            card_h, card_w = 120, 310
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (15, 23, 42), -1)
            cv2.rectangle(overlay, (10, 10), (10 + card_w, 10 + card_h), (56, 189, 248), 1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            cv2.putText(annotated, "POSE TELEMETRY (RTMPose-m)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)

            orient_text = f"Facing: {orient['facing_direction']} | Tilt: {orient['shoulder_tilt_deg']}°"
            cv2.putText(annotated, orient_text, (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

            l_ang = f"{angles['left_arm_angle_deg']:.0f}°" if angles['left_arm_angle_deg'] is not None else "N/A"
            r_ang = f"{angles['right_arm_angle_deg']:.0f}°" if angles['right_arm_angle_deg'] is not None else "N/A"
            angles_text = f"Elbow Angles: L {l_ang} | R {r_ang}"
            cv2.putText(annotated, angles_text, (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

            hw_l = f"{hw['left_wrist_dist_px']:.0f}px" if hw['left_wrist_dist_px'] is not None else "N/A"
            hw_r = f"{hw['right_wrist_dist_px']:.0f}px" if hw['right_wrist_dist_px'] is not None else "N/A"
            hw_text = f"Hand-to-Waist: L {hw_l} | R {hw_r}"
            cv2.putText(annotated, hw_text, (20, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

            vel_l = f"{vel['left_wrist_speed_px_s']:.0f}" if vel['left_wrist_speed_px_s'] is not None else "--"
            vel_r = f"{vel['right_wrist_speed_px_s']:.0f}" if vel['right_wrist_speed_px_s'] is not None else "--"
            vel_text = f"Wrist Velocity: L {vel_l} | R {vel_r} px/s"
            cv2.putText(annotated, vel_text, (20, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

        return annotated


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE CLI TEST & DEMO
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SWORD v2 — RTMPose-m Human Pose Extraction")
    parser.add_argument("--source", type=str, default=None, help="Path to input image or video file")
    parser.add_argument("--webcam", action="store_true", help="Run live webcam pose extraction")
    parser.add_argument("--save", type=str, default=None, help="Path to save annotated output image/video")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu', 'cuda', or 'mps'")
    args = parser.parse_args()

    pipeline = RTMPosePipeline(mode="balanced", device=args.device)

    if not pipeline.is_ready:
        print("Failed to initialize RTMPosePipeline. Please check dependencies.")
        sys.exit(1)

    if args.webcam:
        print("[Live Webcam] Starting RTMPose-m pose extraction... Press 'q' to exit.")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            telemetry = pipeline.process_frame(frame)
            annotated = pipeline.annotate(frame, telemetry)
            cv2.imshow("SWORD v2 — RTMPose-m Live Pose Telemetry", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif args.source:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: Source file '{args.source}' does not exist.")
            sys.exit(1)

        # Check if image or video
        is_video = source_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]
        if is_video:
            cap = cv2.VideoCapture(str(source_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            out_writer = None
            if args.save:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))

            print(f"Processing video: {source_path.name}...")
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                telemetry = pipeline.process_frame(frame, timestamp=frame_idx / fps)
                ann = pipeline.annotate(frame, telemetry)
                if out_writer:
                    out_writer.write(ann)
                frame_idx += 1
            cap.release()
            if out_writer:
                out_writer.release()
            print(f"Finished processing {frame_idx} video frames.")
        else:
            img = cv2.imread(str(source_path))
            if img is None:
                print(f"Error reading image: {source_path}")
                sys.exit(1)
            telemetry = pipeline.process_frame(img)
            annotated = pipeline.annotate(img, telemetry)
            if args.save:
                cv2.imwrite(args.save, annotated)
                print(f"Saved annotated output to: {args.save}")
            else:
                out_path = source_path.parent / f"{source_path.stem}_pose_annotated.jpg"
                cv2.imwrite(str(out_path), annotated)
                print(f"Saved output to: {out_path}")

            # Print telemetry summary to stdout
            print(f"\n--- Extracted Pose Telemetry ({len(telemetry)} people detected) ---")
            for person in telemetry:
                print(f"Person #{person['person_id']}:")
                print(f"  Facing: {person['body_orientation']['facing_direction']}")
                print(f"  Shoulder Tilt: {person['body_orientation']['shoulder_tilt_deg']}°")
                print(f"  Left Arm Angle: {person['arm_angles']['left_arm_angle_deg']}°")
                print(f"  Right Arm Angle: {person['arm_angles']['right_arm_angle_deg']}°")
                print(f"  Hand-to-Waist (L): {person['hand_to_waist']['left_wrist_dist_px']} px")
                print(f"  Hand-to-Waist (R): {person['hand_to_waist']['right_wrist_dist_px']} px")
                print(f"  Hands near waist alert: {person['hand_to_waist']['hands_near_waist']}")
    else:
        print("Running quick synthetic test...")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        telemetry = pipeline.process_frame(blank)
        ann = pipeline.annotate(blank, telemetry)
        print(f"Test complete. Output image shape: {ann.shape}. Telemetry count: {len(telemetry)}")


if __name__ == "__main__":
    main()

