"""
S.W.O.R.D. Phase 4 - Optional Lightweight Fallback Head
Extracts rule-based kinematic features directly from skeleton keypoints:
  1. Hand-to-waist distance (normalized by torso length)
  2. Wrist velocity and acceleration (indicative of sudden strikes or drawing weapons)
  3. Arm angle (shoulder-elbow-wrist joint angles)
  4. Torso orientation and inclination

Trains a lightweight Logistic Regression head combining rule features + PoseC3D probabilities.
Executes on CPU in seconds. Kept strictly separated from the pretrained-only path.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


def extract_rule_features_from_clip(
    keypoints: np.ndarray,      # [M, T, 17, 2] or [T, 17, 2]
    scores: np.ndarray = None,  # [M, T, 17] or [T, 17]
    fps: float = 25.0
) -> np.ndarray:
    """
    Extracts physically grounded kinematic and geometric features from a skeleton clip.
    COCO-17 indices:
      5: L Shoulder, 6: R Shoulder, 7: L Elbow, 8: R Elbow,
      9: L Wrist, 10: R Wrist, 11: L Hip, 12: R Hip
    """
    if keypoints.ndim == 4:
        kp = keypoints[0]  # Take primary person [T, 17, 2]
    else:
        kp = keypoints

    T = kp.shape[0]

    l_shoulder = kp[:, 5, :]
    r_shoulder = kp[:, 6, :]
    l_elbow = kp[:, 7, :]
    r_elbow = kp[:, 8, :]
    l_wrist = kp[:, 9, :]
    r_wrist = kp[:, 10, :]
    l_hip = kp[:, 11, :]
    r_hip = kp[:, 12, :]

    # Torso length normalization scale (shoulder midpoint to hip midpoint)
    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    hip_mid = (l_hip + r_hip) / 2.0
    torso_len = np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    mean_torso = float(np.mean(torso_len)) if np.mean(torso_len) > 1e-3 else 1.0

    # 1. Hand-to-Waist Distance (normalized by torso length)
    dist_l_wrist_waist = np.linalg.norm(l_wrist - hip_mid, axis=1) / mean_torso
    dist_r_wrist_waist = np.linalg.norm(r_wrist - hip_mid, axis=1) / mean_torso

    f_hand_waist = [
        float(np.min(dist_l_wrist_waist)), float(np.mean(dist_l_wrist_waist)), float(np.max(dist_l_wrist_waist)),
        float(np.min(dist_r_wrist_waist)), float(np.mean(dist_r_wrist_waist)), float(np.max(dist_r_wrist_waist)),
    ]

    # 2. Wrist Velocities (pixels/sec normalized by torso length)
    dt = 1.0 / fps
    v_l_wrist = np.linalg.norm(np.diff(l_wrist, axis=0), axis=1) / (mean_torso * dt)
    v_r_wrist = np.linalg.norm(np.diff(r_wrist, axis=0), axis=1) / (mean_torso * dt)

    f_wrist_vel = [
        float(np.max(v_l_wrist)) if len(v_l_wrist) > 0 else 0.0,
        float(np.mean(v_l_wrist)) if len(v_l_wrist) > 0 else 0.0,
        float(np.max(v_r_wrist)) if len(v_r_wrist) > 0 else 0.0,
        float(np.mean(v_r_wrist)) if len(v_r_wrist) > 0 else 0.0,
    ]

    # 3. Arm Angle at Elbow (degrees)
    def compute_joint_angle(p1, p2, p3):
        # Angle at p2 between p1-p2 and p3-p2
        v1 = p1 - p2
        v2 = p3 - p2
        cos_ang = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
        cos_ang = np.clip(cos_ang, -1.0, 1.0)
        return np.degrees(np.arccos(cos_ang))

    l_arm_angle = compute_joint_angle(l_shoulder, l_elbow, l_wrist)
    r_arm_angle = compute_joint_angle(r_shoulder, r_elbow, r_wrist)

    f_arm_angles = [
        float(np.min(l_arm_angle)), float(np.mean(l_arm_angle)), float(np.max(l_arm_angle)),
        float(np.min(r_arm_angle)), float(np.mean(r_arm_angle)), float(np.max(r_arm_angle)),
    ]

    # 4. Torso Inclination (angle relative to vertical y-axis)
    torso_vec = shoulder_mid - hip_mid  # [T, 2]
    vertical = np.array([0.0, -1.0])    # In image coords, up is negative y
    torso_dot = np.sum(torso_vec * vertical, axis=1) / (np.linalg.norm(torso_vec, axis=1) + 1e-6)
    torso_angle = np.degrees(np.arccos(np.clip(torso_dot, -1.0, 1.0)))

    f_torso = [
        float(np.min(torso_angle)), float(np.mean(torso_angle)), float(np.max(torso_angle))
    ]

    feature_vector = np.array(f_hand_waist + f_wrist_vel + f_arm_angles + f_torso, dtype=np.float32)
    return feature_vector


def train_logistic_head(
    clips: List[Dict[str, Any]],
    labels: List[int],
    posec3d_probs: Optional[np.ndarray] = None,
    output_model_path: str = "checkpoints/fallback_logistic_head.pkl"
):
    """
    Trains a tiny Logistic Regression model on CPU in seconds.
    Features: Rule kinematics (19 dims) + Optional PoseC3D probabilities (120 dims).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_list = []
    for i, clip in enumerate(clips):
        rule_feat = extract_rule_features_from_clip(clip["keypoint"], clip.get("keypoint_score"))
        if posec3d_probs is not None:
            full_feat = np.concatenate([rule_feat, posec3d_probs[i]])
        else:
            full_feat = rule_feat
        X_list.append(full_feat)

    X = np.array(X_list)
    y = np.array(labels)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(max_iter=500, C=1.0, class_weight="balanced"))
    ])
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"[FALLBACK] Logistic head trained successfully on {len(X)} samples. Train Acc: {train_acc*100:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(output_model_path)), exist_ok=True)
    with open(output_model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[FALLBACK] Model saved to {output_model_path}")
    return clf


def predict_fallback(
    model_path: str,
    clip: Dict[str, Any],
    posec3d_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    rule_feat = extract_rule_features_from_clip(clip["keypoint"], clip.get("keypoint_score"))
    if posec3d_prob is not None:
        feat = np.concatenate([rule_feat, posec3d_prob]).reshape(1, -1)
    else:
        feat = rule_feat.reshape(1, -1)

    probs = clf.predict_proba(feat)[0]
    return {"pred_class": int(clf.predict(feat)[0]), "probabilities": [round(float(p), 4) for p in probs]}


def main():
    parser = argparse.ArgumentParser(description="Rule-based pose feature extractor & lightweight fallback classifier.")
    parser.add_argument("--demo", action="store_true", help="Run self-test demo on synthetic poses")
    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        print("=== S.W.O.R.D. Phase 4 - Lightweight Fallback Head (Self-Test) ===")
        # Synthetic clip: 48 frames, 17 keypoints
        syn_kps = np.random.uniform(100, 300, (1, 48, 17, 2)).astype(np.float32)
        features = extract_rule_features_from_clip(syn_kps)
        print(f"Extracted {len(features)} kinematic features:")
        print("  - Hand-to-waist features (6):", [round(float(x), 2) for x in features[:6]])
        print("  - Wrist velocity features (4):", [round(float(x), 2) for x in features[6:10]])
        print("  - Arm angle features (6):     ", [round(float(x), 2) for x in features[10:16]])
        print("  - Torso inclination (3):      ", [round(float(x), 2) for x in features[16:19]])
        print("[SUCCESS] Feature extractor verified.")


if __name__ == "__main__":
    main()
