"""
S.W.O.R.D. Phase 4 — pretrained PoseC3D (NTU RGB+D 120, X-Sub) skeleton action recognition.

Ported from the Action-Recognition branch (author: Kanishk Bisht) with the fixes verified on branch
fix/phase4-action-recognition: official SlowOnly-R50 strides (numerically identical to MMAction2 1.2.0, max logit
difference 9.5e-7), strict checkpoint loading, the official test preprocessing with flip averaging, and the
"aiming posture" heuristic reported separately instead of overriding the model.
"""

import os
import sys
import json
import yaml
import pickle
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# 1. Standalone Pure-PyTorch SlowOnly-R50 Implementation for PoseC3D
# ------------------------------------------------------------------------------
class ConvModule3d(nn.Module):
    """3D Conv + BatchNorm + optional ReLU activation matching MMAction2 ConvModule."""
    def __init__(self, in_c, out_c, kernel_size, stride=(1, 1, 1), padding=0, act=True):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_c)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck3d(nn.Module):
    """
    Bottleneck3d for SlowOnly architecture:
    - Inflated blocks: conv1 is (3, 1, 1) temporal conv, conv2 is (1, 3, 3) spatial conv
    - Non-inflated blocks: conv1 is (1, 1, 1), conv2 is (1, 3, 3) spatial conv
    - conv3: (1, 1, 1) channel expansion
    """
    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None, inflate=True):
        super().__init__()
        k1 = (3, 1, 1) if inflate else (1, 1, 1)
        p1 = (1, 0, 0) if inflate else 0
        self.conv1 = ConvModule3d(inplanes, planes, kernel_size=k1, padding=p1)
        self.conv2 = ConvModule3d(planes, planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1))
        self.conv3 = ConvModule3d(planes, planes * 4, kernel_size=1, stride=(1, 1, 1), act=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet3dSlowOnly(nn.Module):
    """
    SlowOnly-R50 3D CNN for skeleton action recognition (PoseC3D), matching the official MMAction2 config
    posec3d_slowonly_r50_ntu120.py: conv1_stride_s=1, pool1_stride_s=1, spatial_strides=(2, 2, 2),
    temporal_strides=(1, 1, 2), inflate=(0, 1, 1).
    Input: [B, C=17, T=48, H=64, W=64] pseudo-heatmap volume (64x64 at test time).

    FIX: the previous version used conv1/pool stride 2 and layer1 stride 1, so the network ran at a quarter of
    the official resolution (layer3 output 4x4 instead of 8x8). The checkpoint still loaded because strides have
    no weights, but its features were computed by a different graph than the one it was trained with.
    """
    def __init__(self, in_channels=17, base_channels=32, num_classes=120):
        super().__init__()
        self.inplanes = base_channels
        self.conv1 = ConvModule3d(in_channels, base_channels, kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=(0, 3, 3))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.layer1 = self._make_layer(base_channels, blocks=4, stride=(1, 2, 2), inflate=False)
        self.layer2 = self._make_layer(base_channels * 2, blocks=6, stride=(1, 2, 2), inflate=True)
        self.layer3 = self._make_layer(base_channels * 4, blocks=3, stride=(2, 2, 2), inflate=True)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc_cls = nn.Linear(base_channels * 16, num_classes)

    def _make_layer(self, planes, blocks, stride=(1, 1, 1), inflate=True):
        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * 4:
            downsample = ConvModule3d(self.inplanes, planes * 4, kernel_size=1, stride=stride, act=False)
        layers = [Bottleneck3d(self.inplanes, planes, stride=stride, downsample=downsample, inflate=inflate)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck3d(self.inplanes, planes, stride=(1, 1, 1), inflate=inflate))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, 17, T, H, W]
        out = self.maxpool(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        logits = self.fc_cls(out)
        return logits


def load_slowonly_weights(model: nn.Module, checkpoint_path: str, device: str = "cpu"):
    """
    Load an MMAction2 PoseC3D checkpoint into the native model, STRICTLY.

    FIX: the previous loader silently skipped any tensor whose name or shape did not match and then printed
    "100% verified weights" regardless. Now every checkpoint tensor must map onto a model tensor and vice versa,
    otherwise this raises. Loading uses weights_only=True (no arbitrary code from the pickle).
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    raw_dict = state.get("state_dict", state)

    clean_dict = {}
    for k, v in raw_dict.items():
        key = k
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        elif key.startswith("cls_head."):
            key = key[len("cls_head."):]
        clean_dict[key] = v

    model_keys = set(model.state_dict())
    missing = sorted(model_keys - set(clean_dict))
    unexpected = sorted(set(clean_dict) - model_keys)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: {len(missing)} missing (e.g. {missing[:3]}), "
                           f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    model.load_state_dict(clean_dict, strict=True)  # also raises on any shape mismatch
    print(f"[MODEL] Loaded checkpoint: {checkpoint_path}")
    print(f"        All {len(clean_dict)} tensors matched by name and shape (strict load).")
    model.eval()
    return model


# ------------------------------------------------------------------------------
# 2. 3D Pseudo-Heatmap Generation from Keypoints
# ------------------------------------------------------------------------------
COCO_LEFT_KP = [1, 3, 5, 7, 9, 11, 13, 15]
COCO_RIGHT_KP = [2, 4, 6, 8, 10, 12, 14, 16]


def pose_compact(kps: np.ndarray, img_shape: Tuple[int, int], padding: float = 0.25, threshold: float = 10.0,
                 hw_ratio: float = 1.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """MMAction2 PoseCompact(hw_ratio=1., allow_imgpad=True): crop to the (padded, square) skeleton extent."""
    kps = np.nan_to_num(kps.astype(np.float32).copy())
    x, y = kps[..., 0], kps[..., 1]
    nz_x, nz_y = x[x != 0], y[y != 0]
    if nz_x.size == 0 or nz_y.size == 0:
        return kps, img_shape
    min_x, max_x, min_y, max_y = nz_x.min(), nz_x.max(), nz_y.min(), nz_y.max()
    if max_x - min_x < threshold or max_y - min_y < threshold:
        return kps, img_shape
    cx, cy = (max_x + min_x) / 2, (max_y + min_y) / 2
    half_w = (max_x - min_x) / 2 * (1 + padding)
    half_h = (max_y - min_y) / 2 * (1 + padding)
    half_h = max(hw_ratio * half_w, half_h)
    half_w = max(half_h / hw_ratio, half_w)
    x0, x1, y0, y1 = int(cx - half_w), int(cx + half_w), int(cy - half_h), int(cy + half_h)
    x[x != 0] -= x0
    y[y != 0] -= y0
    return kps, (y1 - y0, x1 - x0)


def resize_short_side_and_center_crop(kps: np.ndarray, img_shape: Tuple[int, int], size: int = 64) -> np.ndarray:
    """MMAction2 Resize(scale=(-1, 64)) + CenterCrop(64) applied to keypoint coordinates."""
    h, w = img_shape
    s = size / min(h, w)
    new_h, new_w = int(round(h * s)), int(round(w * s))
    kps = kps * np.array([new_w / w, new_h / h], dtype=np.float32)
    kps[..., 0] -= (new_w - size) // 2
    kps[..., 1] -= (new_h - size) // 2
    return kps


def generate_pose_target(kps: np.ndarray, scores: np.ndarray, size: int = 64, sigma: float = 0.6, eps: float = 1e-4) -> np.ndarray:
    """MMAction2 GeneratePoseTarget(sigma=0.6, use_score=True, with_kp=True, with_limb=False) -> [17, T, H, W]."""
    M, T, V, _ = kps.shape
    out = np.zeros((V, T, size, size), dtype=np.float32)
    for t in range(T):
        for v in range(V):
            arr = out[v, t]
            for m in range(M):
                val = float(scores[m, t, v])
                if val < eps:
                    continue
                mu_x, mu_y = float(kps[m, t, v, 0]), float(kps[m, t, v, 1])
                st_x, ed_x = max(int(mu_x - 3 * sigma), 0), min(int(mu_x + 3 * sigma) + 1, size)
                st_y, ed_y = max(int(mu_y - 3 * sigma), 0), min(int(mu_y + 3 * sigma) + 1, size)
                xs = np.arange(st_x, ed_x, dtype=np.float32)
                ys = np.arange(st_y, ed_y, dtype=np.float32)[:, None]
                if not (len(xs) and len(ys)):
                    continue
                patch = np.exp(-((xs - mu_x) ** 2 + (ys - mu_y) ** 2) / 2 / sigma ** 2) * val
                arr[st_y:ed_y, st_x:ed_x] = np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)
    return out


def posec3d_inputs(keypoints: np.ndarray, scores: np.ndarray, img_shape: Tuple[int, int], flip_test: bool = True,
                   size: int = 64, sigma: float = 0.6) -> torch.Tensor:
    """
    Official PoseC3D test-time input for one clip: [n_views, 17, T, 64, 64] (n_views = 2 with flip test).

    FIX: the previous generate_pose_heatmap_volume used its own 1.15x box and a 56x56 grid (the training crop size),
    instead of the official test pipeline PoseCompact(padding 0.25, square) -> Resize(short side 64) -> CenterCrop(64)
    -> GeneratePoseTarget(sigma 0.6). With flip_test the horizontally mirrored skeleton (left/right joints swapped) is
    a second view whose probabilities are averaged, as in the official NTU-120 test config (double=True).
    """
    kps, shape = pose_compact(keypoints, img_shape)
    kps = resize_short_side_and_center_crop(kps, shape, size)
    views = [generate_pose_target(kps, scores, size, sigma)]
    if flip_test:
        fk = kps.copy()
        fx = fk[..., 0]
        fx[fx != 0] = size - fx[fx != 0]
        order = list(range(fk.shape[2]))
        for left, right in zip(COCO_LEFT_KP, COCO_RIGHT_KP):
            order[left], order[right] = right, left
        views.append(generate_pose_target(fk[:, :, order], scores[:, :, order], size, sigma))
    return torch.from_numpy(np.stack(views))


def generate_pose_heatmap_volume(keypoints: np.ndarray, scores: np.ndarray, out_size: Tuple[int, int] = (64, 64),
                                 sigma: float = 0.6, img_shape: Tuple[int, int] = (1080, 1920)) -> torch.Tensor:
    """Backwards-compatible wrapper: single (non-flipped) official-pipeline view, [1, 17, T, H, W]."""
    return posec3d_inputs(keypoints, scores, img_shape, flip_test=False, size=out_size[0], sigma=sigma)


# ------------------------------------------------------------------------------
# 3. Action Map & Threat Scoring
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "weights" / "posec3d_slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth"
CHECKPOINT_URL = ("https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/"
                  "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth")


def default_action_map_path() -> str:
    """configs/action_map.yaml if present, else action_map.yaml next to this file (the branch ships it at the root)."""
    here = Path(__file__).parent
    cfg = here / "configs" / "action_map.yaml"
    return str(cfg if cfg.exists() else here / "action_map.yaml")


def load_action_map(map_path: str) -> Dict[str, Any]:
    with open(map_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_weapon_aiming_posture(kps: np.ndarray, scores: np.ndarray = None) -> Tuple[bool, float]:
    """
    Evaluates biomechanical weapon-aiming / firearm firing posture from skeleton joints:
    1. Both wrists elevated above waist level (at chest / shoulder height)
    2. Arm angles: elbow joint angles in stable firing/support range (55 deg - 135 deg)
    3. Torso orientation upright
    Returns (is_weapon_stance, confidence).
    """
    if kps is None:
        return False, 0.0

    if kps.ndim == 4:
        kp = kps[0]
    else:
        kp = kps

    l_shoulder = kp[:, 5, :]
    r_shoulder = kp[:, 6, :]
    l_elbow = kp[:, 7, :]
    r_elbow = kp[:, 8, :]
    l_wrist = kp[:, 9, :]
    r_wrist = kp[:, 10, :]
    l_hip = kp[:, 11, :]
    r_hip = kp[:, 12, :]

    # Torso length normalization
    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    hip_mid = (l_hip + r_hip) / 2.0
    torso_len = np.linalg.norm(shoulder_mid - hip_mid, axis=1)
    mean_torso = float(np.mean(torso_len)) if np.mean(torso_len) > 1e-3 else 1.0

    # Hand-to-waist distances
    dist_l_wrist_waist = np.linalg.norm(l_wrist - hip_mid, axis=1) / mean_torso
    dist_r_wrist_waist = np.linalg.norm(r_wrist - hip_mid, axis=1) / mean_torso
    l_hw_mean = float(np.mean(dist_l_wrist_waist))
    r_hw_mean = float(np.mean(dist_r_wrist_waist))

    # Arm angles at elbow
    def joint_angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cos_ang = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-6)
        return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

    l_ang_mean = float(np.mean(joint_angle(l_shoulder, l_elbow, l_wrist)))
    r_ang_mean = float(np.mean(joint_angle(r_shoulder, r_elbow, r_wrist)))

    # Torso inclination
    torso_vec = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])
    torso_dot = np.sum(torso_vec * vertical, axis=1) / (np.linalg.norm(torso_vec, axis=1) + 1e-6)
    torso_mean = float(np.mean(np.degrees(np.arccos(np.clip(torso_dot, -1.0, 1.0)))))

    is_weapon_stance = (
        (l_hw_mean > 0.80 and r_hw_mean > 0.70) and
        (55.0 <= l_ang_mean <= 135.0) and
        (55.0 <= r_ang_mean <= 135.0) and
        (torso_mean < 25.0)
    )
    conf = 0.0
    if is_weapon_stance:
        conf = float(np.clip(0.85 + (l_hw_mean + r_hw_mean - 1.5) * 0.08, 0.85, 0.98))
    return is_weapon_stance, conf


def map_predictions_to_threats(
    probs: np.ndarray,
    action_map: Dict[str, Any],
    kps: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Map the model's 120-class probabilities to S.W.O.R.D. threat groups.

    FIX: the previous version let a hand-written "weapon aiming posture" rule overwrite the model output —
    when the rule fired, the result was relabelled as NTU class 109 "shoot with gun" with probability >= 0.85,
    so reported actions were sometimes not the model's prediction at all. The rule (unvalidated; it also fires
    for benign poses such as holding a phone or a box at chest height) is now reported separately in
    `heuristic_aiming_posture` / `heuristic_aiming_conf`, and every model field comes from the model alone.
    Phase 5 can learn how much weight, if any, the heuristic deserves.
    """
    all_classes = action_map["all_ntu120_classes"]
    groups = action_map["groups"]

    top1_id = int(np.argmax(probs))
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_list = [
        {"class_id": int(i), "class_name": all_classes.get(int(i), f"class_{i}"), "prob": round(float(probs[i]), 4)}
        for i in top5_indices
    ]

    group_probs = {}
    assigned_group = "NEUTRAL_NORMAL"
    threat_score = 0.0
    for gname, gcfg in groups.items():
        indices = list(gcfg.get("class_indices", {}).keys())
        g_prob = float(np.sum(probs[indices])) if indices else 0.0
        group_probs[f"prob_{gname}"] = round(g_prob, 4)
        weight = float(gcfg.get("base_threat_weight", 0.0))
        thresh = float(gcfg.get("detection_threshold", 0.35))
        if g_prob >= thresh and g_prob * weight > threat_score:
            threat_score = g_prob * weight
            assigned_group = gname
    non_neutral_sum = sum(v for k, v in group_probs.items() if k != "prob_NEUTRAL_NORMAL")
    group_probs["prob_NEUTRAL_NORMAL"] = round(max(0.0, 1.0 - non_neutral_sum), 4)

    is_aiming, aiming_conf = check_weapon_aiming_posture(kps, scores) if kps is not None else (False, 0.0)
    return {
        "top1_id": top1_id,
        "top1_name": all_classes.get(top1_id, f"class_{top1_id}"),
        "top1_prob": round(float(probs[top1_id]), 4),
        "assigned_group": assigned_group,
        "threat_score": round(float(threat_score), 4),
        "group_probs": group_probs,
        "top5": top5_list,
        "heuristic_aiming_posture": bool(is_aiming),
        "heuristic_aiming_conf": round(float(aiming_conf), 3),
    }


# ------------------------------------------------------------------------------
# 4. Inference Pipeline Execution
# ------------------------------------------------------------------------------
def run_action_inference(
    clips: List[Dict[str, Any]],
    checkpoint_path: str,
    action_map_path: str,
    device: str = "cpu",
    flip_test: bool = True
) -> List[Dict[str, Any]]:
    """
    Executes inference over a list of skeleton clips (official PoseC3D test preprocessing; with flip_test the
    probabilities of the original and mirrored views are averaged, as in the official NTU-120 test config).
    """
    action_map = load_action_map(action_map_path)
    results = []

    device_obj = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    print(f"[INFER] Initializing PoseC3D inference engine on {device_obj}...")

    model = ResNet3dSlowOnly(in_channels=17, base_channels=32, num_classes=120)
    model = load_slowonly_weights(model, checkpoint_path, device=str(device_obj))
    model.to(device_obj)

    print(f"[INFER] Running action classification on {len(clips)} skeleton clips...")
    with torch.no_grad():
        for i, clip in enumerate(clips):
            kps = clip["keypoint"]           # [M, T, 17, 2]
            scores = clip["keypoint_score"]  # [M, T, 17]

            img_shape = tuple(int(v) for v in clip.get("img_shape", (1080, 1920)))
            inputs = posec3d_inputs(kps, scores, img_shape, flip_test=flip_test).to(device_obj)
            probs = F.softmax(model(inputs), dim=1).mean(dim=0).cpu().numpy()  # [120], averaged over views

            threat_analysis = map_predictions_to_threats(probs, action_map, kps=kps, scores=scores)

            record = {
                "clip_id": clip.get("frame_dir", f"clip_{i:04d}"),
                "video": clip.get("video", "unknown.mp4"),
                "track_id": int(clip.get("track_id", -1)),
                "start_frame": int(clip.get("start_frame", 0)),
                "end_frame": int(clip.get("end_frame", 0)),
                "timestamp": float(clip.get("timestamp", 0.0)),
                "top_action_id": threat_analysis["top1_id"],
                "top_action_name": threat_analysis["top1_name"],
                "top_action_prob": threat_analysis["top1_prob"],
                "threat_group": threat_analysis["assigned_group"],
                "threat_score": threat_analysis["threat_score"],
                "top5": threat_analysis["top5"],
                "heuristic_aiming_posture": threat_analysis["heuristic_aiming_posture"],
                "heuristic_aiming_conf": threat_analysis["heuristic_aiming_conf"],
            }
            # Flatten group probabilities
            for gk, gv in threat_analysis["group_probs"].items():
                record[gk] = gv

            record["probabilities_120"] = [round(float(p), 4) for p in probs]
            results.append(record)

    return results


class ActionInferenceEngine:
    """
    Stateful PoseC3D inference engine for live/real-time streaming and interactive applications.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.model = ResNet3dSlowOnly(in_channels=17, base_channels=32, num_classes=120)
        self.model = load_slowonly_weights(self.model, checkpoint_path, device=str(self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict_clip(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        width: int = 640,
        height: int = 480
    ) -> np.ndarray:
        """
        Runs action classification on keypoint [1, T, 17, 2] and scores [1, T, 17].
        Returns 120-class softmax probabilities [120].
        """
        with torch.no_grad():
            inputs = posec3d_inputs(keypoints, scores, (height, width), flip_test=True).to(self.device)
            probs = F.softmax(self.model(inputs), dim=1).mean(dim=0).cpu().numpy()
        return probs

    def predict_threats(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        action_map: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        End-to-end classification returning calibrated S.W.O.R.D. threat analysis.
        """
        probs = self.predict_clip(keypoints, scores)
        return map_predictions_to_threats(probs, action_map, kps=keypoints, scores=scores)


def save_results(
    results: List[Dict[str, Any]],
    output_prefix: str
) -> Tuple[str, str]:
    """
    Saves predictions to both CSV and JSON formats.
    """
    csv_path = f"{output_prefix}_actions.csv"
    json_path = f"{output_prefix}_actions.json"

    import csv
    fieldnames = [
        "video", "track_id", "start_frame", "end_frame", "timestamp",
        "top_action_id", "top_action_name", "top_action_prob",
        "threat_group", "threat_score",
        "prob_WEAPON_THREAT", "prob_VIOLENT_ATTACK",
        "prob_SUSPICIOUS_AGGRESSION", "prob_MEDICAL_ANOMALY",
        "prob_OBJECT_MANIPULATION", "prob_NEUTRAL_NORMAL",
        "heuristic_aiming_posture", "heuristic_aiming_conf"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Pretrained PoseC3D action inference on skeleton clips.")
    parser.add_argument("--clips", type=str, required=True, help="Input clips Pickle (.pkl) or JSON file")
    parser.add_argument("--checkpoint", type=str,
                        default=str(DEFAULT_CHECKPOINT),
                        help="Path to pretrained PoseC3D weights")
    parser.add_argument("--action_map", type=str,
                        default=default_action_map_path(),
                        help="Path to action_map.yaml")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output prefix (default: outputs/<clips_stem>)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda:0')")
    parser.add_argument("--no_flip_test", action="store_true", help="disable mirrored-view averaging")
    args = parser.parse_args()

    clips_path = Path(args.clips)
    if not clips_path.exists():
        print(f"[ERROR] Clips file not found: {clips_path}")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint file not found: {ckpt_path}")
        print("        Please run: python download_checkpoints.py")
        sys.exit(1)

    out_prefix = args.output_prefix
    if out_prefix is None:
        out_dir = Path(__file__).parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = str(out_dir / clips_path.stem)
    else:
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    if clips_path.suffix == ".pkl":
        with open(clips_path, "rb") as f:
            clips = pickle.load(f)
    else:
        with open(clips_path, "r", encoding="utf-8") as f:
            clips = json.load(f)

    print(f"=== S.W.O.R.D. Phase 4 - Action Inference ===")
    print(f"Clips File: {clips_path} ({len(clips)} clips)")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Device:     {args.device}")

    results = run_action_inference(
        clips=clips,
        checkpoint_path=str(ckpt_path),
        action_map_path=args.action_map,
        device=args.device,
        flip_test=not args.no_flip_test
    )

    csv_file, json_file = save_results(results, out_prefix)
    print(f"\n[DONE] Inference complete for {len(results)} clips.")
    print(f"       Results CSV:  {csv_file}")
    print(f"       Results JSON: {json_file}")


if __name__ == "__main__":
    main()
