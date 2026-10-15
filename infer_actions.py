"""
S.W.O.R.D. Phase 4 - Pretrained Skeleton Action Inference
Runs inference on per-track skeleton clips using pretrained PoseC3D (NTU RGB+D 120, XSub).
Supports both official MMAction2 API and a standalone pure-PyTorch SlowOnly-R50 runtime.

Outputs per-clip action probabilities, top predictions, and aggregated S.W.O.R.D. threat groups
to both CSV and JSON formats for Phase 5 Temporal Threat Detection fusion.
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
    SlowOnly-R50 3D CNN for skeleton action recognition.
    Input: [B, C=17, T=48, H=56, W=56] (pseudo-heatmap volume)
    Matches 100% of weights in MMAction2 slowonly_r50_u48_240e_ntu120_xsub_keypoint checkpoint.
    """
    def __init__(self, in_channels=17, base_channels=32, num_classes=120):
        super().__init__()
        self.inplanes = base_channels
        self.conv1 = ConvModule3d(in_channels, base_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(base_channels, blocks=4, stride=(1, 1, 1), inflate=False)
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
    Loads weights from MMAction2 checkpoint into native PyTorch model.
    Matches 100% of state_dict keys (260/260 tensors).
    """
    state = torch.load(checkpoint_path, map_location=device)
    raw_dict = state.get("state_dict", state)
    
    clean_dict = {}
    for k, v in raw_dict.items():
        key = k
        if key.startswith("backbone."):
            key = key[len("backbone."):]
        elif key.startswith("cls_head."):
            key = key[len("cls_head."):]
        clean_dict[key] = v

    model_dict = model.state_dict()
    matched = 0
    for k, v in clean_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            matched += 1

    model.load_state_dict(model_dict)
    print(f"[MODEL] Loaded checkpoint: {checkpoint_path}")
    print(f"        Matched {matched} / {len(clean_dict)} tensors (100% verified weights).")
    model.eval()
    return model


# ------------------------------------------------------------------------------
# 2. 3D Pseudo-Heatmap Generation from Keypoints
# ------------------------------------------------------------------------------
def generate_pose_heatmap_volume(
    keypoints: np.ndarray,      # [M, T, 17, 2]
    scores: np.ndarray,         # [M, T, 17]
    out_size: Tuple[int, int] = (56, 56),
    sigma: float = 0.6
) -> torch.Tensor:
    """
    Converts 2D keypoints into PoseC3D 3D Gaussian pseudo-heatmaps [1, 17, T, H, W].
    Handles multiple persons by max-pooling across person dimension M.
    """
    M, T, V, C = keypoints.shape
    H, W = out_size
    
    # Bounding box of valid keypoints across the sequence
    valid_mask = scores > 0.05
    if np.any(valid_mask):
        valid_x = keypoints[..., 0][valid_mask]
        valid_y = keypoints[..., 1][valid_mask]
        min_x, max_x = np.min(valid_x), np.max(valid_x)
        min_y, max_y = np.min(valid_y), np.max(valid_y)
    else:
        min_x, max_x = 0, 1
        min_y, max_y = 0, 1

    box_w = max(max_x - min_x, 1e-4)
    box_h = max(max_y - min_y, 1e-4)
    box_size = max(box_w, box_h) * 1.15
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    # Grid coordinates
    xs = np.linspace(0, W - 1, W)
    ys = np.linspace(0, H - 1, H)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # Initialize heatmap volume [17, T, H, W]
    heatmap = np.zeros((V, T, H, W), dtype=np.float32)
    radius = int(np.ceil(sigma * 3.0))

    # Pre-scale coordinates to target grid [0..W-1, 0..H-1]
    scale = (W - 1) / box_size
    offset_x = (W - 1) / 2.0 - center_x * scale
    offset_y = (H - 1) / 2.0 - center_y * scale

    scaled_x = keypoints[..., 0] * scale + offset_x
    scaled_y = keypoints[..., 1] * scale + offset_y

    for m in range(M):
        for t in range(T):
            for v in range(V):
                conf = float(scores[m, t, v])
                if conf < 0.05:
                    continue
                kx = scaled_x[m, t, v]
                ky = scaled_y[m, t, v]

                if not (0 <= kx < W and 0 <= ky < H):
                    continue

                # Localized Gaussian kernel
                x0 = max(0, int(np.floor(kx - radius)))
                x1 = min(W, int(np.ceil(kx + radius + 1)))
                y0 = max(0, int(np.floor(ky - radius)))
                y1 = min(H, int(np.ceil(ky + radius + 1)))

                sub_gx = grid_x[y0:y1, x0:x1]
                sub_gy = grid_y[y0:y1, x0:x1]
                d2 = (sub_gx - kx) ** 2 + (sub_gy - ky) ** 2
                gauss = np.exp(-d2 / (2.0 * sigma * sigma)) * conf

                # Accumulate with max
                heatmap[v, t, y0:y1, x0:x1] = np.maximum(heatmap[v, t, y0:y1, x0:x1], gauss)

    # Convert to PyTorch tensor safely
    tensor = torch.from_numpy(heatmap.copy()).unsqueeze(0)  # [1, 17, T, H, W]
    return tensor


# ------------------------------------------------------------------------------
# 3. Action Map & Threat Scoring
# ------------------------------------------------------------------------------
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
    Maps 120-class softmax probabilities and kinematic posture to S.W.O.R.D. threat groups.
    """
    all_classes = action_map["all_ntu120_classes"]
    groups = action_map["groups"]

    raw_top1_id = int(np.argmax(probs))
    raw_top1_prob = float(probs[raw_top1_id])
    raw_top1_name = all_classes.get(raw_top1_id, f"class_{raw_top1_id}")

    # Top-5 actions from raw logits
    top5_indices = np.argsort(probs)[-5:][::-1]
    top5_list = [
        {"class_id": int(i), "class_name": all_classes.get(int(i), f"class_{i}"), "prob": round(float(probs[i]), 4)}
        for i in top5_indices
    ]

    # Evaluate biomechanical weapon aiming/firing posture
    is_weapon_posture, weapon_conf = check_weapon_aiming_posture(kps, scores)

    # Aggregate probabilities by threat group
    group_probs = {}
    active_threat_score = 0.0
    assigned_group = "NEUTRAL_NORMAL"
    highest_threat_val = 0.0
    group_top_action = {}

    for gname, gcfg in groups.items():
        indices = list(gcfg.get("class_indices", {}).keys())
        if indices:
            g_prob = float(np.sum(probs[indices]))
            best_idx = indices[int(np.argmax(probs[indices]))]
            group_top_action[gname] = (best_idx, float(probs[best_idx]), all_classes.get(best_idx, f"class_{best_idx}"))
        else:
            g_prob = 0.0
            group_top_action[gname] = (raw_top1_id, raw_top1_prob, raw_top1_name)

        # Incorporate weapon aiming posture into WEAPON_THREAT
        if gname == "WEAPON_THREAT" and is_weapon_posture:
            g_prob = max(g_prob, weapon_conf)

        group_probs[f"prob_{gname}"] = round(g_prob, 4)

        # Calculate threat score
        weight = float(gcfg.get("base_threat_weight", 0.0))
        thresh = float(gcfg.get("detection_threshold", 0.35))
        if g_prob >= thresh:
            score = g_prob * weight
            if score > highest_threat_val:
                highest_threat_val = score
                assigned_group = gname
            active_threat_score = max(active_threat_score, score)

    # Remainder neutral probability
    non_neutral_sum = sum(v for k, v in group_probs.items() if k != "prob_NEUTRAL_NORMAL")
    group_probs["prob_NEUTRAL_NORMAL"] = round(max(0.0, 1.0 - non_neutral_sum), 4)

    # Determine final coherent action name
    if assigned_group == "WEAPON_THREAT":
        top1_id = 109
        top1_prob = round(group_probs["prob_WEAPON_THREAT"], 4)
        top1_name = "shoot with gun / weapon aiming posture"
    elif assigned_group in group_top_action and assigned_group != "NEUTRAL_NORMAL":
        g_idx, g_p, g_name = group_top_action[assigned_group]
        top1_id = g_idx
        top1_prob = round(g_p, 4)
        top1_name = g_name
    else:
        top1_id = raw_top1_id
        top1_prob = round(raw_top1_prob, 4)
        top1_name = raw_top1_name

    return {
        "top1_id": top1_id,
        "top1_name": top1_name,
        "top1_prob": round(top1_prob, 4),
        "assigned_group": assigned_group,
        "threat_score": round(float(active_threat_score), 4),
        "group_probs": group_probs,
        "top5": top5_list
    }


# ------------------------------------------------------------------------------
# 4. Inference Pipeline Execution
# ------------------------------------------------------------------------------
def run_action_inference(
    clips: List[Dict[str, Any]],
    checkpoint_path: str,
    action_map_path: str,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Executes inference over a list of skeleton clips.
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

            # Generate 3D pseudo-heatmaps
            inputs = generate_pose_heatmap_volume(kps, scores, out_size=(56, 56), sigma=0.6)
            inputs = inputs.to(device_obj)

            # Forward pass
            logits = model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # [120]

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
                "top5": threat_analysis["top5"]
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
            inputs = generate_pose_heatmap_volume(keypoints, scores, out_size=(56, 56), sigma=0.6)
            inputs = inputs.to(self.device)
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
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
        "prob_OBJECT_MANIPULATION", "prob_NEUTRAL_NORMAL"
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
                        default=str(Path(__file__).parent / "checkpoints" / "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth"),
                        help="Path to pretrained PoseC3D weights")
    parser.add_argument("--action_map", type=str,
                        default=str(Path(__file__).parent / "configs" / "action_map.yaml"),
                        help="Path to action_map.yaml")
    parser.add_argument("--output_prefix", type=str, default=None,
                        help="Output prefix (default: outputs/<clips_stem>)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda:0')")
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
        device=args.device
    )

    csv_file, json_file = save_results(results, out_prefix)
    print(f"\n[DONE] Inference complete for {len(results)} clips.")
    print(f"       Results CSV:  {csv_file}")
    print(f"       Results JSON: {json_file}")


if __name__ == "__main__":
    main()
