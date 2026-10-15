# S.W.O.R.D. Phase 4 - Action Recognition

**Surveillance for Weapon Observation using Real-Time Deep Learning (S.W.O.R.D.)**  
Phase 4 focuses on **Pretrained Skeleton-Based Action Recognition** (Zero Training from Scratch).

---

## 1. Pipeline Overview

```
                      [ Raw CCTV Video Stream ]
                                 │
                 (Phase 2 ByteTrack + Phase 3 RTMPose)
                                 │
                                 ▼
      [ Canonical Intermediate Tracking & Pose JSON Schema ]
        - video, fps, width, height
        - per-frame detections: track_id, bbox, keypoints [17,2], scores [17]
                                 │
                                 ▼
                    [ clip_builder.py ]
        - Per-track temporal grouping
        - Missing frame interpolation (linear for gaps <= 3)
        - Quality filter (drop windows if >25% missing or avg conf < 0.3)
        - Sliding-window clips (T=48, stride=16) in MMAction2 format:
            keypoint [M, T, 17, 2]
            keypoint_score [M, T, 17]
                                 │
                                 ▼
                    [ infer_actions.py ]
        - Pretrained PoseC3D (SlowOnly-R50, NTU RGB+D 120 XSub)
        - 3D pseudo-heatmap volume generation (17 channels, 48 frames, 56x56)
        - Softmax probabilities over 120 action classes
        - Action-to-threat behavior group mapping (action_map.yaml)
                                 │
                                 ▼
            [ Action Probabilities & Threat Scores (CSV / JSON) ]
                                 │
                                 ▼
               [ Forward to Phase 5 Temporal Threat Fusion ]
```

---

## 2. Canonical Tracking & Pose Schema

All tracker and pose scripts interface through a tool-agnostic intermediate JSON format:

```json
{
  "video": "surveillance_cam01.mp4",
  "fps": 25.0,
  "width": 1280,
  "height": 720,
  "frames": [
    {
      "frame_idx": 0,
      "persons": [
        {
          "track_id": 3,
          "bbox": [210.5, 120.0, 340.2, 450.8],
          "keypoints": [
            [275.2, 140.1],
            [280.0, 135.4],
            [270.1, 135.6],
            [290.3, 142.0],
            [260.4, 142.1],
            [305.0, 180.2],
            [245.0, 180.5],
            [320.1, 230.4],
            [230.2, 230.8],
            [335.0, 280.0],
            [220.0, 280.2],
            [290.0, 310.0],
            [260.0, 310.5],
            [295.0, 380.0],
            [255.0, 380.2],
            [300.0, 445.0],
            [250.0, 445.5]
          ],
          "scores": [0.95, 0.92, 0.91, 0.88, 0.89, 0.96, 0.97, 0.94, 0.93, 0.90, 0.91, 0.95, 0.96, 0.92, 0.93, 0.89, 0.90]
        }
      ]
    },
    {
      "frame_idx": 1,
      "persons": []
    }
  ]
}
```

### Schema Rules & Validation:
1. `video`: String basename or path of video.
2. `fps`: Frame rate (float > 0).
3. `width`, `height`: Positive integer frame dimensions in pixels.
4. `frame_idx`: Zero-indexed integer frame number.
5. `persons`: List of detected persons. If a frame has no detections, it must be an empty list `[]`.
6. `bbox`: `[x1, y1, x2, y2]` in pixel coordinates.
7. `keypoints`: Exactly 17 `[x, y]` pairs in standard **COCO-17** order:
   - `0: nose`, `1: left_eye`, `2: right_eye`, `3: left_ear`, `4: right_ear`
   - `5: left_shoulder`, `6: right_shoulder`, `7: left_elbow`, `8: right_elbow`
   - `9: left_wrist`, `10: right_wrist`, `11: left_hip`, `12: right_hip`
   - `13: left_knee`, `14: right_knee`, `15: left_ankle`, `16: right_ankle`
8. `scores`: Exactly 17 confidence values in `[0.0, 1.0]`.

---

## 3. Verified Pretrained Models

All model weights are verified against the official OpenMMLab and PYSKL model zoo:

1. **PoseC3D (Primary Model)**:
   - Config: `configs/posec3d_slowonly_r50_ntu120.py`
   - Architecture: SlowOnly-R50 3D CNN (2.05M parameters, 17-channel heatmap volume)
   - Dataset: NTU RGB+D 120 (Cross-Subject split), 120 classes, 48-frame temporal window
   - Weights: `slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth` (8.3 MB)
   - Verified URL: `https://download.openmmlab.com/mmaction/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth`

2. **ST-GCN++ (Alternative Model)**:
   - Config: `configs/stgcnpp_ntu120.py`
   - Architecture: ST-GCN++ (PYSKL, 1.39M parameters)
   - Dataset: NTU RGB+D 120 (Cross-Subject split), 120 classes, 2D HRNet/COCO-17 keypoints
   - Weights: `stgcn_pyskl_ntu120_xsub_hrnet_j.pth` (12.5 MB)
   - Verified URL: `http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/j.pth`

To download models automatically:
```bash
python download_checkpoints.py --model all
```

---

## 4. Action-to-Threat Mapping (`action_map.yaml`)

Selected NTU-120 classes mapped to S.W.O.R.D. threat behavior groups:

| S.W.O.R.D. Threat Group | Base Weight | Threshold | NTU-120 Action Classes |
| :--- | :---: | :---: | :--- |
| **`WEAPON_THREAT`** | 1.00 | 0.25 | **106**: wield knife towards other person<br>**109**: shoot at other person with a gun |
| **`VIOLENT_ATTACK`** | 0.85 | 0.30 | **49**: punching/slapping other person<br>**50**: kicking other person<br>**51**: pushing other person<br>**101**: side kick<br>**105**: hit other person with something<br>**107**: knock over other person (hit with body) |
| **`SUSPICIOUS_AGGRESSION`** | 0.55 | 0.35 | **23**: kicking something<br>**53**: point finger at the other person<br>**56**: touch other person's pocket<br>**108**: grab other person's stuff<br>**115**: follow other person |
| **`MEDICAL_ANOMALY`** | 0.40 | 0.40 | **41**: staggering<br>**42**: falling<br>**43-46**: touch head/chest/back/neck (pain)<br>**47**: nausea/vomiting |
| **`OBJECT_MANIPULATION`** | 0.20 | 0.40 | **4**: drop, **5**: pickup, **6**: throw, **12**: tear up paper, **24**: reach into pocket |
| **`NEUTRAL_NORMAL`** | 0.00 | 0.50 | Walking, sitting, standing, clapping, waving, reading, writing, and all remaining everyday benign actions. |

---

## 5. Execution Guide & CLI Commands

### Step 1: Extract Tracks and Poses (Stand-in until Phase 2/3 scripts are ready)
```powershell
# Option A (Default, fast on CPU): YOLOv8-Pose with ByteTrack
python extract_tracks_poses.py --video sample.mp4 --output data/sample_tracks_poses.json --backend yolo_pose

# Option B (MMPose RTMPose-m top-down):
python extract_tracks_poses.py --video sample.mp4 --output data/sample_tracks_poses.json --backend mmpose_rtm
```

### Step 2: Build Skeleton Clips
```powershell
python clip_builder.py `
  --input data/sample_tracks_poses.json `
  --output_prefix outputs/sample_clips `
  --clip_len 48 `
  --stride 16 `
  --max_gap 3 `
  --max_missing_ratio 0.25 `
  --min_confidence 0.30
```
Generates:
- `outputs/sample_clips.pkl` (MMAction2 dataset annotation format)
- `outputs/sample_clips_manifest.json` (Human-readable metadata)

### Step 3: Run Pretrained Action Inference
```powershell
python infer_actions.py `
  --clips outputs/sample_clips.pkl `
  --checkpoint checkpoints/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth `
  --action_map configs/action_map.yaml `
  --output_prefix outputs/sample `
  --device cpu
```
Generates:
- `outputs/sample_actions.csv` (Per-clip top action, probabilities, threat group, threat score)
- `outputs/sample_actions.json` (Full predictions including 120-class probability vectors)

### Step 4: Render Annotated Output Video (.mp4)
```powershell
python visualize_actions.py `
  --video sample.mp4 `
  --tracks_poses data/sample_tracks_poses.json `
  --predictions outputs/sample_actions.csv `
  --output outputs/sample_annotated.mp4
```
Generates an annotated CCTV video with:
- Color-coded COCO-17 skeletons (limbs and joints)
- Dynamic bounding boxes color-coded by threat level (Green -> Yellow -> Red)
- On-person HUD badge (Track ID, Action Name, Action Probability %, Threat Level)
- Global surveillance header banner (Time, Frame, Active Tracks, System Threat Status)

### Step 5: Evaluate Against Labeled Ground Truth
```powershell
# Generate a template ground truth CSV
python evaluate.py --generate_template --template_path data/test_ground_truth_template.csv

# Evaluate predictions against ground truth
python evaluate.py `
  --ground_truth data/test_ground_truth.csv `
  --predictions outputs/sample_actions.csv
```

### Step 6: Optional Lightweight Fallback (Kinematic Rules + Logistic Regression)
```powershell
# Test rule-based feature extraction (hand-to-waist distance, wrist velocity, arm angle)
python fallback_head.py --demo
```

---

## 6. End-to-End Test Suite

To run the complete automated integration test on a synthetic video sequence:
```powershell
python test_pipeline.py
```
