# ==============================================================================
# S.W.O.R.D. Phase 4 - PoseC3D NTU-120 Model Configuration
# Architecture: SlowOnly-R50 3D CNN on 17-channel COCO Keypoint Pseudo-Heatmaps
# Trained on: NTU RGB+D 120 (Cross-Subject split), 120 classes, 48 frames
# ==============================================================================

model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowOnly',
        depth=50,
        pretrained=None,
        in_channels=17,
        base_channels=32,
        num_stages=3,
        out_indices=(2,),
        stage_blocks=(4, 6, 3),
        conv1_stride_s=1,
        pool1_stride_s=1,
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 2),
        dilations=(1, 1, 1),
    ),
    cls_head=dict(
        type='I3DHead',
        in_channels=512,
        num_classes=120,
        dropout_ratio=0.5,
        average_clips='prob',
    ),
)

dataset_type = 'PoseDataset'
num_classes = 120
clip_len = 48
frame_interval = 1

# COCO-17 keypoint definition
keypoint_info = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle',
}

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

# Standard evaluation / inference pipeline
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=clip_len, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1.0, allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='CenterCrop', crop_size=64),
    dict(type='GeneratePoseTarget', sigma=0.6, use_score=True, with_kp=True, with_limb=False),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=['frame_dir', 'label']),
    dict(type='ToTensor', keys=['imgs']),
]
