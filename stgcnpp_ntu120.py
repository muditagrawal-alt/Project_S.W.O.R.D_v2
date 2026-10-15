# ==============================================================================
# S.W.O.R.D. Phase 4 - ST-GCN++ NTU-120 Model Configuration
# Architecture: Spatio-Temporal Graph Convolutional Network (PYSKL / ST-GCN++)
# Trained on: NTU RGB+D 120 (Cross-Subject split), 120 classes, 2D HRNet/COCO-17
# ==============================================================================

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_type='unit_gcn',
        tcn_type='unit_tcn',
        graph_cfg=dict(layout='coco', mode='spatial'),
        in_channels=3,  # [x, y, score]
        base_channels=64,
        ch_ratio=2,
        num_stages=6,
        inflate_stages=[5],
        down_stages=[5],
        pretrained=None,
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=120,
        in_channels=256,
        dropout=0.0,
    ),
)

dataset_type = 'PoseDataset'
num_classes = 120
num_joints = 17
in_channels = 3  # (x, y, confidence)

keypoint_layout = 'coco'
num_person = 2

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=['frame_dir', 'label']),
    dict(type='ToTensor', keys=['imgs']),
]
