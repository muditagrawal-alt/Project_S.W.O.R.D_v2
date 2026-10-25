"""
Numerical check: is the pure-PyTorch PoseC3D in the Action-Recognition branch the same function as the official
MMAction2 model when both load the same checkpoint?

    venv/bin/python research/tools/phase4/verify_posec3d_equivalence.py \
        --worktree ../SWORD_v2-phase4 --mmaction-python <isolated env with mmaction2>/bin/python --out <dir>

Runs the official ResNet3dSlowOnly + I3DHead (built from the branch's own config values) in an isolated MMAction2
environment, and the branch's model before and after the stride fix in the project venv, on identical inputs.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

OFFICIAL = r'''
import sys, types, numpy as np, torch
for name in ("mmaction.models.localizers.drn", "mmaction.models.localizers.drn.drn"):  # missing from the 1.2.0 wheel
    m = types.ModuleType(name); m.DRN = type("DRN", (), {}); m.__path__ = []; sys.modules[name] = m
from mmaction.models.backbones.resnet3d_slowonly import ResNet3dSlowOnly
from mmaction.models.heads.i3d_head import I3DHead
ckpt, inp, out = sys.argv[1:4]
bb = ResNet3dSlowOnly(depth=50, pretrained=None, in_channels=17, base_channels=32, num_stages=3, out_indices=(2,),
                      stage_blocks=(4, 6, 3), conv1_stride_s=1, pool1_stride_s=1, inflate=(0, 1, 1),
                      spatial_strides=(2, 2, 2), temporal_strides=(1, 1, 2), dilations=(1, 1, 1))
head = I3DHead(num_classes=120, in_channels=512, spatial_type="avg", dropout_ratio=0.5)
sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
bb.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
head.load_state_dict({k[len("cls_head."):]: v for k, v in sd.items() if k.startswith("cls_head.")}, strict=True)
bb.eval(); head.eval()
with torch.no_grad():
    feat = bb(torch.from_numpy(np.load(inp)))
    logits = head(feat)
np.save(out, logits.numpy())
print("official feature map:", tuple(feat.shape))
'''


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def synthetic_clip(T: int = 48) -> tuple:
    """A standing person in a 1920x1080 frame waving the right arm (COCO-17), scores 0.9."""
    base = np.array([[960, 300], [975, 285], [945, 285], [990, 295], [930, 295], [1030, 380], [890, 380], [1060, 480],
                     [860, 480], [1070, 580], [850, 580], [1010, 600], [910, 600], [1015, 750], [905, 750], [1020, 900],
                     [900, 900]], dtype=np.float32)
    kps = np.repeat(base[None, None], T, axis=1)  # [1, T, 17, 2]
    for t in range(T):
        a = np.sin(2 * np.pi * t / 16)
        kps[0, t, 8] = (860 - 60 * a, 400 - 40 * a)   # right elbow
        kps[0, t, 10] = (820 - 120 * a, 300 - 80 * a)  # right wrist
    return kps, np.full((1, T, 17), 0.9, dtype=np.float32), (1080, 1920)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--mmaction-python", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.worktree / "checkpoints" / "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth"

    fixed = load_module(args.worktree / "infer_actions.py", "infer_actions_fixed")
    kps, scores, shape = synthetic_clip()
    x = fixed.posec3d_inputs(kps, scores, shape, flip_test=True).numpy()
    np.save(args.out / "input.npy", x)

    (args.out / "official_runner.py").write_text(OFFICIAL)
    res = subprocess.run([args.mmaction_python, str(args.out / "official_runner.py"), str(ckpt), str(args.out / "input.npy"),
                          str(args.out / "official_logits.npy")], capture_output=True, text=True)
    print(res.stdout.strip() or res.stderr[-800:])
    official = np.load(args.out / "official_logits.npy")

    import torch

    def logits_of(mod):
        m = mod.ResNet3dSlowOnly()
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
        clean = {k.split(".", 1)[1]: v for k, v in sd.items()}
        m.load_state_dict(clean, strict=True)
        m.eval()
        with torch.no_grad():
            return m(torch.from_numpy(x)).numpy()

    original_src = subprocess.run(["git", "-C", str(args.worktree), "show", "HEAD:infer_actions.py"],
                                  capture_output=True, text=True, check=True).stdout
    (args.out / "infer_actions_original.py").write_text(original_src)
    original = load_module(args.out / "infer_actions_original.py", "infer_actions_original")
    results = {}
    softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    p_off = softmax(official).mean(0)
    for name, mod in (("fixed", fixed), ("original_branch", original)):
        lg = logits_of(mod)
        p = softmax(lg).mean(0)
        results[name] = {"max_abs_logit_diff_vs_official": float(np.abs(lg - official).max()),
                         "top1_matches_official": int(p.argmax()) == int(p_off.argmax()),
                         "top5_overlap_with_official": len(set(np.argsort(p)[-5:]) & set(np.argsort(p_off)[-5:])),
                         "top1_class": int(p.argmax()), "top1_prob": float(p.max())}
    results["official"] = {"top1_class": int(p_off.argmax()), "top1_prob": float(p_off.max())}
    (args.out / "equivalence.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
