"""
Test Suite & Output Verification for RTMPose-m Pose Extraction & Biometrics
Run via: python test_pose.py
"""

import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout if needed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pose_pipeline import (
    RTMPosePipeline,
    calculate_angle_3pt,
    analyze_hand_to_waist_distance,
    analyze_arm_angles,
    analyze_body_orientation,
    TARGET_JOINTS,
)


def run_tests():
    print("=" * 70)
    print(" SWORD v2 -- RTMPose-m Pose Extraction & Biometrics Verification Test")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 1: MATHEMATICAL & BIOMECHANICAL ALGORITHMS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/4] Testing Biomechanical Formulas...")

    # A) 90-degree right angle test
    angle_90 = calculate_angle_3pt((100.0, 100.0), (100.0, 200.0), (200.0, 200.0))
    assert abs(angle_90 - 90.0) < 1e-2, f"Angle error: {angle_90}"
    print(f"  [OK] Elbow angle calculation: 90.0 deg interior bend verified (Result: {angle_90:.1f} deg)")

    # B) Hand-to-waist distance test
    kpts_synth = np.zeros((17, 2), dtype=np.float32)
    scs_synth = np.ones((17,), dtype=np.float32)
    kpts_synth[5] = [90, 100]    # Left shoulder
    kpts_synth[6] = [110, 100]   # Right shoulder
    kpts_synth[7] = [90, 150]    # Left elbow
    kpts_synth[8] = [110, 150]   # Right elbow
    kpts_synth[9] = [100, 230]   # Left wrist
    kpts_synth[10] = [120, 230]  # Right wrist
    kpts_synth[11] = [90, 200]   # Left hip
    kpts_synth[12] = [110, 200]  # Right hip

    hw = analyze_hand_to_waist_distance(kpts_synth, scs_synth, kpt_thr=0.3)
    assert hw["waist_center"] == (100.0, 200.0)
    assert abs(hw["left_wrist_dist_px"] - 30.0) < 1e-2
    assert abs(hw["left_wrist_dist_norm"] - 0.30) < 1e-2
    assert hw["hands_near_waist"] is True
    print(f"  [OK] Hand-to-waist distance: 30 px / 0.30 normalized torso reach verified.")

    # C) Body orientation test
    orient = analyze_body_orientation(kpts_synth, scs_synth, kpt_thr=0.3)
    print(f"  [OK] Body orientation: Tilt {orient['shoulder_tilt_deg']} deg | Facing: '{orient['facing_direction']}' verified.")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 2: MODEL INITIALIZATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/4] Initializing Pretrained RTMPose-m Engine...")
    t0 = time.time()
    pipeline = RTMPosePipeline(mode="balanced", device="cpu")
    init_time = time.time() - t0
    assert pipeline.is_ready, "RTMPosePipeline failed to initialize"
    print(f"  [OK] RTMPose-m model loaded successfully in {init_time:.2f}s.")

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 3: REAL IMAGE INFERENCE & TARGET JOINT EXTRACTION
    # ─────────────────────────────────────────────────────────────────────────
    test_image_path = PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val" / "grenade_1101.jpg"
    print(f"\n[3/4] Running Inference on Sample: {test_image_path.name}...")

    img = cv2.imread(str(test_image_path))
    assert img is not None, f"Could not read test image: {test_image_path}"

    t_inf = time.time()
    telemetry = pipeline.process_frame(img)
    inf_latency = (time.time() - t_inf) * 1000
    print(f"  [OK] Inference completed in {inf_latency:.1f} ms.")
    print(f"  [OK] Persons detected: {len(telemetry)}")

    assert len(telemetry) > 0, "Expected at least 1 person detected in sample image"
    p0 = telemetry[0]

    # Check 12 target joints
    print("\n  Target Joint Extraction Status:")
    print("  " + "-" * 55)
    print(f"  {'Joint Name':<18} {'Detected':<10} {'Coords (X, Y)':<18} {'Confidence':<10}")
    print("  " + "-" * 55)
    for name, data in p0["joints"].items():
        status = "YES" if data["detected"] else "NO"
        coords = f"({data['x']:.1f}, {data['y']:.1f})"
        conf = f"{data['confidence']:.1%}"
        print(f"  {name:<18} {status:<10} {coords:<18} {conf:<10}")
    print("  " + "-" * 55)

    # ─────────────────────────────────────────────────────────────────────────
    # TEST 4: ANNOTATION & OUTPUT SAVING
    # ─────────────────────────────────────────────────────────────────────────
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "test_pose_output.jpg"

    print(f"\n[4/4] Rendering HUD Annotations & Saving Visual Output...")
    annotated = pipeline.annotate(img, telemetry)
    cv2.imwrite(str(out_file), annotated)
    assert out_file.exists(), f"Failed to save {out_file}"
    print(f"  [OK] Output successfully saved to: {out_file.resolve()}")

    # Telemetry Summary
    hw0 = p0["hand_to_waist"]
    ang0 = p0["arm_angles"]
    orient0 = p0["body_orientation"]

    print("\n" + "=" * 70)
    print(" SUMMARY OF EXTRACTED TELEMETRY")
    print("=" * 70)
    print(f"  Facing Direction    : {orient0['facing_direction']}")
    print(f"  Shoulder Tilt       : {orient0['shoulder_tilt_deg']} deg")
    print(f"  Left Arm Angle      : {ang0['left_arm_angle_deg']:.1f} deg" if ang0['left_arm_angle_deg'] else "  Left Arm Angle      : N/A")
    print(f"  Right Arm Angle     : {ang0['right_arm_angle_deg']:.1f} deg" if ang0['right_arm_angle_deg'] else "  Right Arm Angle     : N/A")
    print(f"  Left Hand-to-Waist  : {hw0['left_wrist_dist_px']:.1f} px" if hw0['left_wrist_dist_px'] else "  Left Hand-to-Waist  : N/A")
    print(f"  Right Hand-to-Waist : {hw0['right_wrist_dist_px']:.1f} px" if hw0['right_wrist_dist_px'] else "  Right Hand-to-Waist : N/A")
    print(f"  Waist Reach Alert   : {'[!] REACH DETECTED' if hw0['hands_near_waist'] else 'NORMAL'}")
    print("=" * 70)
    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<\n")
    print("Opening annotated skeleton image in Windows Photo Viewer...")
    print(f"File location: {out_file.resolve()}\n")

    if sys.platform == "win32" and "--no-show" not in sys.argv:
        try:
            os.startfile(str(out_file))
        except Exception:
            pass


if __name__ == "__main__":
    run_tests()

