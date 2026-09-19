"""
SWORD v2 — Multi-Model Ensemble Weapon Threat Detection Dashboard
Interactive Streamlit application for testing the 3-model ensemble:
YOLO26s + YOLO26m + RT-DETR-L with Weighted Boxes Fusion (WBF).

Features:
- Image Upload & 4-Panel Multi-Model Inspection
- Video File Analysis with Progress Tracking & MP4 Export
- Live Webcam Feed with Real-Time WBF Detection & Video Recording
- SOTA Benchmark & Evaluation Hub
"""

import os
import sys
import time
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Set OpenCV AVFoundation skip auth to prevent camera initialization hangs on macOS
os.environ["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"

import cv2
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ensemble.ensemble_model import SWORDEnsembleDetector, CLASS_NAMES
from ensemble.wbf import weighted_boxes_fusion

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM HUD STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SWORD v2 — Weapon Detection HUD",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Dark Defense/Security Command Center Theme */
    .stApp {
        background-color: #0A0E17;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Banner / HUD Header */
    .hud-header {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }
    .hud-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #38BDF8;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hud-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-top: 6px;
        letter-spacing: 0.5px;
    }
    
    /* Threat Alert Banners */
    .threat-banner-critical {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.3), rgba(185, 28, 28, 0.15));
        border-left: 6px solid #EF4444;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 16px 0;
        color: #FCA5A5;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .threat-banner-safe {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.25), rgba(5, 150, 105, 0.1));
        border-left: 6px solid #10B981;
        padding: 14px 18px;
        border-radius: 8px;
        margin: 16px 0;
        color: #6EE7B7;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTE FOR 10 WEAPON CLASSES (BGR & RGB)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLORS_RGB = {
    "Pistol": (239, 68, 68),          # Red
    "Shotgun": (249, 115, 22),        # Orange
    "assault rifle": (234, 179, 8),   # Yellow
    "grenade": (16, 185, 129),        # Green
    "knife": (6, 182, 212),           # Cyan
    "sword": (59, 130, 246),          # Blue
    "Sniper Dataset": (168, 85, 247), # Purple
    "SMG": (236, 72, 153),            # Pink
    "rocket launcher": (244, 63, 94), # Rose
    "Machine gun": (20, 184, 166),    # Teal
}

def get_class_color_bgr(name: str) -> Tuple[int, int, int]:
    rgb = CLASS_COLORS_RGB.get(name, (0, 255, 255))
    return (rgb[2], rgb[1], rgb[0])


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHING & INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_detector(use_3model: bool, device: str) -> SWORDEnsembleDetector:
    weights_s = str(PROJECT_ROOT / "weights" / "yolo26s_best.pt")
    weights_m = str(PROJECT_ROOT / "weights" / "yolo26m_best.pt")
    weights_rtdetr = str(PROJECT_ROOT / "weights" / "rtdetr_l_best.pt") if use_3model else None
    params_file = str(PROJECT_ROOT / "results" / "optimized_parameters.json")
    if not os.path.exists(params_file):
        params_file = None
    
    detector = SWORDEnsembleDetector(
        weights_s=weights_s,
        weights_m=weights_m,
        weights_rtdetr=weights_rtdetr,
        device=device,
        params_file=params_file,
    )
    return detector


def draw_bounding_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    conf_thr: float = 0.25,
    allowed_classes: Optional[List[str]] = None,
    show_timestamp: bool = False,
) -> Tuple[np.ndarray, List[Dict]]:
    """Annotate an image with styled bounding boxes and return detection metadata."""
    annotated = img.copy()
    h, w = annotated.shape[:2]
    detections = []

    if len(boxes) == 0:
        if show_timestamp:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                annotated, f"SWORD v2 LIVE | {ts}", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA
            )
        return annotated, detections

    for i in range(len(boxes)):
        score = float(scores[i])
        if score < conf_thr:
            continue
        
        cls_idx = int(labels[i])
        cls_name = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"Class {cls_idx}"

        if allowed_classes and cls_name not in allowed_classes:
            continue

        box = boxes[i]
        # Robust handling of both normalized [0, 1] and absolute pixel coordinates
        if (box[2] <= 1.05 and box[3] <= 1.05) and (w > 1 and h > 1):
            x1, y1 = int(box[0] * w), int(box[1] * h)
            x2, y2 = int(box[2] * w), int(box[3] * h)
        else:
            x1, y1 = int(box[0]), int(box[1])
            x2, y2 = int(box[2]), int(box[3])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color_bgr = get_class_color_bgr(cls_name)

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

        # Label badge
        label_text = f"{cls_name} {score:.1%}"
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        badge_y1 = max(0, y1 - th - baseline - 6)
        badge_y2 = y1
        badge_x2 = min(w - 1, x1 + tw + 8)

        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color_bgr, -1)
        cv2.putText(
            annotated, label_text, (x1 + 4, badge_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        detections.append({
            "Class": cls_name,
            "Confidence": score,
            "Box": [x1, y1, x2, y2],
        })

    if show_timestamp:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            annotated, f"SWORD v2 LIVE | {ts}", (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2, cv2.LINE_AA
        )

    return annotated, detections


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚔️ S.W.O.R.D v2")
    st.markdown("**Surveillance Weapon Object Recognition & Detection**")
    st.markdown("---")

    mode = st.radio(
        "Navigation Mode",
        [
            "📸 Image Analysis",
            "🎥 Video File Inference",
            "📹 Live Camera & Recording",
            "📊 Benchmark & Evaluation Hub",
        ],
        index=0,
    )
    st.markdown("---")
    st.markdown("### ⚙️ Ensemble Configuration")

    model_config = st.selectbox(
        "Ensemble Architecture",
        [
            "3-Model Ensemble (YOLO26s + YOLO26m + RT-DETR-L)",
            "2-Model Ensemble (YOLO26s + YOLO26m)",
        ],
        index=0,
    )
    use_3model = "3-Model" in model_config

    device_option = st.selectbox(
        "Hardware Acceleration",
        ["mps (Apple Silicon GPU)", "cpu", "cuda (NVIDIA)"],
        index=0,
    )
    device = device_option.split()[0]

    conf_thr = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)
    iou_thr = st.slider("WBF IoU Threshold", min_value=0.30, max_value=0.85, value=0.55, step=0.05)

    selected_classes = st.multiselect(
        "Class Filter (All 10 enabled by default)",
        options=CLASS_NAMES,
        default=CLASS_NAMES,
    )

    st.markdown("---")
    st.markdown("<small>Project S.W.O.R.D v2 — Research & Defense AI</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DETECTOR
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Initializing SWORD v2 Ensemble Detector..."):
    try:
        detector = get_detector(use_3model=use_3model, device=device)
        detector.conf_thr = conf_thr
        detector.iou_thr = iou_thr
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure weights are available in `weights/`.")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1: IMAGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if mode == "📸 Image Analysis":
    st.markdown("""
    <div class="hud-header">
        <h1 class="hud-title">📸 Image Threat Inspection & Multi-Model Fusion</h1>
        <div class="hud-subtitle">Upload high-resolution images or select curated validation benchmarks to inspect individual model detections and Weighted Boxes Fusion.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    sample_images_dir = PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val"
    sample_options = ["None (Upload custom image)"]
    if sample_images_dir.exists():
        found_samples = sorted([f.name for f in list(sample_images_dir.glob("*.jpg"))[:15]])
        sample_options += found_samples

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )

    with col2:
        selected_sample = st.selectbox("Or choose a validation sample image:", sample_options)

    image_source = None
    img_bgr = None
    if uploaded_file is not None:
        image_source = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image_source)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    elif selected_sample != "None (Upload custom image)":
        sample_path = sample_images_dir / selected_sample
        img_bgr = cv2.imread(str(sample_path))
        if img_bgr is not None:
            image_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if img_bgr is not None:
        t0 = time.time()
        with st.spinner("Running Multi-Model Inference & Weighted Boxes Fusion..."):
            all_dets = detector.predict_and_compare(img_bgr)
        elapsed_ms = (time.time() - t0) * 1000

        # Retrieve fused detections
        fused = all_dets.get("Ensemble (WBF)", all_dets.get("ensemble"))
        annotated_fused, detections = draw_bounding_boxes(
            img_bgr, fused["boxes"], fused["scores"], fused["labels"],
            conf_thr=conf_thr, allowed_classes=selected_classes
        )
        annotated_fused_rgb = cv2.cvtColor(annotated_fused, cv2.COLOR_BGR2RGB)

        # Threat Alert Banner
        if len(detections) > 0:
            weapon_list = ", ".join([f"{d['Class']} ({d['Confidence']:.1%})" for d in detections])
            st.markdown(f"""
            <div class="threat-banner-critical">
                🚨 <strong>WEAPON THREAT DETECTED:</strong> {weapon_list}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="threat-banner-safe">
                ✅ <strong>ALL CLEAR:</strong> No weapons detected above confidence threshold.
            </div>
            """, unsafe_allow_html=True)

        # Metric Badges
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(detections)}</div><div class="metric-label">Detected Weapons</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed_ms:.1f} ms</div><div class="metric-label">Total Latency</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(detector.models)}</div><div class="metric-label">Models Active</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{conf_thr:.2f}</div><div class="metric-label">Confidence Floor</div></div>', unsafe_allow_html=True)

        st.markdown("### Visual Inspection")
        view_mode = st.radio("View Mode:", ["⚔️ Fused Ensemble Detection", "🔍 Multi-Model Comparison"], horizontal=True)

        if view_mode == "⚔️ Fused Ensemble Detection":
            st.image(annotated_fused_rgb, caption="SWORD v2 Weighted Boxes Fusion Output", use_container_width=True)
        else:
            display_models = [k for k in all_dets.keys() if k != "ensemble"] + (["ensemble"] if "ensemble" in all_dets else [])
            grid_cols = st.columns(len(display_models))
            for i, model_name in enumerate(display_models):
                det_data = all_dets[model_name]
                label_display = "WBF Ensemble" if model_name == "ensemble" else model_name
                with grid_cols[i]:
                    sub_ann, sub_dets = draw_bounding_boxes(
                        img_bgr, det_data["boxes"], det_data["scores"], det_data["labels"],
                        conf_thr=conf_thr, allowed_classes=selected_classes
                    )
                    st.markdown(f"**{label_display}** ({len(sub_dets)} found)")
                    st.image(cv2.cvtColor(sub_ann, cv2.COLOR_BGR2RGB), use_container_width=True)

        if len(detections) > 0:
            st.markdown("### 📋 Detections Log")
            df = pd.DataFrame([{
                "Class": d["Class"],
                "Confidence": f"{d['Confidence']:.2%}",
                "Bounding Box (X1, Y1, X2, Y2)": f"[{d['Box'][0]}, {d['Box'][1]}, {d['Box'][2]}, {d['Box'][3]}]",
            } for d in detections])
            st.dataframe(df, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2: VIDEO FILE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "🎥 Video File Inference":
    st.markdown("""
    <div class="hud-header">
        <h1 class="hud-title">🎥 Video File Threat Analysis & Export</h1>
        <div class="hud-subtitle">Process surveillance video clips with frame-by-frame WBF ensemble inference and export MP4 videos with burned-in detection telemetry.</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader("Upload a Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov", "mkv"])

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        frame_skip = st.slider("Process Every Nth Frame (1 = all frames, 2 = 2x faster)", min_value=1, max_value=5, value=1)
    with col_v2:
        max_duration_sec = st.slider("Max Video Duration to Process (seconds)", min_value=5, max_value=60, value=20)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps

        st.info(f"Video Loaded: {width}×{height} @ {fps:.1f} FPS | Total Duration: {duration:.1f}s ({total_frames} frames)")

        if st.button("🚀 Start Video Analysis & Processing", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            preview_box = st.empty()

            out_temp = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4")
            out_temp.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_temp.name, fourcc, fps / frame_skip, (width, height))

            max_frames_to_process = min(total_frames, int(max_duration_sec * fps))
            frame_idx = 0
            processed_count = 0
            threat_count = 0
            t_start = time.time()

            while cap.isOpened() and frame_idx < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    # Run WBF Ensemble prediction and extract index 0
                    res = detector.predict(frame)[0]
                    ann_frame, dets = draw_bounding_boxes(
                        frame, res["boxes"], res["scores"], res["labels"],
                        conf_thr=conf_thr, allowed_classes=selected_classes,
                        show_timestamp=True,
                    )
                    writer.write(ann_frame)
                    if len(dets) > 0:
                        threat_count += 1

                    if processed_count % 5 == 0:
                        preview_box.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), caption=f"Analyzing Frame {frame_idx}/{max_frames_to_process}", use_container_width=True)

                    processed_count += 1
                    progress_bar.progress(float(frame_idx) / float(max_frames_to_process))
                    elapsed = time.time() - t_start
                    status_text.text(f"Processing: frame {frame_idx}/{max_frames_to_process} ({processed_count / elapsed:.1f} FPS)")

                frame_idx += 1

            cap.release()
            writer.release()
            progress_bar.progress(1.0)
            status_text.success(f"Processing complete! {processed_count} frames analyzed in {time.time() - t_start:.1f}s.")

            final_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix="_web.mp4")
            final_mp4.close()
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", out_temp.name,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "fast", final_mp4.name
            ]
            subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(final_mp4.name, "rb") as fp:
                video_bytes = fp.read()
            st.video(video_bytes)

            st.download_button(
                label="💾 Download Annotated Video (MP4)",
                data=video_bytes,
                file_name="sword_v2_annotated_detection.mp4",
                mime="video/mp4",
            )


# ─────────────────────────────────────────────────────────────────────────────
# MODE 3: LIVE CAMERA & RECORDING
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "📹 Live Camera & Recording":
    st.markdown("""
    <div class="hud-header">
        <h1 class="hud-title">📹 Live Camera Threat Detection & Video Recording</h1>
        <div class="hud-subtitle">Stream live video from your webcam or simulated CCTV stream, view real-time WBF weapon detections, and record telemetry-annotated MP4 video.</div>
    </div>
    """, unsafe_allow_html=True)

    cam_tab1, cam_tab2 = st.tabs(["🔴 Live Stream & Telemetry Recorder", "📸 Browser Snapshot Camera"])

    with cam_tab1:
        st.markdown("#### Live Stream & Recording Controls")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            cam_source_type = st.selectbox(
                "Video Source",
                ["Webcam (Hardware Camera)", "Simulated CCTV Feed (Sample Stream)"],
                index=0,
            )
            cam_index = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0) if "Webcam" in cam_source_type else 0
        with c2:
            record_duration_sec = st.slider("Recording / Session Duration (seconds)", min_value=5, max_value=60, value=15)
        with c3:
            enable_rec = st.checkbox("🔴 Record Annotated Telemetry Video", value=True)
            res_choice = st.selectbox("Resolution", ["640x480", "1280x720"], index=0)

        res_w, res_h = map(int, res_choice.split("x"))

        st.markdown("---")
        action_btn = st.button("🔴 Start Live Stream & Record", type="primary", use_container_width=True)

        video_placeholder = st.empty()
        alert_placeholder = st.empty()
        hud_placeholder = st.empty()
        download_placeholder = st.empty()

        if action_btn:
            # Initialize source
            if "Webcam" in cam_source_type:
                cap = cv2.VideoCapture(int(cam_index))
                if not cap.isOpened():
                    st.error(f"Cannot access camera index {cam_index}. If on macOS, ensure camera permissions are granted.")
                    cap = None
            else:
                # Simulated CCTV stream using sample validation images
                sample_dir = PROJECT_ROOT / "dataset" / "yolo_dataset" / "images" / "val"
                sample_files = sorted(list(sample_dir.glob("*.jpg")))[:30] if sample_dir.exists() else []
                if not sample_files:
                    st.error("No sample images found for simulated stream.")
                    cap = None
                else:
                    cap = "SIMULATED"

            if cap is not None:
                # Video writer setup
                writer = None
                raw_rec_path = None
                if enable_rec:
                    rec_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_rec_raw.mp4")
                    rec_raw.close()
                    raw_rec_path = rec_raw.name
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(raw_rec_path, fourcc, 15.0, (res_w, res_h))

                total_frames = int(record_duration_sec * 15)
                frame_count = 0
                fps_history = []
                detected_threats_total = 0
                t_start = time.time()

                progress_bar = st.progress(0.0)

                try:
                    for i in range(total_frames):
                        t_frame = time.time()
                        
                        if cap == "SIMULATED":
                            img_path = sample_files[i % len(sample_files)]
                            frame = cv2.imread(str(img_path))
                            time.sleep(0.05)
                        else:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("Frame capture interrupted.")
                                break

                        frame = cv2.resize(frame, (res_w, res_h))

                        # Run WBF ensemble inference and unpack index 0
                        preds = detector.predict(frame)[0]
                        ann_frame, dets = draw_bounding_boxes(
                            frame, preds["boxes"], preds["scores"], preds["labels"],
                            conf_thr=conf_thr, allowed_classes=selected_classes,
                            show_timestamp=True,
                        )

                        if writer is not None:
                            writer.write(ann_frame)

                        if len(dets) > 0:
                            detected_threats_total += len(dets)
                            threat_str = ", ".join([f"{d['Class']} ({d['Confidence']:.1%})" for d in dets])
                            alert_placeholder.markdown(f"""
                            <div class="threat-banner-critical">
                                🚨 <strong>LIVE THREAT DETECTED:</strong> {threat_str}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            alert_placeholder.markdown("""
                            <div class="threat-banner-safe">
                                ✅ <strong>LIVE MONITORING SECURE:</strong> No weapons detected in current field of view.
                            </div>
                            """, unsafe_allow_html=True)

                        fps = 1.0 / max(0.001, (time.time() - t_frame))
                        fps_history.append(fps)
                        if len(fps_history) > 30:
                            fps_history.pop(0)
                        avg_fps = sum(fps_history) / len(fps_history)

                        # Update preview
                        video_placeholder.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        hud_placeholder.markdown(f"**Session:** `{i+1}/{total_frames} frames` | **FPS:** `{avg_fps:.1f}` | **Active Threats:** `{len(dets)}` | **Total Threats Logged:** `{detected_threats_total}`")
                        progress_bar.progress(float(i + 1) / float(total_frames))

                finally:
                    if cap != "SIMULATED" and cap is not None:
                        cap.release()
                    if writer is not None:
                        writer.release()
                        progress_bar.progress(1.0)
                        st.success("Session complete! Finalizing recorded video encoding...")

                        # Transcode to H.264 web MP4 via ffmpeg
                        web_rec_file = tempfile.NamedTemporaryFile(delete=False, suffix="_web_rec.mp4")
                        web_rec_file.close()
                        subprocess.run([
                            "ffmpeg", "-y", "-i", raw_rec_path,
                            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "fast", web_rec_file.name
                        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                        with open(web_rec_file.name, "rb") as f:
                            rec_bytes = f.read()

                        download_placeholder.markdown("### 💾 Recorded Telemetry Video Ready")
                        st.video(rec_bytes)
                        st.download_button(
                            label="💾 Download Recorded Telemetry Video (MP4)",
                            data=rec_bytes,
                            file_name=f"sword_v2_live_recording_{time.strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

    with cam_tab2:
        st.markdown("#### Browser Snapshot Camera")
        snap = st.camera_input("Capture an image using your browser camera:")
        if snap is not None:
            snap_img = Image.open(snap).convert("RGB")
            snap_bgr = cv2.cvtColor(np.array(snap_img), cv2.COLOR_RGB2BGR)

            with st.spinner("Analyzing snapshot with SWORD v2 Ensemble..."):
                snap_preds = detector.predict(snap_bgr)[0]
            snap_ann, snap_dets = draw_bounding_boxes(
                snap_bgr, snap_preds["boxes"], snap_preds["scores"], snap_preds["labels"],
                conf_thr=conf_thr, allowed_classes=selected_classes, show_timestamp=True,
            )
            st.image(cv2.cvtColor(snap_ann, cv2.COLOR_BGR2RGB), caption="Snapshot Detection", use_container_width=True)
            if len(snap_dets) > 0:
                threats = ", ".join([f"{d['Class']} ({d['Confidence']:.1%})" for d in snap_dets])
                st.error(f"🚨 Weapons detected: {threats}")
            else:
                st.success("✅ No weapons detected.")


# ─────────────────────────────────────────────────────────────────────────────
# MODE 4: BENCHMARK & EVALUATION HUB
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "📊 Benchmark & Evaluation Hub":
    st.markdown("""
    <div class="hud-header">
        <h1 class="hud-title">📊 SOTA Benchmark & Research Evaluation Hub</h1>
        <div class="hud-subtitle">Direct empirical comparison between Benchmark Results 1 (Baseline) and Benchmark Results 2 (Non-Training SOTA Ensemble) on the 500-image stratified validation dataset on Apple Silicon (MPS).</div>
    </div>
    """, unsafe_allow_html=True)

    b1_file = PROJECT_ROOT / "results" / "benchmark_results_1" / "benchmark_results_1.json"
    b2_file = PROJECT_ROOT / "results" / "benchmark_results_2" / "benchmark_results_2.json"
    comp_md_file = PROJECT_ROOT / "results" / "benchmark_comparison" / "comparison_summary_report.md"

    bench_tab1, bench_tab2, bench_tab3 = st.tabs([
        "🔬 Comparative Analysis (Run 1 vs Run 2)",
        "🚀 Benchmark Results 2 (Optimized SOTA)",
        "📋 Benchmark Results 1 (Baseline)",
    ])

    with bench_tab1:
        st.markdown("### 🏆 Benchmark Comparison: Baseline vs Non-Training Optimized Ensemble")
        if comp_md_file.exists():
            with open(comp_md_file, "r") as fp:
                comp_text = fp.read()
            st.markdown(comp_text)

            fig_comp = PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_benchmark_comparison_1_vs_2.png"
            fig_delta = PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_per_class_delta.png"
            fig_scale = PROJECT_ROOT / "results" / "benchmark_comparison" / "fig_scale_sensitivity_1_vs_2.png"

            c1, c2 = st.columns(2)
            with c1:
                if fig_comp.exists():
                    st.image(str(fig_comp), caption="Macro Metrics Comparison: Run 1 vs Run 2", use_container_width=True)
            with c2:
                if fig_scale.exists():
                    st.image(str(fig_scale), caption="Object Scale Robustness (SO-WDet)", use_container_width=True)

            if fig_delta.exists():
                st.image(str(fig_delta), caption="Per-Class AP@50 Relative Gain (%)", use_container_width=True)
        else:
            st.info("Run `python ensemble/archive_and_compare_benchmarks.py` to generate comparative metrics.")

    with bench_tab2:
        st.markdown("### 🚀 Benchmark Results 2 (Optimized Non-Training SOTA)")
        if b2_file.exists():
            import json
            with open(b2_file) as fp:
                b2_data = json.load(fp)
            rows2 = []
            for name, metrics in b2_data["results"].items():
                rows2.append({
                    "Architecture": name,
                    "mAP@50": f"{metrics.get('mAP50', 0.0):.4f}",
                    "mAP@50:95": f"{metrics.get('mAP50-95', 0.0):.4f}",
                    "Precision": f"{metrics.get('p', 0.0):.4f}",
                    "Recall": f"{metrics.get('r', 0.0):.4f}",
                    "F1 Score": f"{metrics.get('f1', 0.0):.4f}",
                    "Latency (ms)": f"{b2_data['latencies'].get(name, 0.0):.1f}",
                    "FPS": f"{(1000.0 / max(0.1, b2_data['latencies'].get(name, 1.0))):.1f}",
                })
            df2 = pd.DataFrame(rows2)
            st.dataframe(df2, use_container_width=True)

            fig2 = PROJECT_ROOT / "results" / "benchmark_results_2" / "fig_benchmark_comparison_2.png"
            if fig2.exists():
                st.image(str(fig2), caption="Benchmark Results 2 Summary", use_container_width=True)
        else:
            st.info("Benchmark Results 2 not found.")

    with bench_tab3:
        st.markdown("### 📋 Benchmark Results 1 (Baseline Initial Run)")
        if b1_file.exists():
            import json
            with open(b1_file) as fp:
                b1_data = json.load(fp)
            rows1 = []
            for name, metrics in b1_data["results"].items():
                rows1.append({
                    "Architecture": name,
                    "mAP@50": f"{metrics.get('mAP50', 0.0):.4f}",
                    "mAP@50:95": f"{metrics.get('mAP50-95', 0.0):.4f}",
                    "Precision": f"{metrics.get('p', 0.0):.4f}",
                    "Recall": f"{metrics.get('r', 0.0):.4f}",
                    "F1 Score": f"{metrics.get('f1', 0.0):.4f}",
                    "Latency (ms)": f"{b1_data['latencies'].get(name, 0.0):.1f}",
                    "FPS": f"{(1000.0 / max(0.1, b1_data['latencies'].get(name, 1.0))):.1f}",
                })
            df1 = pd.DataFrame(rows1)
            st.dataframe(df1, use_container_width=True)

            fig1 = PROJECT_ROOT / "results" / "benchmark_results_1" / "fig_benchmark_comparison_1.png"
            if fig1.exists():
                st.image(str(fig1), caption="Benchmark Results 1 Summary", use_container_width=True)
        else:
            st.info("Benchmark Results 1 not found.")
