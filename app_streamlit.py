"""
================================================================================
S.W.O.R.D. (Surveillance for Weapon Observation using Real-Time Deep Learning)
Phase 4: Skeleton-Based Action Recognition Streamlit Application
================================================================================
Features:
- Video File Analysis (Uploaded clips, stock surveillance videos, web URLs)
- Live Webcam / RTSP Stream Surveillance
- Real-time COCO-17 Pose Estimation & ByteTrack Tracking
- Pretrained PoseC3D NTU-120 Inference & Surveillance Threat Mapping
- Side-by-side Video Playback, Metrics Cards & Action Timelines
================================================================================
"""

import os
import sys
import time
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure phase4_action directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from extract_tracks_poses import extract_with_ultralytics_bytetrack
from clip_builder import build_clips_from_intermediate_json as build_sliding_window_clips, save_clips
from infer_actions import run_action_inference, save_results, load_action_map, ActionInferenceEngine
from visualize_actions import render_annotated_video, draw_skeleton, THREAT_STYLES, COCO_LIMBS

# ------------------------------------------------------------------------------
# Streamlit Page Setup & Custom CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="S.W.O.R.D. - Action Recognition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container {
        background: #0d1117;
    }
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #58a6ff;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .threat-card {
        padding: 1rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .threat-critical { background: linear-gradient(135deg, #b30000, #ff1a1a); }
    .threat-high { background: linear-gradient(135deg, #d35400, #e67e22); }
    .threat-elevated { background: linear-gradient(135deg, #d4ac0d, #f1c40f); color: black; }
    .threat-medical { background: linear-gradient(135deg, #2471a3, #2980b9); }
    .threat-normal { background: linear-gradient(135deg, #1e8449, #27ae60); }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def download_url_video(url: str, dest_path: str) -> bool:
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        st.error(f"Failed to download video from URL: {e}")
        return False


def run_full_pipeline_on_video(
    video_path: str,
    output_dir: Path,
    device: str,
    progress_bar,
    status_text
) -> Dict[str, Any]:
    """Runs the 4-step S.W.O.R.D. Action Recognition pipeline."""
    video_stem = Path(video_path).stem
    json_path = output_dir / f"{video_stem}_tracks_poses.json"
    clips_prefix = output_dir / f"{video_stem}_clips"
    infer_prefix = output_dir / f"{video_stem}"
    output_video = output_dir / f"{video_stem}_annotated.mp4"
    actions_csv = output_dir / f"{video_stem}_actions.csv"

    ckpt_path = BASE_DIR / "checkpoints" / "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth"
    action_map_path = BASE_DIR / "configs" / "action_map.yaml"

    # Step 1: Detect & Track
    status_text.text("🔍 [Step 1/4] Running YOLOv8-pose + ByteTrack tracking...")
    progress_bar.progress(15)
    extract_with_ultralytics_bytetrack(
        video_path=video_path,
        output_json_path=str(json_path),
        model_name="yolov8n-pose.pt",
        device=device
    )

    # Step 2: Build Skeleton Clips
    status_text.text("📐 [Step 2/4] Constructing sliding-window skeleton clips (T=48)...")
    progress_bar.progress(40)
    clips, stats = build_sliding_window_clips(
        json_path=str(json_path),
        clip_len=48,
        clip_stride=16,
        max_missing_ratio=0.25,
        min_avg_confidence=0.30
    )
    save_clips(clips, str(clips_prefix))

    # Step 3: PoseC3D Pretrained Inference
    status_text.text("🧠 [Step 3/4] Running PoseC3D action recognition (120 NTU classes)...")
    progress_bar.progress(70)
    inference_results = run_action_inference(
        clips=clips,
        checkpoint_path=str(ckpt_path),
        action_map_path=str(action_map_path),
        device=device
    )
    save_results(inference_results, str(infer_prefix))

    # Step 4: Render Annotated Video
    status_text.text("🎬 [Step 4/4] Rendering annotated surveillance video with skeletons & badges...")
    progress_bar.progress(90)
    render_annotated_video(
        video_path=video_path,
        tracks_poses_path=str(json_path),
        predictions_path=str(actions_csv),
        output_video_path=str(output_video),
        conf_thresh=0.25
    )

    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")

    return {
        "annotated_video": str(output_video),
        "actions_csv": str(actions_csv),
        "results": inference_results,
        "tracks_json": str(json_path),
        "total_clips": len(clips)
    }


# ------------------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("S.W.O.R.D. Controls")
    st.markdown("**Phase 4: Action Recognition**")
    st.divider()

    mode = st.radio(
        "Choose Mode:",
        ["📁 Upload / Stock Video", "🌐 Download from URL", "📹 Live Webcam Surveillance"],
        index=0
    )

    st.subheader("Model Configuration")
    device_opt = st.selectbox("Compute Device:", ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"])
    st.caption("Pretrained Model: **PoseC3D (SlowOnly-R50 NTU-120)**")
    st.caption("Tracker: **ByteTrack + YOLOv8-pose**")

    st.divider()
    st.subheader("Threat Sensitivity")
    weapon_thresh = st.slider("Weapon Threat Threshold", 0.10, 0.90, 0.25, 0.05)
    violence_thresh = st.slider("Violence Threshold", 0.10, 0.90, 0.30, 0.05)
    aggression_thresh = st.slider("Aggression Threshold", 0.10, 0.90, 0.35, 0.05)

    st.divider()
    with st.expander("ℹ️ Threat Classes Reference"):
        st.markdown("""
        - 🔴 **WEAPON_THREAT**: Knife wield, shoot gun
        - 🔴 **VIOLENT_ATTACK**: Punch, kick, push, hit
        - 🟠 **SUSPICIOUS**: Follow, snatch, pickpocket
        - 🔵 **MEDICAL**: Falling, collapse, staggering
        - 🟢 **NORMAL**: Standing, walking, clapping
        """)


# ------------------------------------------------------------------------------
# Main Page Header
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">🛡️ S.W.O.R.D. Action Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Surveillance for Weapon Observation using Real-Time Deep Learning & Skeleton AI</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Mode 1 & 2: Video Files (Upload or URL)
# ------------------------------------------------------------------------------
if mode in ["📁 Upload / Stock Video", "🌐 Download from URL"]:
    video_file_path: Optional[str] = None
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "📁 Upload / Stock Video":
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a surveillance video clip (.mp4, .avi, .mov):",
                type=["mp4", "avi", "mov", "mkv"]
            )
        with col2:
            st.markdown("**Or choose a test sample:**")
            sample_choice = st.selectbox(
                "Select pre-loaded video:",
                ["(None - Use uploaded video)", "sample_2.mp4 (Surveillance scene)", "sword_test_sample.mp4"]
            )

        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.flush()
            tfile.close()
            video_file_path = tfile.name
        elif sample_choice == "sample_2.mp4 (Surveillance scene)":
            video_file_path = str(BASE_DIR / "data" / "sample_2.mp4")
        elif sample_choice == "sword_test_sample.mp4":
            video_file_path = str(BASE_DIR / "data" / "sword_test_sample.mp4")

    else:  # URL download
        url_input = st.text_input(
            "Enter Direct Video URL (.mp4):",
            placeholder="https://example.com/cctv_footage.mp4"
        )
        if st.button("Fetch and Load Video"):
            if url_input.strip():
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.close()
                with st.spinner("Downloading video from URL..."):
                    if download_url_video(url_input.strip(), tfile.name):
                        video_file_path = tfile.name
                        st.success("Video downloaded successfully!")
            else:
                st.warning("Please enter a valid URL.")

    # When video is loaded
    if video_file_path:
        st.divider()
        st.subheader("📹 Input Video Preview")
        st.video(video_file_path)

        # Check if pre-existing results exist for this video
        stem = Path(video_file_path).stem
        pre_annotated = output_dir / f"{stem}_annotated.mp4"
        pre_csv = output_dir / f"{stem}_actions.csv"
        pre_json = output_dir / f"{stem}_actions.json"

        btn_label = "⚡ Run S.W.O.R.D. Action Recognition"
        if pre_annotated.exists() and pre_csv.exists():
            st.info(f"💡 Previously processed results found for `{Path(video_file_path).name}`. You can inspect the results below or re-run analysis.")
            btn_label = "🔄 Re-run S.W.O.R.D. Action Recognition"

        run_clicked = st.button(btn_label, type="primary", use_container_width=True)

        if run_clicked:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                t0 = time.time()
                pipeline_out = run_full_pipeline_on_video(
                    video_path=video_file_path,
                    output_dir=output_dir,
                    device=device_opt,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                elapsed = time.time() - t0
                st.success(f"Processing finished in {elapsed:.1f} seconds!")
                st.session_state[f"res_{stem}"] = pipeline_out

            except Exception as e:
                st.error(f"Error during processing: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Display results if available (either from fresh run or existing)
        annotated_path = str(pre_annotated)
        csv_path = str(pre_csv)

        if os.path.exists(annotated_path) and os.path.exists(csv_path):
            st.divider()
            st.subheader("🛡️ S.W.O.R.D. Annotated Surveillance Output")

            df_acts = pd.read_csv(csv_path)

            vcol1, vcol2 = st.columns([3, 2])
            with vcol1:
                st.markdown("**Processed Video with Skeleton & Threat HUD:**")
                st.video(annotated_path)
                with open(annotated_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Annotated Video (.mp4)",
                        data=f.read(),
                        file_name=Path(annotated_path).name,
                        mime="video/mp4",
                        use_container_width=True
                    )

            with vcol2:
                st.markdown("### Threat & Action Summary")

                max_threat_score = 0.0
                highest_group = "NEUTRAL_NORMAL"
                top_action_str = "None"
                top_action_prob = 0.0

                if not df_acts.empty:
                    if "threat_score" in df_acts.columns:
                        max_idx = df_acts["threat_score"].idxmax()
                        max_threat_score = float(df_acts.loc[max_idx, "threat_score"])
                        highest_group = str(df_acts.loc[max_idx, "threat_group"])
                        top_action_str = str(df_acts.loc[max_idx, "top_action_name"])
                        top_action_prob = float(df_acts.loc[max_idx, "top_action_prob"])
                    else:
                        top_action_str = str(df_acts.iloc[0].get("top_action_name", "None"))

                style = THREAT_STYLES.get(highest_group, THREAT_STYLES["NEUTRAL_NORMAL"])

                card_class = "threat-normal"
                if highest_group == "WEAPON_THREAT":
                    card_class = "threat-critical"
                elif highest_group == "VIOLENT_ATTACK":
                    card_class = "threat-high"
                elif highest_group == "SUSPICIOUS_AGGRESSION":
                    card_class = "threat-elevated"
                elif highest_group == "MEDICAL_ANOMALY":
                    card_class = "threat-medical"

                st.markdown(f"""
                <div class="threat-card {card_class}">
                    <h3>{style['badge']}</h3>
                    <p style="margin:0; font-size:1.15rem;">Detected Behavior: <b>{top_action_str}</b></p>
                    <p style="margin:0; font-size:0.95rem;">Model Confidence: <b>{top_action_prob*100:.1f}%</b></p>
                    <p style="margin:0; font-size:0.90rem;">Threat Index: <b>{max_threat_score:.2f}</b></p>
                </div>
                """, unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                m1.metric("Analyzed Clips", len(df_acts))
                m2.metric("Distinct Actions", df_acts["top_action_name"].nunique() if not df_acts.empty else 0)

            # Action Distribution Chart
            if not df_acts.empty and "top_action_name" in df_acts.columns:
                st.divider()
                st.subheader("📊 Detected Action Classes Breakdown")
                
                chart_col1, chart_col2 = st.columns([3, 2])
                with chart_col1:
                    action_counts = df_acts["top_action_name"].value_counts().reset_index()
                    action_counts.columns = ["Action Name", "Clip Count"]
                    st.bar_chart(action_counts.set_index("Action Name"))

                with chart_col2:
                    st.markdown("**Top Observed Actions:**")
                    for _, row in action_counts.iterrows():
                        sub = df_acts[df_acts["top_action_name"] == row["Action Name"]]
                        avg_p = sub["top_action_prob"].mean() * 100
                        grp = sub["threat_group"].iloc[0]
                        st.markdown(f"- **{row['Action Name']}**: {row['Clip Count']} clips (Avg Conf: `{avg_p:.1f}%` | Group: `{grp}`)")

            # Detailed Timeline Table
            st.divider()
            st.subheader("⏱️ Temporal Action Recognition Log")
            if not df_acts.empty:
                display_cols = [c for c in [
                    "track_id", "start_frame", "end_frame", "timestamp",
                    "top_action_name", "top_action_prob", "threat_group", "threat_score"
                ] if c in df_acts.columns]
                
                display_df = df_acts[display_cols].copy()
                if "top_action_prob" in display_df.columns:
                    display_df["top_action_prob"] = (display_df["top_action_prob"] * 100).round(1).astype(str) + "%"

                st.dataframe(display_df, use_container_width=True)
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Actions Spreadsheet (.csv)",
                        data=f.read(),
                        file_name=Path(csv_path).name,
                        mime="text/csv"
                    )
            else:
                st.info("No person tracks exceeded the minimum duration for action classification in this clip.")


# ------------------------------------------------------------------------------
# Mode 3: Live Webcam Surveillance Feed
# ------------------------------------------------------------------------------
elif mode == "📹 Live Webcam Surveillance":
    st.info("🎥 Live Video Mode captures frames from your computer camera, extracts skeletons in real time, and alerts on threats.")

    cam_col1, cam_col2 = st.columns([1, 2])
    with cam_col1:
        camera_id = st.number_input("Camera Index (0 for default webcam):", min_value=0, max_value=5, value=0, step=1)
        start_btn = st.button("▶ Start Live Surveillance", type="primary", use_container_width=True)
        stop_btn = st.button("⏹ Stop Live Surveillance", use_container_width=True)

    with cam_col2:
        feed_placeholder = st.empty()
        status_box = st.empty()

    if start_btn:
        cap = cv2.VideoCapture(int(camera_id))
        if not cap.isOpened():
            st.error(f"Cannot open camera with index {camera_id}. Please verify your webcam is connected.")
        else:
            from ultralytics import YOLO
            status_box.success("Surveillance Active - Monitoring skeleton streams...")

            pose_model = YOLO("yolov8n-pose.pt")
            ckpt_path = BASE_DIR / "checkpoints" / "slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth"
            action_map_path = BASE_DIR / "configs" / "action_map.yaml"

            engine = None
            if ckpt_path.exists():
                try:
                    engine = ActionInferenceEngine(str(ckpt_path), device=device_opt)
                except Exception as ex:
                    st.warning(f"Could not load PoseC3D: {ex}")

            action_map = load_action_map(str(action_map_path))
            track_buffers: Dict[int, List[Dict[str, Any]]] = {}
            active_actions: Dict[int, Dict[str, Any]] = {}

            frame_count = 0
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                h, w = frame.shape[:2]

                # Run YOLO pose tracking
                res = pose_model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, device=device_opt)[0]

                if res.boxes is not None and res.boxes.id is not None and res.keypoints is not None:
                    ids = res.boxes.id.cpu().numpy().astype(int)
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    kpts = res.keypoints.xy.cpu().numpy()
                    scores = res.keypoints.conf.cpu().numpy()

                    for i, tid in enumerate(ids):
                        if tid not in track_buffers:
                            track_buffers[tid] = []

                        track_buffers[tid].append({
                            "kpts": kpts[i],
                            "scores": scores[i],
                            "bbox": xyxy[i]
                        })

                        # Keep rolling buffer of 48 frames
                        if len(track_buffers[tid]) > 48:
                            track_buffers[tid].pop(0)

                        # Run PoseC3D inference every 16 frames when buffer is full
                        if len(track_buffers[tid]) == 48 and frame_count % 16 == 0 and engine is not None:
                            kp_arr = np.zeros((1, 48, 17, 2), dtype=np.float32)
                            sc_arr = np.zeros((1, 48, 17), dtype=np.float32)
                            for t_idx, item in enumerate(track_buffers[tid]):
                                kp_arr[0, t_idx] = item["kpts"]
                                sc_arr[0, t_idx] = item["scores"]

                            probs = engine.predict_clip(kp_arr, sc_arr, w, h)
                            top1 = int(np.argmax(probs))
                            prob1 = float(probs[top1])
                            act_name = action_map["all_classes"].get(top1, f"act_{top1}")

                            # Check threat group
                            cur_group = "NEUTRAL_NORMAL"
                            for gname, gcfg in action_map["groups"].items():
                                g_indices = list(gcfg.get("class_indices", {}).keys())
                                if g_indices and float(np.sum(probs[g_indices])) >= gcfg.get("detection_threshold", 0.35):
                                    cur_group = gname
                                    break

                            active_actions[tid] = {
                                "action": act_name,
                                "prob": prob1,
                                "group": cur_group
                            }

                        # Draw HUD and skeleton
                        act_info = active_actions.get(tid, {
                            "action": "analyzing...", "prob": 0.50, "group": "NEUTRAL_NORMAL"
                        })
                        st_cfg = THREAT_STYLES.get(act_info["group"], THREAT_STYLES["NEUTRAL_NORMAL"])
                        col = st_cfg["color"]

                        # Draw skeleton
                        draw_skeleton(frame, kpts[i].tolist(), scores[i].tolist(), col, 0.25)

                        # Draw BBox
                        bx1, by1, bx2, by2 = [int(v) for v in xyxy[i]]
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)

                        # Draw Badge
                        badge_text = f"ID:{tid:02d} | {act_info['action']} ({int(act_info['prob']*100)}%)"
                        cv2.rectangle(frame, (bx1, max(0, by1 - 22)), (bx1 + len(badge_text) * 8 + 10, by1), col, -1)
                        cv2.putText(frame, badge_text, (bx1 + 4, max(12, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

                # Render surveillance top HUD
                cv2.rectangle(frame, (0, 0), (w, 32), (20, 20, 20), -1)
                cv2.putText(frame, "S.W.O.R.D. LIVE SKELETON ACTION SURVEILLANCE", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"FPS Live | F: {frame_count:05d}", (w - 180, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                # Convert BGR to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

            cap.release()
            status_box.info("Live Surveillance Stopped.")
