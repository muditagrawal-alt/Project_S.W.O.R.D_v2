"""
S.W.O.R.D Phase 2 — Generic Tracking Pipeline Entry Point.
Executes the full modular pipeline:
Video Source -> Dual Detection (Weapon WBF Ensemble + Person)
             -> Dual ByteTrack Tracking
             -> Trajectory & Kinematics (Direction, Velocity, Acceleration)
             -> Zone Transitions & Dwell Analytics
             -> Spatio-Temporal Person <-> Weapon Association
             -> Machine-Readable Structured Outputs & Reproducibility Metadata
"""

import os
import sys
import time
import uuid
import datetime
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import cv2
import numpy as np
import supervision as sv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ..detection import WeaponEnsembleDetector, PersonDetector
    from ..tracking import DualStreamTracker, TrackObservation, TRACKING_CSV_COLUMNS
    from ..analytics import KinematicsEngine, ZoneDwellEngine
    from ..association import SpatioTemporalAssociator
    from ..visualization import VideoVisualizer
except (ImportError, ValueError):
    from persistent_video_tracking.detection import WeaponEnsembleDetector, PersonDetector
    from persistent_video_tracking.tracking import DualStreamTracker, TrackObservation, TRACKING_CSV_COLUMNS
    from persistent_video_tracking.analytics import KinematicsEngine, ZoneDwellEngine
    from persistent_video_tracking.association import SpatioTemporalAssociator
    from persistent_video_tracking.visualization import VideoVisualizer


def resolve_config_path(config_path_str: Optional[str] = None) -> str:
    """Resolve config path across working directory, package, or repository."""
    if config_path_str and os.path.exists(config_path_str):
        return os.path.abspath(config_path_str)

    pkg_default = Path(__file__).resolve().parent.parent / "config" / "tracking_config.yaml"
    if pkg_default.exists():
        return str(pkg_default.resolve())

    repo_default = Path(__file__).resolve().parent.parent.parent / "persistent_video_tracking" / "config" / "tracking_config.yaml"
    if repo_default.exists():
        return str(repo_default.resolve())

    raise FileNotFoundError(f"Configuration file not found: {config_path_str}")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration YAML file."""
    resolved_path = resolve_config_path(config_path)
    with open(resolved_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    source_path: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    device_override: Optional[str] = None,
    save_video: Optional[bool] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the generic persistent video tracking pipeline on an arbitrary video source.
    """
    start_time_wall = time.time()
    run_id = f"run_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    print("=" * 70, flush=True)
    print(f"  S.W.O.R.D. v2 — PERSISTENT VIDEO TRACKING PIPELINE", flush=True)
    print(f"  Run ID: {run_id}", flush=True)
    print(f"  Source: {source_path}", flush=True)
    print("=" * 70, flush=True)

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source video not found: {source_path}")

    # 1. Load Configuration
    cfg = load_config(config_path)

    # Resolve output directory
    if output_dir:
        base_out = output_dir
    else:
        cfg_base = cfg.get("output", {}).get("base_dir", "persistent_video_tracking/outputs")
        if os.path.isabs(cfg_base) or os.path.exists(cfg_base):
            base_out = cfg_base
        else:
            repo_base = PROJECT_ROOT / cfg_base
            pkg_base = Path(__file__).resolve().parent.parent / cfg_base
            if repo_base.exists():
                base_out = str(repo_base.resolve())
            elif pkg_base.exists():
                base_out = str(pkg_base.resolve())
            else:
                base_out = str(repo_base)

    video_stem = Path(source_path).stem
    run_output_dir = os.path.join(base_out, video_stem)

    dirs = {
        "tracking": os.path.join(run_output_dir, "tracking"),
        "trajectories": os.path.join(run_output_dir, "trajectories"),
        "kinematics": os.path.join(run_output_dir, "kinematics"),
        "zones": os.path.join(run_output_dir, "zones"),
        "associations": os.path.join(run_output_dir, "associations"),
        "reports": os.path.join(run_output_dir, "reports"),
        "videos": os.path.join(run_output_dir, "annotated_videos"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 2. Extract Video Properties dynamically
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]) if fourcc_int != 0 else "mp4v"
    cap.release()

    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Fallback only if container FPS is completely corrupt
    duration_s = total_frames / fps if fps > 0 else 0.0

    print(f"[Video Source] Resolution: {width}x{height} | FPS: {fps:.2f} | Frames: {total_frames} | Duration: {duration_s:.2f}s", flush=True)

    device = device_override or cfg.get("models", {}).get("device", "cpu")

    # 3. Initialize Detection Modules
    models_cfg = cfg.get("models", {})
    det_cfg = cfg.get("detection", {})
    w_det_cfg = det_cfg.get("weapon", {})
    p_det_cfg = det_cfg.get("person", {})

    weapon_detector = WeaponEnsembleDetector(
        weights_s=models_cfg.get("weapon_s", "weights/yolo26s_best.pt"),
        weights_m=models_cfg.get("weapon_m", "weights/yolo26m_best.pt"),
        model_weights=tuple(w_det_cfg.get("wbf_weights", [1.0, 1.1])),
        conf_thr=float(w_det_cfg.get("conf_threshold", 0.25)),
        iou_thr=float(w_det_cfg.get("iou_threshold", 0.55)),
        img_size=int(det_cfg.get("img_size", 640)),
        fusion_mode=w_det_cfg.get("fusion_mode", "wbf"),
        conf_type=w_det_cfg.get("conf_type", "avg"),
        device=device,
    )

    person_detector = PersonDetector(
        weights_path=models_cfg.get("person", "yolov8n.pt"),
        conf_thr=float(p_det_cfg.get("conf_threshold", 0.35)),
        img_size=int(det_cfg.get("img_size", 640)),
        device=device,
        coco_person_class_id=int(p_det_cfg.get("coco_person_class_id", 0)),
    )

    # 4. Initialize Tracker
    tr_cfg = cfg.get("tracking", {})
    w_tr = tr_cfg.get("weapon", {})
    p_tr = tr_cfg.get("person", {})

    dual_tracker = DualStreamTracker(
        weapon_activation_thr=float(w_tr.get("track_activation_threshold", 0.25)),
        weapon_lost_buffer=int(w_tr.get("lost_track_buffer", 30)),
        weapon_matching_thr=float(w_tr.get("minimum_matching_threshold", 0.8)),
        person_activation_thr=float(p_tr.get("track_activation_threshold", 0.30)),
        person_lost_buffer=int(p_tr.get("lost_track_buffer", 30)),
        person_matching_thr=float(p_tr.get("minimum_matching_threshold", 0.75)),
        weapon_class_names=weapon_detector.class_names,
    )

    # 5. Initialize Association & Zone Engines
    assoc_cfg = cfg.get("association", {})
    associator = SpatioTemporalAssociator(
        iow_threshold=float(assoc_cfg.get("iow_threshold", 0.15)),
        proximity_distance_px=float(assoc_cfg.get("proximity_distance_px", 120.0)),
        temporal_window_frames=int(assoc_cfg.get("temporal_window_frames", 5)),
        min_association_score=float(assoc_cfg.get("min_association_score", 0.35)),
    )

    an_cfg = cfg.get("analytics", {})
    zone_engine = ZoneDwellEngine(
        zones_config=cfg.get("zones", {}),
        frame_width=width,
        frame_height=height,
        dwell_radius_px=float(an_cfg.get("dwell_radius_px", 50.0)),
        minimum_dwell_seconds=float(an_cfg.get("minimum_dwell_seconds", 1.0)),
        stationary_threshold_px_s=float(an_cfg.get("stationary_velocity_threshold_px_s", 15.0)),
    )

    kinematics_engine = KinematicsEngine(
        stationary_threshold_px_s=float(an_cfg.get("stationary_velocity_threshold_px_s", 15.0)),
        min_segment_length=int(an_cfg.get("min_segment_length", 5)),
        max_continuous_gap=int(an_cfg.get("max_continuous_frame_gap", 1)),
    )

    # 6. Initialize Video Writer if enabled
    should_save_video = save_video if save_video is not None else cfg.get("output", {}).get("save_annotated_video", True)
    video_writer = None
    annotated_video_path = os.path.join(dirs["videos"], f"{video_stem}_tracked.mp4")

    if should_save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
        visualizer = VideoVisualizer(zones_dict=zone_engine.zone_polygons)
    else:
        visualizer = None

    # 7. Main Frame Loop
    cap = cv2.VideoCapture(source_path)
    frame_idx = 0
    video_name = Path(source_path).name

    all_observations: List[TrackObservation] = []
    all_associations: List[AssociationRecord] = []

    print(f"\n[Processing] Starting frame-by-frame analysis on {video_name}...", flush=True)

    loop_start_time = time.time()
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_idx >= max_frames:
            break

        timestamp_s = frame_idx / fps

        # A. Detect Weapons via Ensemble (WBF)
        weapon_dets = weapon_detector.detect(frame)

        # B. Detect Persons via YOLOv8n (Class 0)
        person_dets = person_detector.detect(frame)

        # C. Dual Track Update
        frame_obs, tracked_w, tracked_p = dual_tracker.update(
            weapon_detections=weapon_dets,
            person_detections=person_dets,
            video_name=video_name,
            frame_id=frame_idx,
            timestamp_seconds=timestamp_s,
        )
        all_observations.extend(frame_obs)

        # Split frame observations by type for association & visualization
        w_obs_frame = [o for o in frame_obs if o.object_type == "weapon"]
        p_obs_frame = [o for o in frame_obs if o.object_type == "person"]

        # D. Spatio-Temporal Association
        frame_associations = associator.associate_frame(
            video_name=video_name,
            frame_id=frame_idx,
            timestamp_seconds=timestamp_s,
            weapon_obs_list=w_obs_frame,
            person_obs_list=p_obs_frame,
        )
        all_associations.extend(frame_associations)

        # E. Video Annotation
        if video_writer is not None and visualizer is not None:
            ann_frame = visualizer.annotate_frame(
                frame=frame,
                frame_id=frame_idx,
                timestamp_s=timestamp_s,
                weapon_obs=w_obs_frame,
                person_obs=p_obs_frame,
                associations=frame_associations,
            )
            video_writer.write(ann_frame)

        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == frames_to_process:
            elapsed = time.time() - loop_start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            pct = (frame_idx / frames_to_process) * 100 if frames_to_process > 0 else 100
            print(
                f"  [{video_stem}] Frame {frame_idx}/{frames_to_process} ({pct:.1f}%) | "
                f"{proc_fps:.1f} FPS | Weapons: {len(dual_tracker.weapon_class_history)} IDs | "
                f"Obs: {len(all_observations)}",
                flush=True,
            )

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"[Annotated Video] Saved to: {annotated_video_path}", flush=True)

    # 8. Post-Processing Analytics
    print("\n[Analytics] Computing trajectory segmentation, kinematics, zones, and dwell...", flush=True)

    segments, kinematics = kinematics_engine.compute_kinematics(all_observations, fps=fps)
    zone_events, dwell_episodes = zone_engine.analyze(all_observations, fps=fps)

    # 9. Write Structured Outputs

    # A. Raw Tracking CSV
    tracks_csv_path = os.path.join(dirs["tracking"], "tracks.csv")
    with open(tracks_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKING_CSV_COLUMNS)
        writer.writeheader()
        for obs in all_observations:
            writer.writerow(obs.to_dict())

    # B. Trajectory Segments CSV
    segments_csv_path = os.path.join(dirs["trajectories"], "trajectory_segments.csv")
    if segments:
        seg_fieldnames = list(asdict(segments[0]).keys())
    else:
        seg_fieldnames = ["segment_id", "video_name", "object_type", "track_id", "class_name",
                          "observations_count", "start_frame", "end_frame", "duration_seconds",
                          "is_valid_motion", "total_path_length_px", "displacement_px"]
    with open(segments_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=seg_fieldnames)
        writer.writeheader()
        for seg in segments:
            writer.writerow(asdict(seg))

    # C. Kinematics Observations CSV
    kinematics_csv_path = os.path.join(dirs["kinematics"], "kinematics_observations.csv")
    if kinematics:
        kin_fieldnames = list(kinematics[0].to_dict().keys())
    else:
        kin_fieldnames = ["video_name", "object_type", "track_id", "segment_id", "segment_classification",
                          "from_frame", "to_frame", "from_time_s", "to_time_s", "dt_s", "from_x", "from_y",
                          "to_x", "to_y", "dx_px", "dy_px", "step_distance_px", "vx_px_s", "vy_px_s",
                          "speed_px_s", "is_stationary", "heading_deg", "direction_8way", "turn_angle_deg",
                          "ax_px_s2", "ay_px_s2", "acceleration_px_s2"]
    with open(kinematics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=kin_fieldnames)
        writer.writeheader()
        for kin in kinematics:
            writer.writerow(kin.to_dict())

    # D. Zone Events & Dwell CSV
    zone_events_csv_path = os.path.join(dirs["zones"], "zone_events.csv")
    with open(zone_events_csv_path, "w", newline="", encoding="utf-8") as f:
        z_fields = ["video_name", "object_type", "track_id", "from_zone", "to_zone", "event", "frame_id", "timestamp_seconds", "dwell_seconds"]
        writer = csv.DictWriter(f, fieldnames=z_fields)
        writer.writeheader()
        for ze in zone_events:
            writer.writerow(ze.to_dict())

    dwell_csv_path = os.path.join(dirs["zones"], "dwell_episodes.csv")
    with open(dwell_csv_path, "w", newline="", encoding="utf-8") as f:
        d_fields = ["video_name", "object_type", "track_id", "dwell_type", "zone_name", "start_frame", "end_frame", "start_time_s", "end_time_s", "duration_seconds", "anchor_x", "anchor_y"]
        writer = csv.DictWriter(f, fieldnames=d_fields)
        writer.writeheader()
        for de in dwell_episodes:
            writer.writerow(de.to_dict())

    # E. Associations CSV
    assoc_csv_path = os.path.join(dirs["associations"], "person_weapon_associations.csv")
    with open(assoc_csv_path, "w", newline="", encoding="utf-8") as f:
        a_fields = ["video_name", "frame_id", "timestamp_seconds", "weapon_track_id", "person_track_id", "association_score", "association_method", "association_status", "iow", "distance_px"]
        writer = csv.DictWriter(f, fieldnames=a_fields)
        writer.writeheader()
        for a in all_associations:
            writer.writerow(a.to_dict())

    # F. Kinematic Timeline Plot
    plot_path = os.path.join(dirs["reports"], f"{video_stem}_kinematics_timeline.png")
    VideoVisualizer.generate_kinematic_plots(kinematics, plot_path, video_name=video_name)

    # 10. Research Provenance & Summary Metadata
    weapon_tracks = {o.track_id for o in all_observations if o.object_type == "weapon"}
    person_tracks = {o.track_id for o in all_observations if o.object_type == "person"}
    associated_pairs = {
        (a.weapon_track_id, a.person_track_id)
        for a in all_associations
        if a.association_status == "ASSOCIATED" and a.person_track_id is not None
    }

    # Summary metrics
    summary = {
        "run_id": run_id,
        "video_name": video_name,
        "video_properties": {
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "duration_seconds": duration_s,
            "codec": codec,
        },
        "tracking_metrics": {
            "total_observations": len(all_observations),
            "weapon_observations": sum(1 for o in all_observations if o.object_type == "weapon"),
            "person_observations": sum(1 for o in all_observations if o.object_type == "person"),
            "unique_weapon_tracks": len(weapon_tracks),
            "unique_person_tracks": len(person_tracks),
            "weapon_track_ids": sorted(list(weapon_tracks)),
            "person_track_ids": sorted(list(person_tracks)),
        },
        "kinematics_metrics": {
            "total_trajectory_segments": len(segments),
            "valid_motion_segments": sum(1 for s in segments if s.is_valid_motion),
            "insufficient_depth_segments": sum(1 for s in segments if not s.is_valid_motion),
            "total_kinematic_observations": len(kinematics),
            "acceleration_observations": sum(1 for k in kinematics if k.acceleration_px_s2 is not None),
        },
        "zone_and_dwell_metrics": {
            "configured_zones_count": len(zone_engine.zone_polygons),
            "zone_events_count": len(zone_events),
            "dwell_episodes_count": len(dwell_episodes),
        },
        "association_metrics": {
            "total_association_checks": len(all_associations),
            "successful_associations": sum(1 for a in all_associations if a.association_status == "ASSOCIATED"),
            "unique_associated_weapon_person_pairs": len(associated_pairs),
        },
        "execution": {
            "wall_clock_seconds": round(time.time() - start_time_wall, 2),
            "processing_fps": round(frame_idx / (time.time() - start_time_wall), 2) if (time.time() - start_time_wall) > 0 else 0,
        },
    }

    summary_json_path = os.path.join(dirs["reports"], "run_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Full provenance metadata for research reproducibility
    provenance = {
        "run_id": run_id,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": {
            "path": os.path.abspath(source_path),
            "filename": video_name,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "total_frames": total_frames,
            "processed_frames": frame_idx,
            "duration_s": duration_s,
        },
        "models": {
            "weapon_ensemble": weapon_detector.get_provenance(),
            "person_detector": person_detector.get_provenance(),
        },
        "tracker_config": tr_cfg,
        "analytics_config": an_cfg,
        "zones_config": cfg.get("zones", {}),
        "association_config": assoc_cfg,
        "environment": {
            "python_version": sys.version,
            "torch_version": getattr(sys.modules.get("torch"), "__version__", "unknown"),
            "supervision_version": sv.__version__,
            "opencv_version": cv2.__version__,
        },
        "outputs": {
            "tracking_csv": os.path.abspath(tracks_csv_path),
            "segments_csv": os.path.abspath(segments_csv_path),
            "kinematics_csv": os.path.abspath(kinematics_csv_path),
            "zone_events_csv": os.path.abspath(zone_events_csv_path),
            "dwell_csv": os.path.abspath(dwell_csv_path),
            "associations_csv": os.path.abspath(assoc_csv_path),
            "annotated_video": os.path.abspath(annotated_video_path) if should_save_video else None,
            "summary_json": os.path.abspath(summary_json_path),
        },
    }

    metadata_json_path = os.path.join(dirs["reports"], "run_metadata.json")
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    print("\n" + "=" * 70, flush=True)
    print("  PHASE 2 GENERIC PIPELINE RUN COMPLETE", flush=True)
    print(f"  Processed Frames: {frame_idx}/{total_frames} ({summary['execution']['processing_fps']} FPS)", flush=True)
    print(f"  Weapon Tracks: {len(weapon_tracks)} | Person Tracks: {len(person_tracks)}", flush=True)
    print(f"  Trajectory Segments: {len(segments)} ({summary['kinematics_metrics']['valid_motion_segments']} Valid N>=5)", flush=True)
    print(f"  Kinematics Steps: {len(kinematics)} ({summary['kinematics_metrics']['acceleration_observations']} Acceleration)", flush=True)
    print(f"  Associated Pairs: {len(associated_pairs)}", flush=True)
    print(f"  Outputs saved to: {run_output_dir}", flush=True)
    print("=" * 70 + "\n", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="S.W.O.R.D. v2 Persistent Video Tracking Pipeline")
    parser.add_argument("--source", required=True, type=str, help="Path to input video file (e.g. video.mp4)")
    parser.add_argument("--config", default=None, type=str, help="Path to config YAML (default: persistent_video_tracking/config/tracking_config.yaml)")
    parser.add_argument("--output_dir", default=None, type=str, help="Base directory for outputs (default: persistent_video_tracking/outputs)")
    parser.add_argument("--device", default=None, type=str, help="Device override ('cpu' or 'cuda')")
    parser.add_argument("--no_video", action="store_true", help="Disable rendering annotated video")
    parser.add_argument("--max_frames", default=None, type=int, help="Optional frame limit for fast testing")
    args = parser.parse_args()

    run_pipeline(
        source_path=args.source,
        config_path=args.config,
        output_dir=args.output_dir,
        device_override=args.device,
        save_video=not args.no_video,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
