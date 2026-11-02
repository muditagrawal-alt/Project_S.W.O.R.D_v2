# Persistent Video Tracking — Documentation and Verification Audit

* **Subsystem**: Persistent Video Tracking (S.W.O.R.D. v2)
* **Target Documentation**: [`persistent_video_tracking/README.md`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/README.md)
* **Date of Verification**: 2026-11-02
* **Auditor**: Senior Computer Vision & AI Engineer (Antigravity AI)
* **Repository Root**: `D:\SWORD_v2_COMPLETE_PROJECT`
* **Python Runtime**: Python 3.11.6 (x86_64 Windows PowerShell)

---

## 1. Executive Summary

This audit records the complete technical review, source code inspection, empirical command verification, and rewrite of the authoritative technical documentation for the **Persistent Video Tracking** subsystem in S.W.O.R.D. v2 ([`persistent_video_tracking/README.md`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/README.md)).

All 21 recommended documentation sections have been systematically authored, cross-referenced against actual Python implementations, and validated against the repository's test suites and live execution outputs. Obsolete directory names (`phase2/`), inaccurate CLI parameters (`--input`), and unverified claims have been eliminated.

---

## 2. Sections Added or Rewritten

The documentation was structured into 21 exhaustive sections:

| Section # | Section Title | Description of Contents & Verification |
| :---: | :--- | :--- |
| **1** | Project Title & Overview | Defines the subsystem within S.W.O.R.D. v2, its role bridging frame-level detection and temporal threat reasoning, technologies, and lead architect attribution. |
| **2** | Purpose & Scope | Explains why multi-object tracking is essential for CCTV analytics; documents operational scope across stationary cameras, dual tracking, WBF, footpoints, and gap-aware kinematics. |
| **3** | Key Features | Catalogs 10 verified architectural features: decoupled dual-stream tracking, WBF consensus, ground footpoints, 15-frame class smoother, gap-aware segmentation, 8-way heading, zone dwell, bipartite association, 14-point validator, and Python adapter. |
| **4** | System Architecture | Contains **Module Architecture Diagram** (`graph TD`) and **Track Data Flow Diagram** (`flowchart TD`), detailing relationships between detection, tracking, analytics, association, and visualization. |
| **5** | End-to-End Pipeline Flowchart | Contains comprehensive 18-step **End-to-End Pipeline Flowchart** (`flowchart TD`) mapping video ingestion to structured output generation. |
| **6** | Module & Directory Structure | Complete directory tree isolating core runtime code, tests, examples, verified benchmark outputs, historical research scripts, and external model weights. |
| **7** | Technical Workflow | 18-step technical matrix detailing component, purpose, exact input, exact output, source file with lines, downstream consumer, and implementation status for all 18 processing steps. |
| **8** | Mathematical Foundations | Rigorous mathematical formulas for centroids, ground footpoint $(x_{\text{mid}}, y_2)$, WBF coordinate consensus, gap-aware differentials ($N \to N-1 \to N-2$), 8-way compass sector bounds, point-in-polygon containment, and IoW/proximity association. Explicitly documents image-space pixel units and homography decoupling. |
| **9** | Installation & Environment Setup | Exact Windows PowerShell commands for virtual environment creation, activation, pip upgrades, and installation of pinned dependencies from `requirements.txt`. |
| **10** | Model Weights & Configuration | Documents expected checkpoint locations (`weights/yolo26s_best.pt`, `weights/yolo26m_best.pt`, `yolov8n.pt`), sizes, resolution logic in `resolve_model_path()`, and fallback behavior. |
| **11** | Running the Pipeline | Complete CLI reference for [`persistent_video_tracking.pipeline.run_generic_pipeline`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/pipeline/run_generic_pipeline.py), documenting verified arguments (`--source`, `--config`, `--output_dir`, `--device`, `--no_video`, `--max_frames`), sample execution, headless mode, and expected console logs. |
| **12** | Custom Video Execution | Guide for arbitrary external CCTV MP4 videos and execution instructions for the synthetic video generator ([`examples/create_test_video.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/examples/create_test_video.py)). |
| **13** | Generated Outputs & Data Schemas | Full column-level schemas, types, units, and interpretation notes for all 9 output artifacts: `tracks.csv`, `trajectory_segments.csv`, `kinematics_observations.csv`, `zone_events.csv`, `dwell_episodes.csv`, `person_weapon_associations.csv`, `run_summary.json`, `run_metadata.json`, and annotated media. |
| **14** | Configuration Reference | Master parameters table documenting parameter names, default values, purposes, and effect of increasing/decreasing values in `config/tracking_config.yaml`. Includes copy-paste example for defining custom security zones. |
| **15** | Testing & Validation | Documents Tier 1 unit tests (12 tests) and Tier 2 14-point automated validator. Details all 14 invariant definitions and presents empirical results across 4 benchmark directories (56/56 checks passed). Distinguishes invariant validation from MOTA/IDF1 metrics. |
| **16** | Integration with S.W.O.R.D. Components | Architectural integration guide covering Detection, Human Pose Estimation (Phase 3), Action Recognition (Phase 4), Threat Scoring (Phase 5), and Dashboard. Includes **Integration Architecture Flowchart** (`flowchart LR`) and **Integration Contract Table**. |
| **17** | Public Python Integration API | Detailed, working Python integration example importing [`PersistentVideoTracker`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/tracking/adapter.py) and [`FrameTrackingResult`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/tracking/adapter.py), showing synchronous frame-by-frame processing and finalization. |
| **18** | Troubleshooting | Comprehensive diagnostic table mapping 8 common issues (`ModuleNotFoundError`, missing weights, OpenCV errors, CUDA fallback, column name typos, velocity spikes, empty zone events, label flicker) to causes and corrective actions. |
| **19** | Performance Considerations | Documents real-world CPU throughput ($2.15 - 2.25\text{ FPS}$ on standard laptop CPU), headless speedup, resolution scaling, and resident memory footprint ($< 1.5\text{ GB}$). |
| **20** | Limitations & Future Improvements | Explicitly details pixel-space limitations, detection dependency, and geometric association ethics. Outlines planned improvements (OSNet ReID, interactive 4-point homography calibration, TensorRT export). |
| **21** | Developer Notes & Guidelines | Author attribution, capstone context, mandatory regression commands, and Git contribution constraints. |

---

## 3. Source Files Inspected

Every documented feature was verified against source code:

1. **Configuration**:
   * [`persistent_video_tracking/config/tracking_config.yaml`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/config/tracking_config.yaml): Confirmed model paths, detection thresholds (`conf=0.25`, `iou=0.55`), tracker parameters (`buffer=30`, `matching=0.80/0.75`), kinematics settings (`stationary=15.0 px/s`, `min_seg=5`), and association thresholds (`iow=0.15`, `prox=120px`, `window=5`).
   * [`persistent_video_tracking/config/test_zone_config.yaml`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/config/test_zone_config.yaml): Confirmed multi-polygon zone coordinates (`zone_alpha`, `zone_beta`).
2. **Detection Subsystem**:
   * [`persistent_video_tracking/detection/weapon_ensemble.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/detection/weapon_ensemble.py): Confirmed YOLO26s + YOLO26m parallel inference, dynamic model path resolution, device fallback, and 10 custom weapon class mappings.
   * [`persistent_video_tracking/detection/person_detector.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/detection/person_detector.py): Confirmed YOLOv8n inference filtered strictly for COCO class 0 (`person`) and footpoint computation $(x_{\text{mid}}, y_2)$.
   * [`persistent_video_tracking/detection/wbf.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/detection/wbf.py): Confirmed confidence-weighted consensus box fusion at IoU threshold $0.55$.
3. **Tracking Subsystem**:
   * [`persistent_video_tracking/tracking/dual_tracker.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/tracking/dual_tracker.py): Confirmed two isolated `supervision.ByteTrack` instances, independent track ID counters, and 15-frame rolling majority vote class smoother.
   * [`persistent_video_tracking/tracking/schema.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/tracking/schema.py): Confirmed canonical `TrackObservation` dataclass fields and column lists.
   * [`persistent_video_tracking/tracking/adapter.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/tracking/adapter.py): Confirmed public integration adapter class `PersistentVideoTracker` and dataclass `FrameTrackingResult`.
4. **Analytics Subsystem**:
   * [`persistent_video_tracking/analytics/kinematics.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/analytics/kinematics.py): Confirmed gap-aware trajectory partitioner (splitting when $\Delta f > 1$), $N-1$ velocity rule, $N-2$ acceleration rule, 8-way compass sector binning, and $15.0\text{ px/s}$ stationary threshold.
   * [`persistent_video_tracking/analytics/zones_dwell.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/analytics/zones_dwell.py): Confirmed `cv2.pointPolygonTest` spatial containment, `ZONE_ENTRY`/`ZONE_EXIT`/`ZONE_TRANSITION` state machine, and strictly non-negative dwell times.
5. **Association Subsystem**:
   * [`persistent_video_tracking/association/person_weapon_matcher.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/association/person_weapon_matcher.py): Confirmed IoW ($\ge 0.15$), Euclidean proximity ($\le 120\text{ px}$), 5-frame sliding window voting, and preservation of raw immutable track IDs.
6. **Visualization Subsystem**:
   * [`persistent_video_tracking/visualization/visualizer.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/visualization/visualizer.py): Confirmed bounding box rendering, ID badges (`#W1`, `#P1`), 30-frame breadcrumb motion trails, dashed association links, zone overlays, and Matplotlib timeseries timeline plot generation.
7. **Pipeline Orchestration & Validation**:
   * [`persistent_video_tracking/pipeline/run_generic_pipeline.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/pipeline/run_generic_pipeline.py): Confirmed CLI argument parser (`--source`, `--config`, `--output_dir`, `--device`, `--no_video`, `--max_frames`), dynamic FPS probing, output generation, and metadata export.
   * [`persistent_video_tracking/pipeline/validate_pipeline.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/pipeline/validate_pipeline.py): Confirmed all 14 automated invariant check functions.
   * [`persistent_video_tracking/pipeline/test_modules.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/pipeline/test_modules.py) and [`persistent_video_tracking/tests/test_modules.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/tests/test_modules.py): Confirmed unit tests for IoW, trajectory segmentation, association, and zone logic.
8. **Utilities & Dependencies**:
   * [`persistent_video_tracking/examples/create_test_video.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/examples/create_test_video.py): Confirmed synthetic video generation.
   * [`persistent_video_tracking/requirements.txt`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/requirements.txt): Confirmed pinned versions.

---

## 4. Commands Verified

The following Windows PowerShell commands were executed and verified directly in the environment:

### A. Unit Test Suite Execution
```powershell
python -m unittest persistent_video_tracking.tests.test_modules persistent_video_tracking.pipeline.test_modules -v
```
* **Result**: `Ran 12 tests in 0.001s` — **OK (12/12 passed)**

### B. 14-Point Automated Validator Execution
```powershell
python -m persistent_video_tracking.pipeline.validate_pipeline --run_dir persistent_video_tracking/outputs/sample_test_video
```
* **Result**: `Summary: 14/14 checks passed` — **PASS**

### C. Multi-Benchmark Invariant Validation
Validated all 4 benchmark run outputs across all 14 invariants:
* `outputs/sample_video_01`: 14 / 14 Passed
* `outputs/sample_video_02`: 14 / 14 Passed
* `outputs/sample_video_03`: 14 / 14 Passed
* `outputs/sample_test_video`: 14 / 14 Passed
* **Cumulative**: **56 / 56 Invariant Checks Passed (100% PASS)**

### D. Pipeline Execution on Synthetic Fixture
```powershell
python -m persistent_video_tracking.pipeline.run_generic_pipeline `
    --source persistent_video_tracking/sample_test_video.mp4 `
    --output_dir persistent_video_tracking/outputs/sample_test_video `
    --max_frames 40
```
* **Result**: Completed in 18.5s (~2.2 FPS on CPU), generating full set of 9 output artifacts.

### E. Synthetic Video Generator
```powershell
python persistent_video_tracking/examples/create_test_video.py
```
* **Result**: Successfully generated 60-frame $1280 \times 720$ MP4 video fixture.

---

## 5. Links and Diagrams Checked

* **Markdown Hyperlinks**: Automated Python link validation verified **36 clickable `file://` links** across the document. **0 broken links** detected.
* **Mermaid Diagrams**: All 4 Mermaid code blocks validated for GitHub rendering compatibility:
  1. Module Architecture Diagram: `graph TD` (Section 4)
  2. Track Data Flow Diagram: `flowchart TD` (Section 4)
  3. End-to-End Pipeline Flowchart: `flowchart TD` (Section 5)
  4. Integration Architecture Flowchart: `flowchart LR` (Section 16)
* **Obsolete Reference Audit**: Automated grep confirmed **0 occurrences of `phase2`** across `persistent_video_tracking/README.md`.

---

## 6. Unverified or Pending Integration Points

The documentation clearly and honestly distinguishes implemented components from pending integrations:

| Component | Interface Status | Realization Status | Details |
| :--- | :---: | :---: | :--- |
| **Video Ingestion & Probing** | Implemented | Verified | Dynamic OpenCV metadata extraction (`fps`, `width`, `height`, `frames`). |
| **YOLO26s + YOLO26m WBF Ensemble** | Implemented | Verified | Dual model inference and consensus clustering. |
| **YOLOv8n Person Detector** | Implemented | Verified | Class 0 filtering and ground footpoint extraction. |
| **Dual ByteTrack Tracking** | Implemented | Verified | Segregated ID namespaces for `#W` and `#P`. |
| **Gap-Aware Motion Kinematics** | Implemented | Verified | Splitting on $\Delta f > 1$; $N-1$ speeds, $N-2$ accelerations. |
| **Security Zones & Dwell Analytics**| Implemented | Verified | Polygon containment and non-negative dwell times. |
| **Person–Weapon Association** | Implemented | Verified | Geometric IoW and Euclidean proximity with rolling vote. |
| **Public Python API Adapter** | Implemented | Verified | `PersistentVideoTracker` and `FrameTrackingResult`. |
| **Automated Invariant Validator** | Implemented | Verified | 14 invariant check suite. |
| **Human Pose Estimation (Phase 3)** | Interface Defined | **Integration Pending** | Adapter provides person bounding box crops; pose model pipeline not yet wired. |
| **Action Recognition (Phase 4)** | Interface Defined | **Integration Pending** | Continuous trajectory segments and velocity vectors exported; temporal classifier not yet wired. |
| **Threat Scoring Engine (Phase 5)** | Interface Defined | **Integration Pending** | Associations, dwell, and kinematics exported; Bayesian/rule threat fusion engine not yet wired. |
| **Streamlit Dashboard (`app.py`)** | Adapter Ready | **UI Wiring Pending** | Tracker adapter ready for integration into central S.W.O.R.D. Streamlit application. |

---

## 7. Implementation Gaps & Clarifications Resolved in Documentation

1. **CLI Flag Consistency (`--source` vs `--input`)**:
   * *Finding*: Earlier documentation drafts referenced `--input`, but [`pipeline/run_generic_pipeline.py`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/pipeline/run_generic_pipeline.py#L499) implements `--source`.
   * *Resolution*: Documented `--source` across all CLI commands and options in the README.
2. **Trajectory CSV Column Naming (`observations_count`)**:
   * *Finding*: In [`trajectories/trajectory_segments.csv`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/outputs/sample_test_video/trajectories/trajectory_segments.csv), the column is named `observations_count` (plural), whereas some ad-hoc evaluation scripts used `observation_count` (singular).
   * *Resolution*: Formally documented `observations_count` in the output schema table in Section 13.
3. **Image-Space Coordinates vs Real-World Metric Speed**:
   * *Finding*: Kinematics computations yield pixel displacements ($\text{px/s}$ and $\text{px/s}^2$). Converting to real-world metric units ($\text{m/s}$) requires a 4-point planar homography matrix transformation, which is not built into the standalone 2D video tracker.
   * *Resolution*: Explicitly documented that velocities are in pixel space, warned against treating pixel speed as metres per second, and listed interactive homography calibration under planned improvements (Section 7, 8, 20).
4. **Hardware Acceleration Context**:
   * *Finding*: The developer environment operates on CPU (`torch.cuda.is_available() == False`), achieving ~2.15–2.25 FPS at 720p.
   * *Resolution*: Accurately documented CPU performance as the baseline, explained that CUDA is auto-selected when available, and avoided claiming 30+ FPS real-time performance on CPU.
5. **Ethical & Forensic Boundaries of Spatial Association**:
   * *Finding*: Geometric proximity between a weapon box and a person box does not prove intent, legal ownership, or active brandishing.
   * *Resolution*: Added a dedicated forensic alert in Section 8 and Section 13 emphasizing that spatio-temporal co-presence is a spatial feature requiring grip and pose confirmation downstream before raising high-priority security alarms.

---

## 8. Conclusion

[`persistent_video_tracking/README.md`](file:///d:/SWORD_v2_COMPLETE_PROJECT/persistent_video_tracking/README.md) is now completely finalized, technically precise, fully cross-referenced against the working Python source code, and verified by live unit tests and invariant validator executions. It provides an authoritative, developer-ready reference for the Persistent Video Tracking subsystem in S.W.O.R.D. v2.
