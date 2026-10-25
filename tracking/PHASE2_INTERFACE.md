# Phase 2: how to plug your tracker in

Phases 1, 3, 4, 5 and 6 are already connected end to end. Phase 2 (persistent person tracking and trajectory
features) is a slot. A temporary stand-in fills it for now: COCO YOLO26n, ByteTrack and `tracking/trajectory.py`.
Your module replaces that stand-in. Nothing else in the system has to change.

## The contract

Your tracker needs two methods:

```python
class MyTracker:
    def update(self, frame_bgr, t):
        """frame_bgr: HxWx3 uint8 (OpenCV BGR); t: seconds since stream start.
        Return (people, zone_events)."""
        people = [{
            "track_id": 3,                 # int, persistent across frames (required)
            "bbox": [x1, y1, x2, y2],      # pixels in this frame (required)
            "score": 0.91,                 # detection confidence (required)
            "trajectory": {                # optional; every key is optional
                "speed_heights_s": 0.4,    # speed in body heights per second -> "running" evidence
                "zone": "entrance",        # zone name or None -> "restricted zone" evidence
                "velocity_px_s": [vx, vy], "acceleration_px_s2": [ax, ay],
                "heading_deg": 90.0, "dwell_s": 3.2,
                "speed_m_s": 1.3,          # only with a calibrated homography
            },
        }]
        zone_events = [{"type": "zone_enter", "track_id": 3, "zone": "entrance", "timestamp": t}]
        return people, zone_events

    def reset(self):
        """Called at the start of every new video or stream."""
```

The full specification is in [interface.py](interface.py), as the `Phase2Tracker` protocol.

## What the other phases use from you

| Field | Used by | Effect if missing or poor |
|---|---|---|
| `track_id` | Phase 3 wrist velocity, Phase 4 per-person 3 s skeleton windows, Phase 5 per-person evidence and persistence | ID switches reset a person's evidence and action history; fragmented tracks never reach Phase 4's 1.5 s minimum window |
| `bbox` | Phase 3 top-down pose (RTMPose-m runs inside your boxes), weapon association, plausibility prior | Loose or missed boxes mean bad or missing keypoints |
| `trajectory.speed_heights_s` | Phase 5 "running" channel (ELEVATED) | Channel stays 0 |
| `trajectory.zone` | Phase 5 restricted-zone dwell (ELEVATED) | Channel stays 0 |

## Using it

```python
from pipeline.system import SwordSystem
system = SwordSystem(fps=25, phase2=MyTracker(...))
rec = system.process(frame, frame_idx, t)   # Phases 1-5 on one frame
```

The service (`python -m service`) and the research runner (`research/tools/integration/run_full_pipeline.py`)
build the system with `SwordSystem(...)`. Pass `phase2=` there too, or change the default in
`tracking/pipeline.py`.

## Check it before plugging it in

```bash
python -m tracking.contract --tracker mypackage.tracker:MyTracker --kwargs '{"fps": 10}' --video research/data/videos/vtest.avi
```

This reports every contract violation (types, box sanity, duplicate IDs in one frame, zone events) and basic
tracking statistics: track count, median track length, single-frame tracks, tracks with gaps, latency.

## What the tests found about the stand-in

The mock-attack CCTV test set is sampled at **2 fps**. Measured on all 20 segments (default full-pipeline run,
`diagnostics_mockattack` in `research/runs/*_full-pipeline-default-yolo26s-640/results.json`):
- the stand-in keeps person tracks for a median of 2–11.5 s per segment;
- on average 8% of tracks last only one frame;
- every tracked person got a pose.

ID switches can't be measured, because the dataset has no identity ground truth. People are large enough that IoU
matching mostly holds at 2 fps. Small hand-held weapons do not hold: their median IoU between consecutive frames is
0.08, so weapon tracks are no longer used to filter detections.

Phase 4 needs at least 1.5 s of continuous track per person. A Phase 2 module that keeps identities through
occlusions and crossings (appearance features, better motion models) gives Phases 4–5 longer histories. Compare runs
before and after with `research/tools/integration/run_full_pipeline.py`.
