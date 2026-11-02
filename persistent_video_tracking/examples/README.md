# Examples & Testing Assets

This directory contains utility scripts and test assets for evaluating **Persistent Video Tracking**.

## Generating a Synthetic Test Video

You can generate a self-contained synthetic video (no external datasets or images required) using:

```bash
python persistent_video_tracking/examples/create_test_video.py
```

This creates `sample_test_video.mp4` (1280x720 @ 20 FPS, 60 frames) featuring predictable sinusoidal horizontal motion for rapid smoke testing and regression verification.

## Running the Pipeline on the Example Video

```bash
python -m persistent_video_tracking.pipeline.run_generic_pipeline \
  --input persistent_video_tracking/examples/sample_test_video.mp4 \
  --output_dir persistent_video_tracking/outputs
```

## Validating Output

```bash
python -m persistent_video_tracking.pipeline.validate_pipeline \
  --run_dir persistent_video_tracking/outputs/sample_test_video
```
