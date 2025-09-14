# Traffic Monitoring Scripts - Part 1

This directory contains the complete Python scripts referenced in the article "How My Gaming PC Accidentally Became a Real-Time City Traffic Monitor".

## Scripts Overview

### 1. Static Detection (`01_static_detection.py`)
- **Purpose**: Basic vehicle detection in a single image
- **Key Feature**: Uses YOLOv8 to detect and classify vehicles with confidence scores
- **Usage**: Good starting point to test if YOLO detection works on your images

### 2. Video Tracking (`02_video_tracking.py`)
- **Purpose**: Processes video files to track vehicles and calculate speeds
- **Key Features**:
  - Vehicle tracking across frames using ByteTrack
  - Perspective transformation for real-world distance calculations
  - Speed calculation using physics (distance/time)
- **Usage**: Process recorded video files for traffic analysis

### 3. Real-time RTSP (`03_realtime_rtsp.py`)
- **Purpose**: Live processing of RTSP camera streams
- **Key Features**:
  - Real-time stream processing
  - Performance monitoring (inference timing)
  - Optimized for continuous operation
- **Usage**: Connect to live camera feeds for real-time traffic monitoring

## Requirements

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. If you don't have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies and run the scripts:

```bash
# Install dependencies
uv sync

# Run any script
uv run python 01_static_detection.py
uv run python 02_video_tracking.py
uv run python 03_realtime_rtsp.py
```

## Configuration Notes

- **Perspective transformation points**: The `src_points` and `dst_points` arrays need to be calibrated for your specific camera setup and viewing angle
- **RTSP URL**: Update the `rtsp_url` variable with your camera's connection string
- **Model weights**: Scripts use YOLOv8 nano (`yolov8n.pt`) - first run will download the model automatically

## Getting Started

1. Start with `01_static_detection.py` to verify YOLO works with your setup
2. Move to `02_video_tracking.py` with a test video file
3. Configure perspective transformation points for your camera view
4. Deploy `03_realtime_rtsp.py` for live monitoring

## Important Setup Step

The perspective transformation is crucial for accurate speed calculations. You'll need to:
1. Identify 4 points in your camera view that form a rectangle on the ground
2. Measure the real-world distances of that rectangle
3. Update the `src_points` and `dst_points` accordingly

Without proper calibration, speed readings will be inaccurate.