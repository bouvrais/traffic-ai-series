"""
Step 3: Real-time RTSP Stream Processing with Speed Calculation
Processes a live RTSP stream to track vehicles and calculate their speeds in real-time.
"""

import cv2
import time
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque

# RTSP stream configuration
rtsp_url = "rtsp://username:password@your_rtsp_url"

# Initialize the tracker and model
tracker = sv.ByteTrack()
model = YOLO('yolov8n.pt')

# Perspective transformation setup
#   This is setup specific and depends on your camera and framing.
#   src_points are four arbitrary points in the image
#   dst_points are their corresponding siblings in the real world coordinate system
src_points = np.float32([[0, 724], [0, 605], [2005, 684], [2191, 747]])
dst_points = np.float32([[0, 0], [8.7, 0], [8.7, 40.55], [0, 40.55]])
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

def pixel_to_world(x, y):
    pixel_point = np.array([[[x, y]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(pixel_point, transform_matrix)
    return world_point[0][0][0], world_point[0][0][1]

# Store vehicle positions for speed calculation
vehicle_positions = {}

def process_rtsp_stream(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    # Configure the stream for optimal performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

    # Get stream FPS from the source
    stream_fps = cap.get(cv2.CAP_PROP_FPS)
    if stream_fps <= 0:
        print("Warning: Could not detect stream FPS, defaulting to 30 FPS")
        stream_fps = 30

    print(f"Detected stream FPS: {stream_fps}")

    # Frame-based timing for accurate speed calculations
    frame_time_interval = 1.0 / stream_fps
    frame_count = 0

    print(f"Starting RTSP stream processing at {stream_fps} FPS")

    while True:
        _, frame = cap.read()

        video_time = frame_count * frame_time_interval
        frame_count += 1

        # Run the detection inference
        inference_begin_time = time.time()
        results = model(frame, verbose=False)[0]
        inference_end_time = time.time()

        # Post-processing start
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 2]  # cars only
        detections = tracker.update_with_detections(detections)

        # Process vehicle tracking and speed calculations
        for i, tracker_id in enumerate(detections.tracker_id):
            x_center = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
            y_center = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
            world_x, world_y = pixel_to_world(x_center, y_center)

            if tracker_id in vehicle_positions:
                prev_x, prev_y, prev_time = vehicle_positions[tracker_id]
                distance = np.sqrt((world_x - prev_x)**2 + (world_y - prev_y)**2)
                time_diff = video_time - prev_time

                if time_diff > 0 and time_diff < 1.0: # If we have seen that vehicle before (and less than a second ago)
                    speed_kmh = (distance / time_diff) * 3.6

                    inference_time = (inference_end_time - inference_begin_time) * 1000

                    print(f"Vehicle {tracker_id}: {speed_kmh:.1f} km/h (inference: {inference_time:.1f}ms)")

                    # Only update position if we're tracking this vehicle (time_diff < 1.0)
                    vehicle_positions[tracker_id] = (world_x, world_y, video_time)
                else:
                    # Remove stale vehicles
                    del vehicle_positions[tracker_id]
            else:
                # New vehicle, store its position
                vehicle_positions[tracker_id] = (world_x, world_y, video_time)

if __name__ == "__main__":
    try:
        process_rtsp_stream(rtsp_url)
    except KeyboardInterrupt:
        print("\nStopping RTSP processing...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()