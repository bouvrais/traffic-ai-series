"""
Step 2: Video Processing with Vehicle Tracking and Speed Calculation
Processes a video file to track vehicles and calculate their speeds using perspective transformation.
"""

import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np

# Initialize the tracker and model
tracker = sv.ByteTrack()
model = YOLO('yolov8n.pt')

# Load video file
video_path = "capture.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

print(f"Processing video: {video_path}")

# Perspective transformation setup
src_points = np.float32([[0, 724], [0, 605], [2005, 684], [2191, 747]])
dst_points = np.float32([[0, 0], [8.7, 0], [8.7, 40.55], [0, 40.55]])
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

def pixel_to_world(x, y):
    pixel_point = np.array([[[x, y]]], dtype=np.float32)
    world_point = cv2.perspectiveTransform(pixel_point, transform_matrix)
    return world_point[0][0][0], world_point[0][0][1]

# Store vehicle positions for speed calculation
vehicle_positions = {}

# Get video FPS for accurate timing
video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_time_interval = 1.0 / video_fps
frame_count = 0

print(f"Video FPS: {video_fps}")

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_count * frame_time_interval
    frame_count += 1

    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 2]  # cars only
    detections = tracker.update_with_detections(detections)

    for i, tracker_id in enumerate(detections.tracker_id):
        x_center = (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2
        y_center = (detections.xyxy[i][1] + detections.xyxy[i][3]) / 2
        world_x, world_y = pixel_to_world(x_center, y_center)

        if tracker_id in vehicle_positions:
            prev_x, prev_y, prev_time = vehicle_positions[tracker_id]
            distance = np.sqrt((world_x - prev_x)**2 + (world_y - prev_y)**2)
            time_diff = current_time - prev_time

            if time_diff > 0:
                speed_kmh = (distance / time_diff) * 3.6
                print(f"Vehicle {tracker_id}: {speed_kmh:.1f} km/h")

        vehicle_positions[tracker_id] = (world_x, world_y, current_time)

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Video processing completed.")