"""
Step 1: Static Vehicle Detection with YOLO
Basic script to detect vehicles in a single image using YOLOv8.
"""

from ultralytics import YOLO
import cv2

# Load the pre-trained model
model = YOLO('yolov8n.pt')  # 'n' for nano - fastest version

# Read a single frame from camera
frame = cv2.imread('test_frame.jpg')

# Run detection
results = model(frame, verbose=False)

# Extract detections for vehicles only
for result in results:
    boxes = result.boxes
    for box in boxes:
        if box.cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
            conf = box.conf.item()
            if conf > 0.5:  # Only confident detections
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = model.names[int(box.cls)]
                print(f"Found {class_name} with {conf:.2f} confidence at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")