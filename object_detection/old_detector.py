import torch
import yaml
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load configuration
with open("configs/config_720p.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["model"]["path"]
CONFIDENCE_THRESHOLD = config["model"]["confidence_threshold"]


class ObjectDetector:
    def __init__(self):
        """Initialize YOLO model and RealSense camera"""
        # self.model = YOLO(MODEL_PATH)
        self.model = YOLO("./runs/detect/train6/weights/best.pt")

        # Initialize RealSense camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(config)

        # Get depth scale
        self.align = rs.align(rs.stream.color)
        self.depth_scale = self.get_depth_scale()

    def get_depth_scale(self):
        """Retrieve the depth scale from the RealSense sensor"""
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        return depth_sensor.get_depth_scale()

    def get_frames(self):
        """Retrieve synchronized RGB and depth frames"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        return color_image, depth_image

    def detect_objects(self, image, depth_frame):
        """Detect objects and return bounding boxes with real-world coordinates"""
        results = self.model(image)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if confidence > CONFIDENCE_THRESHOLD:
                # Get depth at center of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                depth_value = depth_frame[center_y, center_x] * self.depth_scale  # Convert to meters

                detections.append({
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "confidence": confidence,
                    "class_id": class_id,
                    "depth": depth_value,
                    "center": (center_x, center_y)
                })

        return detections