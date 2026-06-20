"""
detect.py — YOLOv8 inference wrapper for Gridlock
  Wraps Ultralytics YOLO with a clean Detection dataclass output.
  Supports image, video frame, and webcam inference.
"""

from __future__ import annotations
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from ultralytics import YOLO


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single detected object in a frame."""
    cls: str              # class name
    conf: float           # confidence score
    bbox: List[float]     # [x1, y1, x2, y2] in pixel coords
    track_id: int = -1    # assigned by tracker (-1 = untracked)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


# ─── Detector ─────────────────────────────────────────────────────────────────

class GridlockDetector:
    """
    YOLOv8 inference wrapper.

    Usage:
        detector = GridlockDetector("models/weights/best.pt")
        detections = detector.predict(frame)
    """

    CLASS_NAMES = [
        "motorcycle", "car", "bus", "truck",
        "person", "helmet", "no_helmet"
    ]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.device = device

        model_path = Path(model_path)
        if not model_path.exists() and str(model_path) != "yolov8n.pt":
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Train first with: python src/train.py"
            )

        print(f"[GridlockDetector] Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        self.model.to(device)
        print(f"[GridlockDetector] Ready on device={device}")

    def predict(self, frame: np.ndarray, track: bool = False) -> List[Detection]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: numpy BGR image (H x W x 3)
            track: if True, use ByteTrack for ID persistence across frames
        Returns:
            List[Detection]
        """
        if track:
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )

        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls_idx = int(boxes.cls[i].item())
                cls_name = self.model.names.get(cls_idx, str(cls_idx))
                conf = float(boxes.conf[i].item())
                bbox = boxes.xyxy[i].tolist()
                track_id = int(boxes.id[i].item()) if (track and boxes.id is not None) else -1
                detections.append(Detection(
                    cls=cls_name,
                    conf=conf,
                    bbox=bbox,
                    track_id=track_id,
                ))

        return detections

    def predict_image(self, image_path: str) -> tuple[np.ndarray, List[Detection]]:
        """Load image from path, run prediction, return (frame, detections)."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return frame, self.predict(frame)

    def stream_video(self, video_path: str, callback=None):
        """
        Generator that yields (frame_id, frame, detections) for each video frame.
        Optionally calls callback(frame_id, frame, detections) per frame.
        
        Usage:
            for fid, frame, dets in detector.stream_video("traffic.mp4"):
                violations = checker.check(dets, fid)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_id = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                detections = self.predict(frame, track=True)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if callback:
                    callback(frame_id, frame, detections, timestamp)

                yield frame_id, frame, detections, timestamp
                frame_id += 1
        finally:
            cap.release()


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    # Ensure project root is on path so `src.*` imports resolve
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    path = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
    detector = GridlockDetector(path)
    print("Detector loaded. Pass an image path as argument to test.")
    if len(sys.argv) > 2:
        frame, dets = detector.predict_image(sys.argv[2])
        for d in dets:
            print(f"  {d.cls:12s}  conf={d.conf:.2f}  bbox={[round(x) for x in d.bbox]}")
        print(f"Total detections: {len(dets)}")
