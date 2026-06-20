"""
utils.py — Shared helper functions for Gridlock prototype
  - Drawing bounding boxes with violation labels
  - IoU / overlap computation
  - Logging violation records to CSV
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─── Color palette per class ─────────────────────────────────────────────────
CLASS_COLORS = {
    "motorcycle": (255, 165, 0),    # orange
    "car":        (0, 200, 255),    # cyan
    "bus":        (100, 100, 255),  # blue
    "truck":      (180, 50, 255),   # purple
    "person":     (50, 205, 50),    # lime
    "helmet":     (0, 255, 128),    # green
    "no_helmet":  (0, 0, 255),      # red
    "plate":      (255, 255, 0),    # yellow
    "violation":  (0, 0, 255),      # red highlight
}

VIOLATION_COLORS = {
    "NO_HELMET":      (0, 0, 255),
    "TRIPLE_RIDING":  (255, 0, 180),
    "SIGNAL_JUMP":    (0, 128, 255),
}


# ─── Drawing helpers ─────────────────────────────────────────────────────────

def draw_bbox(frame, bbox, label, color=None, thickness=2):
    """Draw a single bounding box with label on frame (in-place)."""
    x1, y1, x2, y2 = map(int, bbox)
    color = color or CLASS_COLORS.get(label.lower(), (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_violation_overlay(frame, violation_type, bbox):
    """Draw a flashing red border around a violated vehicle."""
    x1, y1, x2, y2 = map(int, bbox)
    color = VIOLATION_COLORS.get(violation_type, (0, 0, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    label = f"⚠ {violation_type}"
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def annotate_frame(frame, detections, violations):
    """
    Draw all detections and violations on a copy of frame.
    
    Args:
        frame: numpy BGR image
        detections: list of Detection namedtuples
        violations: list of Violation namedtuples
    Returns:
        annotated frame (copy)
    """
    out = frame.copy()
    violated_ids = {v.vehicle_id for v in violations}

    for det in detections:
        label = f"{det.cls} {det.conf:.2f}"
        color = CLASS_COLORS.get(det.cls, (200, 200, 200))
        if det.track_id in violated_ids:
            color = (0, 0, 255)
        draw_bbox(out, det.bbox, label, color)

    for v in violations:
        draw_violation_overlay(out, v.violation_type, v.bbox)

    return out


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def iou(box_a, box_b):
    """Compute Intersection over Union of two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def overlap_ratio(inner_box, outer_box):
    """
    What fraction of inner_box is inside outer_box?
    Used to check if a person/helmet is 'on' a motorcycle.
    """
    xa = max(inner_box[0], outer_box[0])
    ya = max(inner_box[1], outer_box[1])
    xb = min(inner_box[2], outer_box[2])
    yb = min(inner_box[3], outer_box[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_inner = (inner_box[2]-inner_box[0]) * (inner_box[3]-inner_box[1])
    return inter / area_inner if area_inner > 0 else 0.0


def get_overlapping(boxes_a, ref_box, threshold=0.3):
    """Return items from boxes_a whose overlap with ref_box >= threshold."""
    return [b for b in boxes_a if overlap_ratio(b.bbox, ref_box) >= threshold]


# ─── CSV Logging ─────────────────────────────────────────────────────────────

# LOG_DIR is resolved relative to project root at call time, not import time.
# This avoids creating directories in the wrong CWD when imported from tests
# or when Streamlit is launched from a different working directory.
LOG_DIR = Path("outputs/logs")


def _ensure_log_dir() -> Path:
    """Resolve and create log directory on first use, not at import time."""
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_log_path() -> Path:
    today = datetime.now().strftime("%Y%m%d")
    return _ensure_log_dir() / f"violations_{today}.csv"


def save_violation_to_csv(violation, plate_text="UNKNOWN"):
    """Append a single violation record to today's CSV log."""
    log_path = get_log_path()
    record = {
        "violation_id": f"V{datetime.now().strftime('%H%M%S%f')[:10]}",
        "plate_number": plate_text,
        "violation_type": violation.violation_type,
        "confidence": round(violation.confidence, 3),
        "timestamp": violation.timestamp,
        "frame_id": violation.frame_id,
        "image_path": violation.image_path or "",
    }
    df = pd.DataFrame([record])
    write_header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=write_header, index=False)
    return log_path


def load_violations_log():
    """Load today's violation log as a DataFrame."""
    log_path = get_log_path()
    if log_path.exists():
        return pd.read_csv(log_path)
    return pd.DataFrame(columns=[
        "violation_id", "plate_number", "violation_type",
        "confidence", "timestamp", "frame_id", "image_path"
    ])
