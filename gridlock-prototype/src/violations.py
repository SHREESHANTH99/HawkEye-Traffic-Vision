"""
violations.py — Rule-based traffic violation checker for Gridlock
  Analyses detections from YOLOv8 and applies violation rules:
    1. NO_HELMET     — motorcycle rider with no helmet detected
    2. TRIPLE_RIDING — motorcycle with 3+ person bboxes overlapping
    3. SIGNAL_JUMP   — vehicle in red-light ROI (optional, requires ROI config)

  Each rule returns a list of Violation objects ready for logging.
"""

from __future__ import annotations
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.detect import Detection
from src.utils import get_overlapping, save_violation_to_csv


# ─── Violation dataclass ──────────────────────────────────────────────────────

@dataclass
class Violation:
    violation_type: str          # "NO_HELMET" | "TRIPLE_RIDING" | "SIGNAL_JUMP"
    vehicle_id: int              # track_id of the offending vehicle
    bbox: List[float]            # bbox of the vehicle
    confidence: float            # avg detection confidence
    frame_id: int
    timestamp: str
    image_path: Optional[str] = None
    plate_text: str = "UNKNOWN"
    rider_count: int = 1         # relevant for TRIPLE_RIDING


# ─── Violation Checker ────────────────────────────────────────────────────────

class ViolationChecker:
    """
    Apply violation rules to a list of Detection objects.

    Usage:
        checker = ViolationChecker()
        violations = checker.check(detections, frame_id=42, timestamp="...")
    """

    def __init__(
        self,
        overlap_threshold: float = 0.30,
        triple_riding_threshold: int = 3,
        signal_roi: Optional[List[int]] = None,
    ):
        """
        Args:
            overlap_threshold: min overlap fraction to associate person/helmet with vehicle
            triple_riding_threshold: number of riders that triggers triple-riding
            signal_roi: [x1,y1,x2,y2] pixel region of the red-light zone in the frame
        """
        self.overlap_threshold = overlap_threshold
        self.triple_threshold = triple_riding_threshold
        self.signal_roi = signal_roi  # set via app config

    # ── Main entry point ──────────────────────────────────────────────────────

    def check(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: str,
    ) -> List[Violation]:
        """Run all violation rules and return combined list."""
        violations = []
        violations.extend(self._check_helmet(detections, frame_id, timestamp))
        violations.extend(self._check_triple_riding(detections, frame_id, timestamp))
        if self.signal_roi:
            violations.extend(self._check_signal_jump(detections, frame_id, timestamp))
        return violations

    # ── Rule 1: No Helmet ─────────────────────────────────────────────────────

    def _check_helmet(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: str,
    ) -> List[Violation]:
        """
        Flag any motorcycle whose overlapping person has no helmet.
        Strategy:
          - For each motorcycle, find persons overlapping it (lower half).
          - For each such person, check if a helmet overlaps their upper body.
          - If any rider lacks a helmet → NO_HELMET violation.
        """
        violations = []
        motorcycles = [d for d in detections if d.cls == "motorcycle"]
        persons     = [d for d in detections if d.cls == "person"]
        helmets     = [d for d in detections if d.cls == "helmet"]
        no_helmets  = [d for d in detections if d.cls == "no_helmet"]

        for moto in motorcycles:
            riders = get_overlapping(persons, moto.bbox, self.overlap_threshold)

            for rider in riders:
                # Check if a helmet overlaps this rider's upper-body region
                rx1, ry1, rx2, ry2 = rider.bbox
                upper_body = [rx1, ry1, rx2, ry1 + (ry2 - ry1) * 0.5]
                rider_helmets = get_overlapping(helmets, upper_body, threshold=0.2)

                # Also check if model explicitly flagged no_helmet on this person
                rider_no_helmet = get_overlapping(no_helmets, rider.bbox, threshold=0.3)

                if not rider_helmets or rider_no_helmet:
                    violations.append(Violation(
                        violation_type="NO_HELMET",
                        vehicle_id=moto.track_id,
                        bbox=moto.bbox,
                        confidence=round((moto.conf + rider.conf) / 2, 3),
                        frame_id=frame_id,
                        timestamp=timestamp,
                    ))
                    break  # one violation per motorcycle

        return violations

    # ── Rule 2: Triple Riding ─────────────────────────────────────────────────

    def _check_triple_riding(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: str,
    ) -> List[Violation]:
        """
        Flag motorcycles with 3 or more overlapping person detections.
        """
        violations = []
        motorcycles = [d for d in detections if d.cls == "motorcycle"]
        persons     = [d for d in detections if d.cls == "person"]

        for moto in motorcycles:
            riders = get_overlapping(persons, moto.bbox, self.overlap_threshold)
            if len(riders) >= self.triple_threshold:
                avg_conf = round(
                    sum(r.conf for r in riders) / len(riders), 3
                )
                violations.append(Violation(
                    violation_type="TRIPLE_RIDING",
                    vehicle_id=moto.track_id,
                    bbox=moto.bbox,
                    confidence=avg_conf,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    rider_count=len(riders),
                ))

        return violations

    # ── Rule 3: Signal Jump (optional) ───────────────────────────────────────

    def _check_signal_jump(
        self,
        detections: List[Detection],
        frame_id: int,
        timestamp: str,
    ) -> List[Violation]:
        """
        Flag any vehicle whose bbox intersects the configured red-light ROI.
        Requires self.signal_roi = [x1, y1, x2, y2] to be set.
        """
        if not self.signal_roi:
            return []

        violations = []
        vehicles = [d for d in detections if d.cls in ("motorcycle", "car", "bus", "truck")]
        rx1, ry1, rx2, ry2 = self.signal_roi

        for v in vehicles:
            vx1, vy1, vx2, vy2 = v.bbox
            # Check bbox overlap with ROI
            if vx1 < rx2 and vx2 > rx1 and vy1 < ry2 and vy2 > ry1:
                violations.append(Violation(
                    violation_type="SIGNAL_JUMP",
                    vehicle_id=v.track_id,
                    bbox=v.bbox,
                    confidence=round(v.conf, 3),
                    frame_id=frame_id,
                    timestamp=timestamp,
                ))

        return violations


# ─── Violation summary helper ─────────────────────────────────────────────────

def summarize_violations(violations: List[Violation]) -> dict:
    """Return counts by type."""
    summary = {"NO_HELMET": 0, "TRIPLE_RIDING": 0, "SIGNAL_JUMP": 0}
    for v in violations:
        summary[v.violation_type] = summary.get(v.violation_type, 0) + 1
    return summary


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.detect import Detection
    # Synthetic test: one motorcycle, one person without helmet
    dets = [
        Detection("motorcycle", 0.91, [100, 200, 300, 400], track_id=1),
        Detection("person",     0.88, [120, 180, 280, 390], track_id=2),
        # No helmet detection → should trigger NO_HELMET
    ]
    checker = ViolationChecker()
    violations = checker.check(dets, frame_id=0, timestamp="2026-06-20 12:00:00")
    for v in violations:
        print(f"[VIOLATION] {v.violation_type}  vehicle_id={v.vehicle_id}  conf={v.confidence}")
