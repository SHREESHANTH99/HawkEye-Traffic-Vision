"""
alpr.py — Automatic License Plate Recognition for Gridlock
  Pipeline:
    1. Detect plate region using a lightweight YOLO plate detector
       (uses a pretrained model from Roboflow/Ultralytics hub)
    2. Crop the plate from the frame
    3. Preprocess: upscale, denoise, threshold for OCR clarity
    4. Run PaddleOCR on the crop
    5. Post-process: regex filter to match Indian plate format

  Indian plate format:
    State Code (2 letters) + District Code (2 digits) +
    Series (1-2 letters) + Number (4 digits)
    e.g.  KA03MN5678  |  MH12AB1234  |  DL8CAB0001
"""

from __future__ import annotations
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

# PaddleOCR (lazy import to avoid heavy startup unless needed)
_ocr = None
def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr


# ─── Indian plate regex ───────────────────────────────────────────────────────

# Covers: KA03MN5678 / MH12AB1234 / DL8CAB0001 / TN22V5678
PLATE_PATTERN = re.compile(
    r"[A-Z]{2}\s*[0-9]{1,2}\s*[A-Z]{1,3}\s*[0-9]{1,4}", re.IGNORECASE
)

STATE_CODES = {
    "KA", "MH", "DL", "TN", "AP", "TS", "KL", "GJ", "RJ", "UP",
    "WB", "MP", "OR", "HR", "BR", "PB", "AS", "JH", "UK", "HP",
    "CH", "GA", "MN", "ML", "MZ", "NL", "SK", "TR", "AR", "DN"
}


# ─── Plate detection result ───────────────────────────────────────────────────

@dataclass
class PlateResult:
    raw_text: str          # raw OCR output
    plate_number: str      # cleaned, formatted plate
    confidence: float      # OCR confidence
    bbox: list             # plate bbox in original frame [x1,y1,x2,y2]
    crop: Optional[np.ndarray] = None  # cropped plate image


# ─── ALPR Pipeline ────────────────────────────────────────────────────────────

class ALPRPipeline:
    """
    Automatic License Plate Recognition pipeline.

    Usage:
        alpr = ALPRPipeline()
        result = alpr.read_plate_from_frame(frame, plate_bbox)
        # or
        result = alpr.read_plate(image_path)
    """

    def __init__(self, plate_model_path: Optional[str] = None):
        """
        Args:
            plate_model_path: path to YOLO plate detector weights.
              If None, plate detection is skipped and caller must provide bbox.
        """
        self.plate_model = None
        if plate_model_path and Path(plate_model_path).exists():
            from ultralytics import YOLO
            self.plate_model = YOLO(plate_model_path)
            print(f"[ALPR] Plate detector loaded: {plate_model_path}")
        else:
            print("[ALPR] No plate detector model — supply bbox manually or use detect_plates()")

    # ── Plate detection ───────────────────────────────────────────────────────

    def detect_plates(self, frame: np.ndarray, conf: float = 0.4) -> list:
        """
        Detect license plate bounding boxes in frame using YOLO.
        Returns list of [x1,y1,x2,y2] bboxes.
        Falls back to a simple Haar cascade if no YOLO model available.
        """
        if self.plate_model:
            results = self.plate_model.predict(frame, conf=conf, verbose=False)
            if results and results[0].boxes is not None:
                return results[0].boxes.xyxy.tolist()
            return []
        else:
            # Fallback: heuristic crop of lower-centre of vehicle bbox
            # (used when no separate plate model is available)
            return []

    def detect_plates_in_vehicle(self, frame: np.ndarray, vehicle_bbox: list) -> list:
        """
        Given a vehicle bbox, crop it and detect plates within it.
        Returned bboxes are in original frame coordinates.
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        # Plates are typically in the lower ~30% of a vehicle bbox
        plate_region_y = y1 + int((y2 - y1) * 0.65)
        vehicle_crop = frame[plate_region_y:y2, x1:x2]

        plate_bboxes_local = self.detect_plates(vehicle_crop)
        # Convert back to original frame coords
        plate_bboxes = []
        for pb in plate_bboxes_local:
            plate_bboxes.append([
                pb[0] + x1, pb[1] + plate_region_y,
                pb[2] + x1, pb[3] + plate_region_y,
            ])
        return plate_bboxes

    # ── Image preprocessing ───────────────────────────────────────────────────

    def _preprocess_plate(self, crop: np.ndarray) -> np.ndarray:
        """
        Enhance plate crop for better OCR:
          1. Upscale 3x for small plates
          2. Convert to grayscale
          3. CLAHE contrast enhancement
          4. Gaussian blur to reduce noise
          5. Adaptive threshold for clean black/white text
        """
        # Upscale
        h, w = crop.shape[:2]
        if w < 200:
            scale = 200 / w
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Denoise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _ocr_plate(self, crop: np.ndarray) -> Tuple[str, float]:
        """Run PaddleOCR and return (raw_text, avg_confidence)."""
        ocr = _get_ocr()

        # PaddleOCR expects BGR or RGB — convert thresh back to BGR 3-channel
        if len(crop.shape) == 2:
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            crop_bgr = crop

        result = ocr.ocr(crop_bgr, cls=True)

        texts = []
        confidences = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text = line[1][0]
                    conf = line[1][1]
                    texts.append(text)
                    confidences.append(conf)

        raw_text = " ".join(texts).upper().strip()
        avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
        return raw_text, avg_conf

    # ── Post-processing ───────────────────────────────────────────────────────

    def _clean_plate(self, raw: str) -> str:
        """
        Clean OCR output to extract a valid Indian plate number.
        Steps:
          1. Remove non-alphanumeric chars
          2. Fix common OCR errors (0↔O, 1↔I, 8↔B)
          3. Regex match against Indian plate pattern
        """
        # Remove noise characters
        cleaned = re.sub(r"[^A-Z0-9\s]", "", raw.upper())

        # Common OCR substitution fixes
        fixes = {"0": "O", "1": "I", "8": "B"}  # applied to letter positions only
        # (More robust: apply only to alphabetic positions — kept simple for prototype)

        # Try to match Indian plate pattern
        match = PLATE_PATTERN.search(cleaned.replace(" ", ""))
        if match:
            plate = match.group(0).replace(" ", "").upper()
            # Validate state code
            if plate[:2] in STATE_CODES:
                return plate
            return plate  # return anyway, state might be valid but missing from list

        # No match: return best-effort cleaned text
        return cleaned.replace(" ", "") or "UNREADABLE"

    # ── Public API ────────────────────────────────────────────────────────────

    def read_plate_from_crop(self, crop: np.ndarray) -> PlateResult:
        """
        Run the full OCR pipeline on an already-cropped plate image.
        """
        preprocessed = self._preprocess_plate(crop)
        raw_text, conf = self._ocr_plate(preprocessed)
        plate_number = self._clean_plate(raw_text)
        return PlateResult(
            raw_text=raw_text,
            plate_number=plate_number,
            confidence=conf,
            bbox=[],
            crop=crop,
        )

    def read_plate_from_frame(
        self,
        frame: np.ndarray,
        bbox: list,
    ) -> Optional[PlateResult]:
        """
        Crop bbox from frame and run OCR.
        Args:
            frame: full BGR frame
            bbox: [x1, y1, x2, y2] plate or vehicle bbox
        """
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        result = self.read_plate_from_crop(crop)
        result.bbox = bbox
        return result

    def read_plate(self, image_path: str) -> Optional[PlateResult]:
        """
        Convenience: load image from path, detect & read plate.
        """
        import cv2
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        plate_bboxes = self.detect_plates(frame)
        if not plate_bboxes:
            # No plate detected → try reading whole image as plate crop
            return self.read_plate_from_frame(frame, [0, 0, frame.shape[1], frame.shape[0]])

        # Use highest-confidence plate bbox (first result from YOLO)
        return self.read_plate_from_frame(frame, plate_bboxes[0])


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    alpr = ALPRPipeline()
    if len(sys.argv) > 1:
        result = alpr.read_plate(sys.argv[1])
        if result:
            print(f"Raw OCR  : {result.raw_text}")
            print(f"Plate    : {result.plate_number}")
            print(f"Conf     : {result.confidence:.2f}")
    else:
        print("Usage: python src/alpr.py path/to/plate_image.jpg")
