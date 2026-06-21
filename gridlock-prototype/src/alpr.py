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
import sys
import os
# Allow running as a standalone script: python src/alpr.py image.jpg
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        import easyocr
        _ocr = easyocr.Reader(['en'], gpu=False)
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
    detection_tier: str = ""
    ocr_confidence_raw: float = 0.0


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

    def detect_plates(self, frame: np.ndarray, conf: float = 0.4) -> Tuple[list, str]:
        """
        Detect license plate bounding boxes in frame using YOLO.
        Returns Tuple[list of [x1,y1,x2,y2] bboxes, detection_tier].
        Falls back to a simple contour heuristic if no YOLO model available.
        """
        if self.plate_model:
            results = self.plate_model.predict(frame, conf=conf, verbose=False)
            if results and results[0].boxes is not None:
                return results[0].boxes.xyxy.tolist(), "trained_model"
            return [], "trained_model"
        else:
            # Fallback: heuristic crop of lower-centre of vehicle bbox
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(filtered, 30, 200)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            candidates = []
            frame_area = frame.shape[0] * frame.shape[1]
            if frame_area == 0:
                return [], "heuristic"
                
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if h < 15: continue
                area = w * h
                if area > 0.8 * frame_area: continue
                ar = w / float(h)
                
                # Accept single row (2.5-6.0) or double row (1.5-3.0)
                if 1.5 <= ar <= 6.0:
                    target_ar = 4.5 if ar >= 3.0 else 2.0
                    score = area / (abs(ar - target_ar) + 0.5)
                    candidates.append((score, [x, y, x + w, y + h]))
                    
            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                return [c[1] for c in candidates], "heuristic"
            else:
                return [[0, 0, frame.shape[1], frame.shape[0]]], "heuristic"

    def detect_plates_in_vehicle(self, frame: np.ndarray, vehicle_bbox: list) -> Tuple[list, str]:
        """
        Given a vehicle bbox, crop it and detect plates within it.
        Returned bboxes are in original frame coordinates.
        """
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        # Plates are typically in the lower ~30% of a vehicle bbox
        plate_region_y = y1 + int((y2 - y1) * 0.65)
        vehicle_crop = frame[plate_region_y:y2, x1:x2]

        plate_bboxes_local, tier = self.detect_plates(vehicle_crop)
        
        # Tier 3: Robustness fallback
        if not plate_bboxes_local and tier == "trained_model":
            old_model = self.plate_model
            self.plate_model = None
            plate_bboxes_local, tier = self.detect_plates(vehicle_crop)
            self.plate_model = old_model

        # Convert back to original frame coords
        plate_bboxes = []
        for pb in plate_bboxes_local:
            plate_bboxes.append([
                pb[0] + x1, pb[1] + plate_region_y,
                pb[2] + x1, pb[3] + plate_region_y,
            ])
        return plate_bboxes, tier

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
        """Run EasyOCR and return (raw_text, avg_confidence)."""
        ocr = _get_ocr()

        # EasyOCR accepts numpy arrays directly. Grayscale or BGR is fine.
        # We can pass the crop directly without conversion.
        try:
            result = ocr.readtext(crop)
        except Exception:
            result = []

        texts = []
        confidences = []
        if result:
            for line in result:
                # line is a tuple: (bbox, text, prob)
                if len(line) >= 3:
                    text = line[1]
                    conf = line[2]
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
          2. Apply position-aware OCR error correction:
               - Letter positions (0-1, 4-5): digits corrected to look-alike letters
               - Digit positions (2-3, 6-9): letters corrected to look-alike digits
          3. Regex match against Indian plate pattern
        """
        # Step 1: Remove noise characters, uppercase
        cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())

        # Step 2: Position-aware OCR correction
        # Indian plate structure: LL DD L{1,2} D{4}
        # Positions: 0,1 = letters | 2,3 = digits | 4,5(,6) = letters | rest = digits
        DIGIT_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G"}
        LETTER_TO_DIGIT = {"O": "0", "I": "1", "B": "8", "S": "5", "G": "6", "Z": "2"}

        corrected = list(cleaned)
        for i, ch in enumerate(corrected):
            if i < 2:
                # Must be letters (state code) — fix digits to letters
                if ch in DIGIT_TO_LETTER:
                    corrected[i] = DIGIT_TO_LETTER[ch]
            elif i < 4:
                # Must be digits (district code) — fix letters to digits
                if ch in LETTER_TO_DIGIT:
                    corrected[i] = LETTER_TO_DIGIT[ch]
            elif i >= len(cleaned) - 4:
                # Last 4 must be digits (serial number) — fix letters to digits
                if ch in LETTER_TO_DIGIT:
                    corrected[i] = LETTER_TO_DIGIT[ch]
            # Middle positions (series letters) — leave as-is

        cleaned = "".join(corrected)

        # Step 3: Regex match
        match = PLATE_PATTERN.search(cleaned)
        if match:
            plate = match.group(0).upper()
            if plate[:2] in STATE_CODES:
                return plate
            return plate  # return anyway; state may be valid but not in our list

        # No match: return best-effort corrected text
        return cleaned or "UNREADABLE"

    # ── Public API ────────────────────────────────────────────────────────────

    def read_plate_from_crop(self, crop: np.ndarray) -> PlateResult:
        """
        Run the full OCR pipeline on an already-cropped plate image.
        """
        h, w = crop.shape[:2]
        ar = w / float(h) if h > 0 else 0
        
        preprocessed = self._preprocess_plate(crop)
        
        # Double-row detection (Tier 3)
        if ar < 3.0 and h >= 30:
            half = preprocessed.shape[0] // 2
            top = preprocessed[:half, :]
            bottom = preprocessed[half:, :]
            raw_top, conf_top = self._ocr_plate(top)
            raw_bottom, conf_bottom = self._ocr_plate(bottom)
            raw_text = raw_top + " " + raw_bottom
            conf = (conf_top + conf_bottom) / 2.0
            raw_conf = conf
            
            # Dynamic Retry (Tier 3)
            if conf < 0.5:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                rt, ct = self._ocr_plate(gray[:h//2, :])
                rb, cb = self._ocr_plate(gray[h//2:, :])
                conf2 = (ct + cb) / 2.0
                if conf2 > conf:
                    raw_text = rt + " " + rb
                    conf = conf2
        else:
            raw_text, conf = self._ocr_plate(preprocessed)
            raw_conf = conf
            # Dynamic Retry (Tier 3)
            if conf < 0.5:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                rt, ct = self._ocr_plate(gray)
                if ct > conf:
                    raw_text = rt
                    conf = ct

        plate_number = self._clean_plate(raw_text)
        return PlateResult(
            raw_text=raw_text,
            plate_number=plate_number,
            confidence=conf,
            bbox=[],
            crop=crop,
            detection_tier="",
            ocr_confidence_raw=raw_conf
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

        plate_bboxes, tier = self.detect_plates(frame)
        if not plate_bboxes:
            # No plate detected → try reading whole image as plate crop
            result = self.read_plate_from_frame(frame, [0, 0, frame.shape[1], frame.shape[0]])
            if result:
                result.detection_tier = tier
            return result

        # Use highest-confidence plate bbox (first result from YOLO)
        result = self.read_plate_from_frame(frame, plate_bboxes[0])
        if result:
            result.detection_tier = tier
        return result


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
