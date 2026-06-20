"""
api/main.py — Gridlock FastAPI Backend
  Wraps the existing src/ detection pipeline behind REST + WebSocket endpoints.
  Models are lazy-loaded as singletons (expensive YOLO / PaddleOCR weights
  only load on first request, not at import time).

Usage:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import sys
import os
# Ensure project root is on sys.path so `from src.X import Y` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import json
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import (
    FastAPI, File, HTTPException, Query, UploadFile,
    WebSocket, WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Gridlock — Traffic Violation Detection API",
    version="1.0.0",
    description="YOLOv8 + PaddleOCR violation detection REST and WebSocket API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # CRA / alt dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-loaded singletons ───────────────────────────────────────────────────

_detector = None
_alpr = None


def get_detector(conf: float = 0.45) -> Any:
    """Return the shared detector singleton, loading on first call."""
    global _detector
    if _detector is None:
        from src.detect import GridlockDetector
        _detector = GridlockDetector(conf_threshold=conf)
    return _detector


def get_alpr() -> Any:
    """Return the shared ALPR singleton, loading on first call."""
    global _alpr
    if _alpr is None:
        try:
            from src.alpr import ALPRPipeline
            _alpr = ALPRPipeline()
        except ImportError as e:
            raise HTTPException(
                status_code=500,
                detail=f"ALPR deps missing (pip install paddleocr paddlepaddle): {e}",
            )
    return _alpr


def get_checker(
    overlap: float = 0.30,
    triple: int = 3,
    signal_roi: Optional[List[int]] = None,
) -> Any:
    """ViolationChecker is lightweight — recreate per request with user params."""
    from src.violations import ViolationChecker
    return ViolationChecker(
        overlap_threshold=overlap,
        triple_riding_threshold=triple,
        signal_roi=signal_roi,
    )


# ─── Pydantic response models ─────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"


class DetectionOut(BaseModel):
    cls: str
    conf: float
    bbox: List[float]
    track_id: int


class ViolationOut(BaseModel):
    violation_type: str
    vehicle_id: int
    bbox: List[float]
    confidence: float
    frame_id: int
    timestamp: str
    plate_text: str
    rider_count: int


class DetectImageResponse(BaseModel):
    detections: List[DetectionOut]
    violations: List[ViolationOut]
    annotated_image_b64: str = Field(
        description="JPEG annotated frame as base64 — use as <img src='data:image/jpeg;base64,...'>"
    )
    frame_id: int = 0
    timestamp: str


class ALPRResponse(BaseModel):
    raw_text: str
    plate_number: str
    confidence: float


class ViolationRecord(BaseModel):
    violation_id: str = ""
    plate_number: str = ""
    violation_type: str = ""
    confidence: float = 0.0
    timestamp: str = ""
    frame_id: Any = 0
    image_path: str = ""


class ViolationLogResponse(BaseModel):
    count: int
    records: List[Dict[str, Any]]


class ViolationSummaryResponse(BaseModel):
    total: int
    NO_HELMET: int
    TRIPLE_RIDING: int
    SIGNAL_JUMP: int


class DatasetFetchRoboflowRequest(BaseModel):
    api_key: str
    workspace: str
    project: str
    version: int
    format: str = "yolov8"


class DatasetFetchResponse(BaseModel):
    status: str
    message: str
    path: str


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to BGR numpy image, raising 400 on failure."""
    arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image — ensure it's a valid JPEG/PNG/BMP.")
    return frame


def _frame_to_b64(frame: np.ndarray) -> str:
    """Encode a BGR frame to JPEG base64 string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _detection_to_dict(d) -> dict:
    return {
        "cls": d.cls,
        "conf": round(float(d.conf), 4),
        "bbox": [round(float(x), 2) for x in d.bbox],
        "track_id": int(d.track_id),
    }


def _violation_to_dict(v) -> dict:
    return {
        "violation_type": v.violation_type,
        "vehicle_id": int(v.vehicle_id),
        "bbox": [round(float(x), 2) for x in v.bbox],
        "confidence": round(float(v.confidence), 4),
        "frame_id": int(v.frame_id),
        "timestamp": v.timestamp,
        "plate_text": v.plate_text,
        "rider_count": int(v.rider_count),
    }


def _run_alpr_on_violation(frame: np.ndarray, violation, alpr) -> str:
    """Detect plate within a vehicle bbox and return plate text."""
    try:
        plate_bboxes = alpr.detect_plates_in_vehicle(frame, violation.bbox)
        if plate_bboxes:
            result = alpr.read_plate_from_frame(frame, plate_bboxes[0])
            if result:
                return result.plate_number
    except Exception:
        pass
    return "UNKNOWN"


# ─── Endpoints ────────────────────────────────────────────────────────────────

# GET /health
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check."""
    return {"status": "ok"}


# POST /detect/image
@app.post("/detect/image", response_model=DetectImageResponse, tags=["Detection"])
async def detect_image(
    file: UploadFile = File(..., description="Traffic image (JPEG/PNG/BMP)"),
    conf_threshold: float = Query(0.45, ge=0.1, le=0.9, description="YOLO confidence threshold"),
    overlap_threshold: float = Query(0.30, ge=0.1, le=0.8, description="Person-vehicle overlap threshold"),
    triple_threshold: int = Query(3, ge=2, le=5, description="Riders to trigger triple-riding violation"),
    run_alpr: bool = Query(True, description="Run ALPR plate reading on violated vehicles"),
):
    """
    Detect violations in a single uploaded image.
    Returns detections, violations, and annotated image as base64 JPEG.
    """
    raw = await file.read()
    frame = _decode_image(raw)

    # Apply per-request conf threshold without rebuilding the model
    det = get_detector()
    old_conf = det.conf
    det.conf = conf_threshold

    try:
        detections = det.predict(frame)
    finally:
        det.conf = old_conf

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    checker = get_checker(overlap=overlap_threshold, triple=triple_threshold)
    violations = checker.check(detections, frame_id=0, timestamp=ts)

    # ALPR on each violation
    if run_alpr and violations:
        try:
            alpr = get_alpr()
            for v in violations:
                v.plate_text = _run_alpr_on_violation(frame, v, alpr)
        except HTTPException:
            pass  # ALPR not installed — leave plate_text as UNKNOWN

    # Save to CSV log
    try:
        from src.utils import save_violation_to_csv
        for v in violations:
            save_violation_to_csv(v, v.plate_text)
    except Exception:
        pass

    # Annotate frame
    try:
        from src.utils import annotate_frame
        annotated = annotate_frame(frame, detections, violations)
    except Exception:
        annotated = frame

    return {
        "detections": [_detection_to_dict(d) for d in detections],
        "violations": [_violation_to_dict(v) for v in violations],
        "annotated_image_b64": _frame_to_b64(annotated),
        "frame_id": 0,
        "timestamp": ts,
    }


# POST /alpr/read
@app.post("/alpr/read", response_model=ALPRResponse, tags=["ALPR"])
async def alpr_read(
    file: UploadFile = File(..., description="Plate crop image"),
):
    """Read an Indian license plate from a cropped plate image."""
    raw = await file.read()
    frame = _decode_image(raw)

    try:
        alpr = get_alpr()
    except HTTPException as e:
        raise e

    try:
        result = alpr.read_plate_from_crop(frame)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {e}")

    if result is None:
        raise HTTPException(status_code=400, detail="Could not read plate from image.")

    return {
        "raw_text": result.raw_text,
        "plate_number": result.plate_number,
        "confidence": round(float(result.confidence), 4),
    }


# GET /violations/log
@app.get("/violations/log", response_model=ViolationLogResponse, tags=["Violations"])
async def violations_log(
    limit: int = Query(50, ge=1, le=500, description="Max records to return"),
):
    """Return today's violation log, newest first."""
    try:
        from src.utils import load_violations_log
        df = load_violations_log()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read log: {e}")

    if df.empty:
        return {"count": 0, "records": []}

    # Newest first, limited
    df = df.iloc[::-1].head(limit)
    records = df.fillna("").to_dict(orient="records")
    return {"count": len(records), "records": records}


# GET /violations/summary
@app.get("/violations/summary", response_model=ViolationSummaryResponse, tags=["Violations"])
async def violations_summary():
    """Return violation counts by type for today."""
    try:
        from src.utils import load_violations_log
        df = load_violations_log()
        
        counts = {
            "total": len(df),
            "NO_HELMET": int((df["violation_type"] == "NO_HELMET").sum()),
            "TRIPLE_RIDING": int((df["violation_type"] == "TRIPLE_RIDING").sum()),
            "SIGNAL_JUMP": int((df["violation_type"] == "SIGNAL_JUMP").sum()),
        }
        return counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read log: {e}")

# DELETE /violations/log
@app.delete("/violations/log", tags=["Violations"])
async def clear_violations():
    """Delete today's violation log."""
    try:
        from src.utils import clear_violations_log
        clear_violations_log()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not clear log: {e}")


# WS /ws/detect/video
@app.websocket("/ws/detect/video")
async def ws_detect_video(websocket: WebSocket):
    """
    Stream video frames for violation detection.
    Protocol:
      1. Client sends raw video bytes as first binary message.
      2. Server processes every 3rd frame, sends JSON per frame.
      3. Server sends {"done": true} on completion.
    """
    await websocket.accept()
    tmp_path = None

    try:
        # Receive video bytes
        video_bytes = await websocket.receive_bytes()

        # Write to temp file (cv2.VideoCapture needs a path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        det = get_detector()
        checker = get_checker()

        # Count total frames for progress
        cap = cv2.VideoCapture(tmp_path)
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        cap.release()

        try:
            alpr = get_alpr()
        except Exception:
            alpr = None

        frame_skip = 3
        for fid, frame, detections, ts in det.stream_video(tmp_path):
            if fid % frame_skip != 0:
                continue

            violations = checker.check(detections, frame_id=fid, timestamp=ts)

            if alpr:
                for v in violations:
                    v.plate_text = _run_alpr_on_violation(frame, v, alpr)
                    try:
                        from src.utils import save_violation_to_csv
                        save_violation_to_csv(v, v.plate_text)
                    except Exception:
                        pass

            try:
                from src.utils import annotate_frame
                annotated = annotate_frame(frame, detections, violations)
            except Exception:
                annotated = frame

            progress = round(min(fid / total_frames, 1.0), 4)
            payload = {
                "frame_id": fid,
                "progress": progress,
                "detections": [_detection_to_dict(d) for d in detections],
                "violations": [_violation_to_dict(v) for v in violations],
                "annotated_image_b64": _frame_to_b64(annotated),
            }
            await websocket.send_text(json.dumps(payload))

        await websocket.send_text(json.dumps({"done": True}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink(missing_ok=True)


# POST /dataset/fetch/roboflow
@app.post("/dataset/fetch/roboflow", response_model=DatasetFetchResponse, tags=["Dataset"])
async def dataset_fetch_roboflow(body: DatasetFetchRoboflowRequest):
    """
    Pull a Roboflow dataset version via the roboflow Python package.
    Requires: pip install roboflow
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="roboflow package not installed. Run: pip install roboflow",
        )

    dest = "data/raw/roboflow"
    try:
        rf = Roboflow(api_key=body.api_key)
        project = rf.workspace(body.workspace).project(body.project)
        version = project.version(body.version)
        version.download(body.format, location=dest)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Roboflow API error: {e}",
        )

    return {
        "status": "ok",
        "message": f"Downloaded {body.workspace}/{body.project} v{body.version} ({body.format}) to {dest}",
        "path": str(Path(dest).resolve()),
    }


# POST /dataset/fetch/kaggle
@app.post("/dataset/fetch/kaggle", response_model=DatasetFetchResponse, tags=["Dataset"])
async def dataset_fetch_kaggle(
    dataset_slug: str = Query(..., description="Kaggle dataset slug: owner/name"),
):
    """
    Pull a Kaggle dataset via kagglehub.
    Requires kaggle.json in ~/.kaggle/ and pip install kagglehub.
    """
    try:
        import kagglehub
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="kagglehub not installed. Run: pip install kagglehub",
        )

    try:
        path = kagglehub.dataset_download(dataset_slug)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Kaggle API error: {e}",
        )

    return {
        "status": "ok",
        "message": f"Downloaded {dataset_slug} via kagglehub",
        "path": str(path),
    }
