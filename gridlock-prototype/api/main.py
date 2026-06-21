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

# ─── GPU auto-detection ───────────────────────────────────────────────────────

try:
    import torch
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _DEVICE = "cpu"

print(f"[api/main.py] Compute device: {_DEVICE.upper()}")

# ─── Two-stage helmet pipeline constants ──────────────────────────────────────

_HELMET_MODEL_URL   = "https://raw.githubusercontent.com/Viddesh1/Bike-Helmet-Detectionv2/main/weights/best.pt"
_HELMET_MODEL_LOCAL = Path(__file__).parent.parent / "gridlock-prototype" / "models" / "bike_helmet_yolov8.pt"
# Normalise path so it works whether called from repo root or api/ dir
if not _HELMET_MODEL_LOCAL.parent.exists():
    _HELMET_MODEL_LOCAL = Path(__file__).parent.parent / "models" / "bike_helmet_yolov8.pt"
_HELMET_MODEL_LOCAL.parent.mkdir(parents=True, exist_ok=True)

_HELMET_CONF = 0.40
_COCO_CONF   = 0.30
_MIN_VEH_PX  = 40
_HEAD_PAD    = 0.35   # 35% extra upward padding on crop

# COCO class IDs
_CLS_PERSON = 0; _CLS_BICYCLE = 1; _CLS_CAR = 2
_CLS_MOTO   = 3; _CLS_BUS     = 5; _CLS_TRUCK = 7
_BIKE_CLASSES  = {_CLS_MOTO, _CLS_BICYCLE}
_MOTOR_CLASSES = {_CLS_CAR, _CLS_BUS, _CLS_TRUCK}
_ALL_CLASSES   = [_CLS_PERSON, _CLS_BICYCLE, _CLS_CAR, _CLS_MOTO, _CLS_BUS, _CLS_TRUCK]
_CLS_LABEL     = {0:"PERSON",1:"BICYCLE",2:"CAR",3:"MOTO",5:"BUS",7:"TRUCK"}

_COLORS = {
    "NO_HELMET":     (0,  80,  255),
    "TRIPLE_RIDING": (0,  0,   255),
    "OK":            (0,  200, 0),
    "SCAN":          (160,160, 0),
    "SMALL":         (60, 60,  60),
}

JUDGE_SERVER = "http://localhost:8001/api/violation"

# ─── Lazy-loaded singletons ───────────────────────────────────────────────────

_detector      = None
_alpr          = None
_coco_model    = None   # YOLO COCO model for two-stage pipeline
_helmet_model  = None   # helmet YOLO model
_no_helmet_ids : set = set()
_helmet_ids    : set = set()


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


# ─── Two-stage pipeline helpers ───────────────────────────────────────────────

def _ensure_helmet_weights() -> Optional[Path]:
    """Download helmet weights once and cache; reuse on later runs."""
    if _HELMET_MODEL_LOCAL.exists() and _HELMET_MODEL_LOCAL.stat().st_size > 1_000_000:
        return _HELMET_MODEL_LOCAL
    print(f"[api] Downloading helmet weights (~88MB) from:\n      {_HELMET_MODEL_URL}")
    try:
        import requests as req
        with req.get(_HELMET_MODEL_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = _HELMET_MODEL_LOCAL.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(_HELMET_MODEL_LOCAL)
        print(f"[api] Helmet weights saved → {_HELMET_MODEL_LOCAL}")
        return _HELMET_MODEL_LOCAL
    except Exception as e:
        print(f"[api][WARN] Could not download helmet weights: {e}")
        return None


def _get_coco_model():
    """Singleton YOLO COCO model for the two-stage pipeline."""
    global _coco_model
    if _coco_model is None:
        from ultralytics import YOLO
        print("[api] Loading COCO tracking model (yolov8s.pt)…")
        _coco_model = YOLO("yolov8s.pt")
        _coco_model.to(_DEVICE)
        print(f"[api] COCO model ready on {_DEVICE.upper()}")
    return _coco_model


def _get_helmet_model():
    """Singleton helmet YOLO model."""
    global _helmet_model, _no_helmet_ids, _helmet_ids
    if _helmet_model is None:
        weights = _ensure_helmet_weights()
        if weights is None:
            print("[api][WARN] No helmet weights — helmet checks will be skipped.")
            return None
        try:
            from ultralytics import YOLO
            print(f"[api] Loading helmet model: {weights}")
            _helmet_model = YOLO(str(weights))
            _helmet_model.to(_DEVICE)
            _no_helmet_ids = {
                cid for cid, name in _helmet_model.names.items()
                if any(kw in name.lower() for kw in ["no_helmet","no-helmet","without","bare"])
            }
            _helmet_ids = {
                cid for cid, name in _helmet_model.names.items()
                if "helmet" in name.lower() and cid not in _no_helmet_ids
            }
            print(f"[api] Helmet IDs={_helmet_ids}  No-helmet IDs={_no_helmet_ids}")
        except Exception as e:
            print(f"[api][WARN] Could not load helmet model: {e}")
            _helmet_model = None
    return _helmet_model


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _crop_head(frame: np.ndarray, box: list) -> np.ndarray:
    h, w = frame.shape[:2]
    bh = box[3] - box[1]
    top_ex = int(bh * _HEAD_PAD)
    x1 = max(0, int(box[0]) - 20)
    y1 = max(0, int(box[1]) - 20 - top_ex)
    x2 = min(w, int(box[2]) + 20)
    y2 = min(h, int(box[3]) + 20)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame[:10, :10].copy()
    # Upscale small crops
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 224:
        scale = 224 / min(ch, cw)
        crop = cv2.resize(crop, (max(int(cw*scale), 224), max(int(ch*scale), 224)),
                          interpolation=cv2.INTER_LANCZOS4)
    return crop.copy()


def _check_helmet(crop_img: np.ndarray) -> str:
    """Returns 'NO_HELMET', 'HELMET', or 'SKIP'."""
    hmodel = _get_helmet_model()
    if hmodel is None:
        return "SKIP"
    try:
        results = hmodel.predict(crop_img, conf=_HELMET_CONF, verbose=False, imgsz=320)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return "NO_HELMET"
        clses = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        relevant = [(c, cf) for c, cf in zip(clses, confs)
                    if c in _helmet_ids or c in _no_helmet_ids]
        if not relevant:
            return "NO_HELMET"
        best = max(relevant, key=lambda x: x[1])[0]
        return "NO_HELMET" if best in _no_helmet_ids else "HELMET"
    except Exception as e:
        print(f"  [HELMET ERR] {e}")
        return "SKIP"


def _encode_jpeg(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def _annotate(frame: np.ndarray, bikes: list, persons: list,
              motor_vehs: list, violations: list) -> np.ndarray:
    """Draw bounding boxes + labels on frame."""
    disp = frame.copy()
    viol_types = {v["vehicle_id"]: v["violation_type"] for v in violations}

    for p in persons:
        x1,y1,x2,y2 = [int(v) for v in p["box"]]
        cv2.rectangle(disp, (x1,y1), (x2,y2), (190,190,0), 1)

    for veh in motor_vehs:
        x1,y1,x2,y2 = [int(v) for v in veh["box"]]
        lbl = _CLS_LABEL.get(veh["cls"], "VEH")
        color = _COLORS["SCAN"]
        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 1)
        cv2.putText(disp, f"{lbl}#{veh['id']}", (x1, max(14, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    for veh in bikes:
        x1,y1,x2,y2 = [int(v) for v in veh["box"]]
        vid = veh["id"]
        lbl = _CLS_LABEL.get(veh["cls"], "MOTO")
        vtype = viol_types.get(vid, "")
        if vtype == "NO_HELMET":
            color = _COLORS["NO_HELMET"]
            tag   = f"{lbl}#{vid} NO HELMET"
        elif vtype == "TRIPLE_RIDING":
            color = _COLORS["TRIPLE_RIDING"]
            tag   = f"{lbl}#{vid} TRIPLE!"
        else:
            color = _COLORS["OK"]
            tag   = f"{lbl}#{vid}"
        cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
        cv2.putText(disp, tag, (x1+1, max(15, y1-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3)
        cv2.putText(disp, tag, (x1, max(14, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)
    return disp


async def _post_to_judge(vehicle_id: int, violation_type: str,
                          crop_bytes: bytes, frame_id: int) -> None:
    """Fire-and-forget POST to the Judge Feed server on port 8001."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                JUDGE_SERVER,
                data={"vehicle_id": str(vehicle_id),
                      "violation_type": violation_type,
                      "frame_id": str(frame_id)},
                files={"image": ("crop.jpg", crop_bytes, "image/jpeg")},
            )
    except Exception:
        pass   # judge server offline — ignore silently


def _two_stage_process_frame(
    frame: np.ndarray,
    frame_id: int,
    model,
    confirmed_ids: set,
) -> tuple:
    """
    Run the two-stage detection on a single frame.
    Returns (annotated_frame, violations_this_frame).
    confirmed_ids is a shared set to avoid re-flagging the same vehicle.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── COCO tracking ────────────────────────────────────────────────────
    try:
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=_COCO_CONF, classes=_ALL_CLASSES,
            device=_DEVICE, verbose=False, imgsz=640, iou=0.45,
        )
    except Exception as e:
        print(f"[YOLO ERR] {e}")
        return frame.copy(), []

    persons   : list = []
    bikes     : list = []
    motor_vehs: list = []
    violations: list = []

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        res    = results[0]
        bboxes = res.boxes.xyxy.cpu().numpy()
        clses  = res.boxes.cls.cpu().numpy().astype(int)
        confs  = res.boxes.conf.cpu().numpy()
        ids    = (res.boxes.id.cpu().numpy().astype(int)
                  if res.boxes.id is not None
                  else np.full(len(clses), -1, dtype=int))

        for i in range(len(clses)):
            box = bboxes[i].tolist()
            det = {"box": box, "conf": float(confs[i]),
                   "id": int(ids[i]), "cls": int(clses[i]),
                   "bw": box[2]-box[0], "bh": box[3]-box[1]}
            c = det["cls"]
            if   c == _CLS_PERSON:    persons.append(det)
            elif c in _BIKE_CLASSES:  bikes.append(det)
            elif c in _MOTOR_CLASSES: motor_vehs.append(det)

    # ── Bike violation checks ─────────────────────────────────────────────
    for veh in bikes:
        vid, box = veh["id"], veh["box"]
        bw, bh   = veh["bw"], veh["bh"]

        if bw < _MIN_VEH_PX or bh < _MIN_VEH_PX or vid < 0:
            continue

        riders = [p for p in persons if _boxes_overlap(p["box"], box)]
        count  = len(riders)

        # Triple riding — geometry only
        if count >= 3:
            key = (vid, "TRIPLE_RIDING")
            if key not in confirmed_ids:
                confirmed_ids.add(key)
                try:
                    cb = _encode_jpeg(_crop_head(frame, box))
                except Exception:
                    cb = b""
                violations.append({
                    "violation_type": "TRIPLE_RIDING",
                    "vehicle_id": vid,
                    "bbox": box,
                    "confidence": round(sum(r["conf"] for r in riders)/len(riders), 3),
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "plate_text": "UNKNOWN",
                    "rider_count": count,
                    "_crop_bytes": cb,
                })
            continue

        # Helmet check — YOLO second stage (once per vehicle ID)
        key = (vid, "HELMET_CHECK")
        if key not in confirmed_ids:
            confirmed_ids.add(key)
            crop_img = _crop_head(frame, box)
            verdict  = _check_helmet(crop_img)
            if verdict == "NO_HELMET":
                vkey = (vid, "NO_HELMET")
                if vkey not in confirmed_ids:
                    confirmed_ids.add(vkey)
                    try:
                        cb = _encode_jpeg(crop_img)
                    except Exception:
                        cb = b""
                    violations.append({
                        "violation_type": "NO_HELMET",
                        "vehicle_id": vid,
                        "bbox": box,
                        "confidence": round(veh["conf"], 3),
                        "frame_id": frame_id,
                        "timestamp": ts,
                        "plate_text": "UNKNOWN",
                        "rider_count": count,
                        "_crop_bytes": cb,
                    })

    annotated = _annotate(frame, bikes, persons, motor_vehs, violations)
    return annotated, violations


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


# WS /ws/detect/video  — Two-stage helmet pipeline (mirrors edge_client.py)
@app.websocket("/ws/detect/video")
async def ws_detect_video(websocket: WebSocket):
    """
    Stream video frames for violation detection using the two-stage pipeline:
      1. COCO YOLOv8s tracks vehicles + persons per frame.
      2. For each motorcycle/bicycle, a specialized helmet YOLO model
         checks cropped head regions for NO_HELMET violations.
      3. Confirmed violations are also POSTed to the Judge Feed (port 8001).

    Protocol:
      1. Client sends raw video bytes as first binary message.
      2. Server processes every 3rd frame, sends JSON per frame:
           {frame_id, progress, violations, annotated_image_b64}
      3. Server sends {"done": true} on completion.
    """
    await websocket.accept()
    tmp_path = None

    try:
        params     = websocket.query_params
        frame_skip = int(params.get("frame_skip", 3))

        # Receive video bytes from browser
        video_bytes = await websocket.receive_bytes()

        # Write to temp file — cv2.VideoCapture needs a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        # Load models (singletons — cached after first call)
        model = _get_coco_model()
        _get_helmet_model()   # pre-warm so first frame isn't slow

        cap          = cv2.VideoCapture(tmp_path)
        total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        cap.release()

        confirmed_ids: set = set()   # track which vehicles already flagged
        frame_id = 0

        cap = cv2.VideoCapture(tmp_path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1

                # Resize wide frames
                h0, w0 = frame.shape[:2]
                if w0 > 1280:
                    scale = 1280 / w0
                    frame = cv2.resize(frame, (1280, int(h0 * scale)))

                # Skip frames for speed — still send every frame to browser
                if frame_id % frame_skip != 0:
                    progress = round(min(frame_id / total_frames, 1.0), 4)
                    payload  = {
                        "frame_id": frame_id,
                        "progress": progress,
                        "violations": [],
                        "annotated_image_b64": _frame_to_b64(frame),
                    }
                    await websocket.send_text(json.dumps(payload))
                    continue

                annotated, violations = _two_stage_process_frame(
                    frame, frame_id, model, confirmed_ids
                )

                # Bridge confirmed violations to Judge Feed
                for v in violations:
                    crop_bytes = v.pop("_crop_bytes", b"")
                    if crop_bytes:
                        await _post_to_judge(
                            v["vehicle_id"], v["violation_type"],
                            crop_bytes, frame_id
                        )

                progress = round(min(frame_id / total_frames, 1.0), 4)
                payload  = {
                    "frame_id":           frame_id,
                    "progress":           progress,
                    "violations":         violations,
                    "annotated_image_b64": _frame_to_b64(annotated),
                }
                await websocket.send_text(json.dumps(payload))

        finally:
            cap.release()

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
