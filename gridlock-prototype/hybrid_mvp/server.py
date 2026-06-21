"""
server.py  —  HawkEye FastAPI Backend
======================================
Endpoints:
  POST /api/violation  — Receive violation from edge_client, save image, log to DB, broadcast via WS
  GET  /api/stats      — Current violation counts
  GET  /api/violations — Paginated recent violations list
  WS   /ws             — Live WebSocket feed for dashboard

Database: SQLite via SQLAlchemy  (traffic_analytics.db)
Static  : /static/violations/    (served crop images)
"""

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import uvicorn
from fastapi import (
    FastAPI, File, Form, UploadFile, WebSocket,
    WebSocketDisconnect, HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import (
    Column, Integer, String, DateTime, create_engine, func
)
from sqlalchemy.orm import declarative_base, sessionmaker

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH        = "traffic_analytics.db"
STATIC_DIR     = Path("static")
VIOLATIONS_DIR = STATIC_DIR / "violations"
VIOLATIONS_DIR.mkdir(parents=True, exist_ok=True)

VALID_VIOLATIONS = {
    "HELMET_CHECK", "TRIPLE_RIDING", "SEATBELT_CHECK",
    "WRONG_WAY", "LANE_VIOLATION", "MOBILE_PHONE", "OVERLOADING",
}

# ── DATABASE ──────────────────────────────────────────────────────────────────
engine  = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base    = declarative_base()


class ViolationLog(Base):
    __tablename__ = "violation_log"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    violation_type = Column(String(64),  nullable=False, index=True)
    vehicle_id     = Column(String(32),  nullable=False)
    frame_id       = Column(Integer,     nullable=True)
    image_path     = Column(String(256), nullable=True)
    timestamp      = Column(DateTime,    default=datetime.utcnow, index=True)


# Create tables explicitly at startup
Base.metadata.create_all(bind=engine)
print(f"[DB] SQLite ready: {Path(DB_PATH).resolve()}")

# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HawkEye Traffic Analytics",
    description="Real-time traffic violation detection backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file directory — serves violation crop images
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── WEBSOCKET MANAGER ──────────────────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        print(f"[WS] Client connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        print(f"[WS] Client disconnected. Total: {len(self.active)}")

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── ENDPOINTS ──────────────────────────────────────────────────────────────────

@app.post("/api/violation")
async def receive_violation(
    violation_type: str = Form(...),
    vehicle_id:     str = Form(...),
    frame_id:       str = Form("0"),
    image:          UploadFile = File(...),
) -> JSONResponse:
    """
    Receive a confirmed violation crop from edge_client.py.
    Saves image, writes DB record, broadcasts to dashboard via WebSocket.
    """
    # Validate violation type
    if violation_type not in VALID_VIOLATIONS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown violation_type: {violation_type}")

    # Save image to disk
    ts        = datetime.utcnow()
    filename  = f"{violation_type}_v{vehicle_id}_f{frame_id}_{int(ts.timestamp())}.jpg"
    file_path = VIOLATIONS_DIR / filename
    try:
        contents = await image.read()
        file_path.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image save failed: {exc}")

    rel_url = f"/static/violations/{filename}"

    # Write to DB
    db = Session()
    try:
        record = ViolationLog(
            violation_type=violation_type,
            vehicle_id=str(vehicle_id),
            frame_id=int(frame_id) if frame_id.isdigit() else None,
            image_path=rel_url,
            timestamp=ts,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        record_id = record.id
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB write failed: {exc}")
    finally:
        db.close()

    print(f"[POST] #{record_id} {violation_type} veh={vehicle_id} frame={frame_id}")

    # Build broadcast payload for dashboard
    payload = {
        "event":          "violation",
        "id":             record_id,
        "violation_type": violation_type,
        "vehicle_id":     str(vehicle_id),
        "frame_id":       frame_id,
        "image_url":      rel_url,
        "timestamp":      ts.isoformat(),
    }

    # Broadcast to all connected WebSocket clients
    await manager.broadcast(payload)

    return JSONResponse(content={"verdict": "logged", "id": record_id, **payload})


@app.get("/api/stats")
def get_stats() -> JSONResponse:
    """Return total counts per violation type."""
    db = Session()
    try:
        rows = (
            db.query(ViolationLog.violation_type, func.count(ViolationLog.id))
            .group_by(ViolationLog.violation_type)
            .all()
        )
        stats = {vt: cnt for vt, cnt in rows}
        stats["total"] = sum(stats.values())
        return JSONResponse(content=stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@app.get("/api/violations")
def get_violations(limit: int = 50, offset: int = 0) -> JSONResponse:
    """Return recent violations (newest first)."""
    db = Session()
    try:
        rows = (
            db.query(ViolationLog)
            .order_by(ViolationLog.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
        data = [
            {
                "id":             r.id,
                "violation_type": r.violation_type,
                "vehicle_id":     r.vehicle_id,
                "frame_id":       r.frame_id,
                "image_url":      r.image_path,
                "timestamp":      r.timestamp.isoformat() if r.timestamp else None,
            }
            for r in rows
        ]
        return JSONResponse(content={"violations": data, "total": len(data)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        # Send current stats immediately on connect
        db = Session()
        rows = (
            db.query(ViolationLog.violation_type, func.count(ViolationLog.id))
            .group_by(ViolationLog.violation_type)
            .all()
        )
        db.close()
        stats = {vt: cnt for vt, cnt in rows}
        await ws.send_json({"event": "init_stats", "stats": stats})

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                if data == "ping":
                    await ws.send_text("pong")
            except asyncio.TimeoutError:
                await ws.send_text("ping")
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception as exc:
        print(f"[WS ERR] {exc}")
        manager.disconnect(ws)


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "HawkEye v2.0"}


# ── STARTUP ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  HawkEye Traffic Analytics Server v2.0")
    print("  http://localhost:8001")
    print("  WebSocket: ws://localhost:8001/ws")
    print("  API docs : http://localhost:8001/docs")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
