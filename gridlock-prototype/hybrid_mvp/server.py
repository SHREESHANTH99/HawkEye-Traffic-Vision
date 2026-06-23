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
    Column, Integer, String, DateTime, Boolean, create_engine, func
)
from sqlalchemy.orm import declarative_base, sessionmaker
import os
_db_path = os.getenv("DB_PATH", "/app/data/traffic_analytics.db")
os.makedirs(os.path.dirname(_db_path), exist_ok=True)
DATABASE_URL = f"sqlite:///{_db_path}"
_static_dir = os.getenv("STATIC_DIR", "/app/data/static/violations")
os.makedirs(_static_dir, exist_ok=True)
VIOLATIONS_DIR = Path(_static_dir)
VALID_VIOLATIONS = {
    "HELMET_CHECK", "TRIPLE_RIDING", "SEATBELT_CHECK",
    "WRONG_WAY", "LANE_VIOLATION", "MOBILE_PHONE", "OVERLOADING",
    "NO_HELMET"
}
engine  = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base    = declarative_base()


class ViolationLog(Base):
    __tablename__ = "violation_log"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    violation_type = Column(String(64),  nullable=False, index=True)
    vehicle_id     = Column(String(32),  nullable=False)
    frame_id       = Column(Integer,     nullable=True)
    image_path     = Column(String(256), nullable=True)
    plate_text     = Column(String(64),  nullable=True)
    plate_valid    = Column(Boolean,     default=False)
    timestamp      = Column(DateTime,    default=datetime.utcnow, index=True)


Base.metadata.create_all(bind=engine)
print(f"[DB] SQLite ready: {Path(_db_path).resolve()}")
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
app.mount("/static/violations", StaticFiles(directory=_static_dir), name="static")


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
@app.post("/api/violation")
async def receive_violation(
    violation_type: str = Form(...),
    vehicle_id:     str = Form(...),
    frame_id:       str = Form("0"),
    plate_text:     str = Form("UNKNOWN"),
    plate_valid:    str = Form("false"),
    image:          UploadFile = File(...),
) -> JSONResponse:
    if violation_type not in VALID_VIOLATIONS:
        raise HTTPException(status_code=400,
                            detail=f"Unknown violation_type: {violation_type}")

    ts        = datetime.utcnow()
    filename  = f"{violation_type}_v{vehicle_id}_f{frame_id}_{int(ts.timestamp())}.jpg"
    file_path = VIOLATIONS_DIR / filename

    try:
        contents = await image.read()
        file_path.write_bytes(contents)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Image save failed: {exc}")

    rel_url = f"/static/violations/{filename}"
    db = Session()

    try:
        record = ViolationLog(
            violation_type=violation_type,
            vehicle_id=str(vehicle_id),
            frame_id=int(frame_id) if frame_id.isdigit() else None,
            image_path=rel_url,
            plate_text=plate_text,
            plate_valid=(plate_valid.lower() == "true"),
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
    payload = {
        "event":          "violation",
        "id":             record_id,
        "violation_type": violation_type,
        "vehicle_id":     str(vehicle_id),
        "frame_id":       frame_id,
        "plate_text":     plate_text,
        "plate_valid":    (plate_valid.lower() == "true"),
        "image_url":      rel_url,
        "timestamp":      ts.isoformat(),
    }
    await manager.broadcast(payload)
    return JSONResponse(content={"verdict": "logged", "id": record_id, **payload})


@app.get("/api/stats")

def get_stats() -> JSONResponse:
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
                "plate_text":     r.plate_text,
                "plate_valid":    r.plate_valid,
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
        db = Session()
        rows = (
            db.query(ViolationLog.violation_type, func.count(ViolationLog.id))
            .group_by(ViolationLog.violation_type)
            .all()
        )
        db.close()
        stats = {vt: cnt for vt, cnt in rows}
        await ws.send_json({"event": "init_stats", "stats": stats})

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


@app.delete("/api/violations")
def delete_all_violations() -> JSONResponse:
    db = Session()
    try:
        db.query(ViolationLog).delete()
        db.commit()
        for f in VIOLATIONS_DIR.glob("*.jpg"):
            f.unlink(missing_ok=True)
        return JSONResponse(content={"status": "cleared", "message": "All violations deleted."})
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "HawkEye v2.0"}


if __name__ == "__main__":
    print("="*60)
    print("  HawkEye Traffic Analytics Server v2.0")
    print("  http://0.0.0.0:8001")
    print("  WebSocket: ws://0.0.0.0:8001/ws")
    print("  API docs : http://0.0.0.0:8001/docs")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="warning")
