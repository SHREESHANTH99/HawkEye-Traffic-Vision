# HawkEye Hybrid MVP — Filter-and-Judge Traffic AI

> **Fast Edge Detector → Vision LLM Reasoner → Live Dashboard**

---

## Architecture

```
test_traffic.mp4
       │
       ▼
┌─────────────────────────────┐
│  edge_client.py             │  ← Fast Filter (YOLOv8s + ByteTrack)
│  • Detects motorcycles      │    Runs at full video FPS
│  • Counts overlapping riders│    Non-blocking thread dispatch
│  • Crops violation frames   │
└────────────┬────────────────┘
             │  HTTP POST /api/violation
             │  (vehicle_id, type, JPEG crop)
             ▼
┌─────────────────────────────┐
│  server.py  (FastAPI)       │  ← The Judge (async, never blocks)
│  • Receives crop            │
│  • Sends to Ollama (base64) │
│  • Parses YES / NO          │
│  • Saves image + SQLite row │
│  • WebSocket broadcast      │
└────────────┬────────────────┘
             │  ws://localhost:8000/ws/dashboard
             ▼
┌─────────────────────────────┐
│  dashboard.html             │  ← Live Web UI
│  • Violation cards + images │
│  • KPI counters             │
│  • Event log terminal       │
└─────────────────────────────┘
```

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | Already set up (`.venv`) |
| Ollama | [ollama.com/download](https://ollama.com/download) |
| Vision model | `ollama pull moondream` ← fast, 1.7 GB |
| Test video | Place as `hybrid_mvp/test_traffic.mp4` |

> **moondream** is recommended for speed. For better accuracy use `llava`:
> `ollama pull llava` (4 GB, slower).

---

## Quick Start — 3 Terminals

### Terminal 1 — Start Ollama (if not running as service)
```powershell
ollama serve
```

### Terminal 2 — Start FastAPI Server
```powershell
# From: gridlock-prototype/
.venv\Scripts\activate
cd hybrid_mvp
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 3 — Run Edge Client
```powershell
# From: gridlock-prototype/hybrid_mvp/
# Make sure test_traffic.mp4 is in this folder
..\..\.venv\Scripts\python.exe edge_client.py
```

Then open **[http://localhost:8000/static/dashboard.html](http://localhost:8000/static/dashboard.html)** in your browser.

---

## File Structure

```
hybrid_mvp/
├── edge_client.py       ← YOLO + ByteTrack fast filter
├── server.py            ← FastAPI + Ollama judge + SQLite + WebSocket
├── dashboard.html       ← Dev copy (open directly in browser)
├── requirements.txt     ← Python deps
├── test_traffic.mp4     ← YOUR video goes here
├── violations.db        ← SQLite (auto-created on first run)
└── static/
    ├── dashboard.html   ← Served by FastAPI at /static/dashboard.html
    └── violations/      ← Saved confirmed violation JPEG crops
```

---

## Violation Rules

| Type | Trigger | LLM Prompt |
|------|---------|------------|
| `TRIPLE_RIDING` | ≥3 person bboxes overlap one motorcycle | "Does this show 3+ people on ONE motorcycle? YES/NO" |
| `HELMET_CHECK`  | ≥1 person overlapping motorcycle | "Is a rider missing a helmet? YES/NO" |

Each unique `(motorcycle_track_id, violation_type)` pair is dispatched **once** — no duplicate events.

---

## REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/violation` | Edge client dispatch (image + form data) |
| `GET`  | `/api/violations` | List all confirmed violations |
| `GET`  | `/api/stats` | KPI counts by type |
| `WS`   | `/ws/dashboard` | Real-time push to dashboard |
| `GET`  | `/docs` | Auto-generated Swagger UI |

---

## Tuning

Edit the CONFIG block at the top of each file:

**`edge_client.py`**
```python
TRIPLE_RIDING_THRESH = 3     # change to 2 for earlier triggers
CONF_THRESHOLD       = 0.40  # lower = more detections
POST_TIMEOUT         = 4.0   # raise if Ollama is on a slow machine
```

**`server.py`**
```python
OLLAMA_MODEL   = "moondream"  # or "llava"
OLLAMA_TIMEOUT = 60.0         # seconds
```

---

## Switching the LLM Model

```powershell
# Lightweight & fast (default)
ollama pull moondream
set OLLAMA_MODEL=moondream

# More accurate
ollama pull llava
set OLLAMA_MODEL=llava

# Then restart server.py
```

---

## Database Schema

```sql
CREATE TABLE violation_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    violation_id TEXT UNIQUE,         -- UUID
    type         TEXT,                -- TRIPLE_RIDING | HELMET_CHECK
    vehicle_id   INTEGER,             -- ByteTrack motorcycle ID
    frame_id     INTEGER,
    image_url    TEXT,                -- /static/violations/<file>.jpg
    llm_response TEXT,               -- raw Ollama output ("YES" etc.)
    timestamp    DATETIME
);
```

---

*Built for the HawkEye Traffic Vision research project.*
