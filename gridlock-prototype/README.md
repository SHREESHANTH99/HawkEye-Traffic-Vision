# Gridlock — Traffic Violation Detection Prototype

> AI-powered detection of traffic violations in Indian road conditions using YOLOv8 + PaddleOCR + Ollama LLM Judge.

---

## 🚀 Quick Start

The full system uses up to **four processes** running side by side. Processes 1, 2, and 4 are required; process 3 is optional and only needed to populate the Judge Feed page.

### 1. Main Backend (FastAPI + YOLO) — port 8080

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the main detection backend
uvicorn api.main:app --reload --port 8080
```

### 2. Judge Backend (hybrid_mvp) — port 8001

```powershell
# In a new terminal (same venv)
.\venv\Scripts\Activate.ps1

# Install hybrid_mvp dependencies
pip install -r hybrid_mvp/requirements.txt

# Run the judge server
uvicorn server:app --reload --port 8001 --app-dir hybrid_mvp
```

### 3. Edge Client (optional) — LLM-confirmed violation pipeline

```powershell
# In a new terminal (same venv) — only needed to populate Judge Feed
.\venv\Scripts\Activate.ps1
python hybrid_mvp/edge_client.py
```

> **Note:** The rest of the app works fully without this step. The Judge Feed page will show a clear empty state indicating that `edge_client.py` is not running.

### 4. Frontend (React + Vite) — port 5173

```powershell
# In a new terminal
cd frontend

# Install UI dependencies (first time only)
npm install

# Start the dev server
npm run dev
```

Navigate to `http://localhost:5173` to view the application!

---

## 📁 Project Structure

```text
gridlock-prototype/
├── api/                       ← Main FastAPI backend (port 8080)
│   └── main.py                   /detect/image, /violations/log, /ws/detect/video
├── hybrid_mvp/                ← Judge backend (port 8001) + edge client
│   ├── server.py                 /api/stats, /api/violations, /ws
│   ├── edge_client.py            YOLOv8+ByteTrack → Ollama judge → POST violations
│   └── dashboard.html            Standalone fallback dashboard (legacy)
├── frontend/                  ← React web application (single UI for both backends)
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.js         API client for main backend (api/main.py)
│   │   │   └── judgeClient.js    API client for judge backend (hybrid_mvp/server.py)
│   │   ├── components/        ← Shared UI components
│   │   ├── contexts/          ← Global state (SettingsContext)
│   │   └── pages/
│   │       ├── Dashboard.jsx     Main detection KPIs + violation log
│   │       ├── DetectionStudio.jsx  Upload images/video for YOLO inference
│   │       ├── Settings.jsx      Model thresholds and violation rules
│   │       └── JudgeFeed.jsx     LLM-confirmed violations from hybrid_mvp
├── src/
│   ├── detect.py              ← YOLOv8 inference wrapper
│   ├── violations.py          ← Rule-based violation logic
│   ├── alpr.py                ← PaddleOCR plate reading
│   ├── utils.py               ← Helpers (drawing, logging)
│   └── train.py               ← Training script
├── data/
│   ├── images/{train,val}/    ← Training images
│   └── labels/{train,val}/    ← YOLO format labels
├── models/weights/            ← best.pt after training
├── outputs/
│   ├── logs/                  ← Violation CSVs (main pipeline)
│   └── frames/                ← Saved annotated frames
├── data.yaml                  ← YOLO dataset config
└── requirements.txt
```

---

## 🧠 Detection Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | motorcycle | Two-wheelers |
| 1 | car | Passenger cars |
| 2 | bus | Buses |
| 3 | truck | Trucks/HGVs |
| 4 | person | Pedestrians/riders |
| 5 | helmet | Protective helmets |
| 6 | no_helmet | Head without helmet |

---

## 🚨 Violation Rules

### Main Pipeline (api/main.py)

| Violation | Logic |
|-----------|-------|
| **No Helmet** | Person overlapping motorcycle with no helmet detection |
| **Triple Riding** | 3+ persons overlapping a single motorcycle |
| **Signal Jump** | Vehicle bbox intersects red-light ROI zone |

### Judge Pipeline (hybrid_mvp)

| Violation | Logic |
|-----------|-------|
| **Helmet Check** | YOLOv8 detection → Ollama VLM confirmation |
| **Triple Riding** | YOLOv8 detection → Ollama VLM confirmation |
| **Seatbelt Check** | YOLOv8 detection → Ollama VLM confirmation |
| **Wrong Way** | YOLOv8 detection → Ollama VLM confirmation |
| **Lane Violation** | YOLOv8 detection → Ollama VLM confirmation |
| **Mobile Phone** | YOLOv8 detection → Ollama VLM confirmation |
| **Overloading** | YOLOv8 detection → Ollama VLM confirmation |

---

## 📊 CSV Output Format

```csv
violation_id, plate_number, violation_type, confidence, timestamp, frame_id, image_path
V001, KA03MN5678, NO_HELMET, 0.87, 2026-06-20 11:32:01, 142, outputs/frames/frame_142.jpg
```

---

## 📈 Training

```powershell
# Train from scratch
python src/train.py --epochs 50 --batch 8

# Validate existing weights
python src/train.py --validate-only --weights models/gridlock_v1/weights/best.pt
```

Target metrics: **mAP50 ≥ 0.50** after 50 epochs on validation set.

---

## 🔧 Dataset Sources

Datasets can be fetched directly via API POST requests:

1. **Roboflow (Indian Plates)**
```bash
curl -X POST http://localhost:8080/dataset/fetch/roboflow \
    -H 'Content-Type: application/json' \
    -d '{"api_key": "YOUR_KEY", "workspace": "your-workspace", "project": "indian-number-plates", "version": 1, "format": "yolov8"}'
```

2. **Kaggle (IDD Subset)**
```bash
curl -X POST http://localhost:8080/dataset/fetch/kaggle?dataset_slug=owner/dataset-name
```

---

## 💡 Tech Stack

- **Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **LLM Judge**: [Ollama](https://ollama.ai) (vision-language model for violation confirmation)
- **OCR**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **API Backend**: [FastAPI](https://fastapi.tiangolo.com) (two independent servers)
- **UI Frontend**: [React (Vite)](https://vitejs.dev) (single app consuming both backends)
- **CV**: [OpenCV](https://opencv.org)