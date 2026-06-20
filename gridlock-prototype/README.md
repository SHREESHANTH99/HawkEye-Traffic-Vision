# Gridlock — Traffic Violation Detection Prototype

> AI-powered detection of traffic violations in Indian road conditions using YOLOv8 + PaddleOCR.

---

## 🚀 Quick Start

Gridlock features a decoupled architecture with a FastAPI backend for heavy AI inference and a modern React (Vite) frontend for a responsive, multi-page Dashboard.

### 1. Start the Backend (FastAPI + YOLO)

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the backend on port 8000
uvicorn api.main:app --reload --port 8000
```

### 2. Start the Frontend (React + Vite)

```powershell
# Open a new terminal
cd frontend

# Install UI dependencies
npm install

# Start the dev server
npm run dev
```

Navigate to `http://localhost:5173` to view the Dashboard!

---

## 📁 Project Structure

```text
gridlock-prototype/
├── api/                       ← FastAPI backend routes and models
│   └── main.py
├── frontend/                  ← React web application
│   ├── src/
│   │   ├── components/        ← React UI components
│   │   ├── contexts/          ← Global state (SettingsContext)
│   │   ├── pages/             ← Route pages (Dashboard, Detect, Settings)
│   │   └── api/               ← Client requests to FastAPI
├── data/
│   ├── images/{train,val}/    ← training images
│   └── labels/{train,val}/    ← YOLO format labels
├── models/
│   └── weights/               ← best.pt after training
├── src/
│   ├── detect.py              ← YOLOv8 inference wrapper
│   ├── violations.py          ← rule-based violation logic
│   ├── alpr.py                ← PaddleOCR plate reading
│   ├── utils.py               ← helpers (drawing, logging)
│   └── train.py               ← training script
├── outputs/
│   ├── logs/                  ← violation CSVs
│   └── frames/                ← saved annotated frames
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

| Violation | Logic |
|-----------|-------|
| **No Helmet** | Person overlapping motorcycle with no helmet detection |
| **Triple Riding** | 3+ persons overlapping a single motorcycle |
| **Signal Jump** | Vehicle bbox intersects red-light ROI zone |

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

Datasets can now be fetched directly via API POST requests instead of manual downloads:

1. **Roboflow (Indian Plates)**
```bash
curl -X POST http://localhost:8000/dataset/fetch/roboflow \
    -H 'Content-Type: application/json' \
    -d '{"api_key": "YOUR_KEY", "workspace": "your-workspace", "project": "indian-number-plates", "version": 1, "format": "yolov8"}'
```

2. **Kaggle (IDD Subset)**
```bash
curl -X POST http://localhost:8000/dataset/fetch/kaggle?dataset_slug=owner/dataset-name
```

---

## 💡 Tech Stack

- **Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **OCR**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **API Backend**: [FastAPI](https://fastapi.tiangolo.com)
- **UI Frontend**: [React (Vite)](https://vitejs.dev)
- **CV**: [OpenCV](https://opencv.org)