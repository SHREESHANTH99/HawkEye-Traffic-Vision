# HawkEye Traffic Vision (Gridlock)

An advanced, AI-powered computer vision pipeline for real-time traffic violation detection and automated enforcement. HawkEye leverages state-of-the-art object detection (YOLOv8), multi-object tracking (ByteTrack), and Optical Character Recognition (EasyOCR) alongside a modern React frontend to deliver a complete edge-to-cloud smart city solution.

## 🌟 Key Features

### 1. Multi-Violation Detection Engine
- **No-Helmet Detection**: Utilizes a tiered approach to detect riders without helmets on two-wheelers.
- **Triple Riding Detection**: Calculates intersection-over-union (IoU) between bounding boxes to accurately count riders on a single motorcycle.
- **Automated License Plate Recognition (ALPR)**: Employs `EasyOCR` combined with Indian-plate-specific regex validation logic to extract, clean, and validate license plates. Flags uncertain reads natively in the UI.

### 2. Dual-Backend Architecture
The system employs two distinct, concurrently running FastAPI servers to simulate a robust edge-to-cloud environment:
- **Main API (`api/main.py` on Port 8000)**: The pull-based detection pipeline. Exposes endpoints for synchronous image detection, standalone ALPR processing, and historical violation fetching.
- **Hybrid "Judge" MVP (`hybrid_mvp/server.py` on Port 8001)**: The push-based cloud aggregator. Receives confirmed violation payloads from edge clients, logs them into an SQLite database (`traffic_analytics.db`), saves violation crops, and streams them to the UI via WebSockets.

### 3. Edge Processing Client
- **`edge_client.py`**: A standalone script that ingests video files, applies YOLOv8 and ByteTrack to track vehicles across frames, crops the regions of interest, and verifies violations. Includes integration with **Moondream (via Ollama)** as an experimental Vision-Language Model (VLM) "Judge" to double-check ambiguous frames before alerting the cloud.

### 4. Modern React Dashboard
- **Tech Stack**: Built with Vite and React, heavily styled with custom dark-mode CSS variables and glassmorphic aesthetics.
- **Judge Feed**: A real-time, WebSocket-powered feed displaying verified violations as they are broadcasted from the Hybrid MVP backend.
- **Live Violation Log**: Tabular overview of historical violations with direct ALPR analysis and CSV export capabilities.
- **Dynamic Configuration**: UI sliders to adjust Confidence thresholds, Person-Vehicle Overlap bounds, and Triple-Riding triggers on the fly.
- **Data Integrity**: Visually alerts reviewers to malformed or uncertain plate reads (e.g., missing characters) with distinct styling and tooltips.

---

## 🏗️ Project Structure

```text
.
├── gridlock-prototype/
│   ├── api/
│   │   └── main.py                 # Primary FastAPI Backend (Port 8000)
│   ├── frontend/                   # React + Vite Dashboard
│   │   ├── src/
│   │   │   ├── components/         # Reusable UI components (Sidebar, ViolationLog)
│   │   │   └── pages/              # Primary views (JudgeFeed, Settings)
│   ├── hybrid_mvp/
│   │   ├── server.py               # Aggregator FastAPI Backend (Port 8001)
│   │   └── edge_client.py          # Video processing & tracking client
│   ├── src/
│   │   ├── app.py                  # Core YOLOv8 inference wrapper
│   │   └── alpr.py                 # EasyOCR pipeline & Regex validation
│   └── requirements.txt            # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+**
- **Node.js 18+** (for the React Frontend)
- **Ollama** (Optional, required only if using the Moondream VLM Judge in the edge client)

### 1. Setup the Python Environment

Navigate to the prototype directory and install the dependencies. *Note: EasyOCR and YOLOv8 will automatically utilize your GPU if CUDA is available.*

```bash
cd gridlock-prototype
python -m venv venv
# Activate virtual environment (Windows)
venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Backends

You will need two separate terminal windows for the backends. Ensure your virtual environment is activated in both.

**Terminal 1 (Main API - Port 8000):**
```bash
python -m uvicorn api.main:app --port 8000
```

**Terminal 2 (Hybrid Judge Server - Port 8001):**
```bash
python hybrid_mvp/server.py
```

### 3. Start the React Frontend

Open a third terminal window to start the Vite development server.

```bash
cd gridlock-prototype/frontend
npm install
npm run dev
```

The application will be accessible at `http://localhost:5173`.

---

## 🚦 Running the Edge Pipeline

To simulate real-time processing of a traffic feed, you can run the edge client against a video file. This script tracks vehicles, processes violations, runs ALPR, and posts the results to the Hybrid Judge Server (which then broadcasts to your React UI).

```bash
cd gridlock-prototype
# Activate virtual environment
venv\Scripts\activate

# Run the edge client
python hybrid_mvp/edge_client.py
```

As the script processes frames, watch the **Judge Feed** tab in your React UI populate with real-time violation crops and extracted license plates.

---

## 🧠 ALPR & Regex Validation

The Indian license plate recognition pipeline in `src/alpr.py` is highly tuned:
1. **Extraction**: Crops the bottom 35% of a tracked vehicle.
2. **Enhancement**: Applies CLAHE and Gaussian Blur.
3. **OCR**: Extracts raw text using PyTorch-backed `EasyOCR`.
4. **Correction**: Applies positional heuristics (e.g., swapping `0` for `O` in letter positions).
5. **Strict Validation**: Matches against a strict `^[A-Z]{2}\s*[0-9]{1,2}\s*[A-Z]{1,3}\s*[0-9]{4}$` Regex pattern.
6. **UI Feedback**: If the extraction fails the strict validation (e.g., a dropped digit resulting in a 9-character plate), it is passed to the UI with a `plate_valid: false` flag, where it is visually highlighted with a red asterisk `*` for human review.

---

## 🛠️ Technology Stack

- **Computer Vision**: Ultralytics YOLOv8, ByteTrack, OpenCV, EasyOCR
- **Backend APIs**: FastAPI, Pydantic, SQLAlchemy, WebSockets
- **Frontend**: React.js, Vite, Vanilla CSS Variables
- **Experimental AI**: Moondream1 (Small Vision Language Model) via Ollama
