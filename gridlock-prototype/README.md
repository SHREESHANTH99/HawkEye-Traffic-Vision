# Gridlock — Traffic Violation Detection Prototype

> AI-powered detection of traffic violations in Indian road conditions using YOLOv8 + PaddleOCR.

---

## 🚀 Quick Start

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download IDD dataset
python src/download_idd.py

# 4. Train YOLOv8 (after dataset is ready)
python src/train.py

# 5. Launch Streamlit dashboard
streamlit run src/app.py
```

---

## 📁 Project Structure

```
gridlock-prototype/
├── data/
│   ├── images/{train,val}/    ← training images
│   ├── labels/{train,val}/    ← YOLO format labels
│   └── raw/                   ← downloaded datasets
├── models/
│   └── weights/               ← best.pt after training
├── src/
│   ├── app.py                 ← Streamlit UI
│   ├── detect.py              ← YOLOv8 inference wrapper
│   ├── violations.py          ← rule-based violation logic
│   ├── alpr.py                ← PaddleOCR plate reading
│   ├── utils.py               ← helpers (drawing, logging)
│   ├── train.py               ← training script
│   └── download_idd.py        ← dataset download
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

```
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

1. **IDD** — [Kaggle: sovitrath/indian-driving-dataset](https://www.kaggle.com/datasets/sovitrath/indian-driving-dataset-segmentation-part-2) (via `src/download_idd.py`)
2. **Indian License Plates** — [Roboflow Universe](https://universe.roboflow.com) → search "Indian License Plate" → Export YOLOv8
3. **Custom Helmet/Triple-Riding** — Extract frames from dashcam video, annotate on [Roboflow](https://app.roboflow.com), export YOLOv8

---

## 💡 Tech Stack

- **Detection**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **OCR**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- **UI**: [Streamlit](https://streamlit.io)
- **CV**: [OpenCV](https://opencv.org)
