"""
train.py — YOLOv8 training script for Gridlock prototype
  Run this after dataset is prepared in data/ folder.
  Saves best weights to models/gridlock_v1/weights/best.pt
"""

from ultralytics import YOLO
import torch
from pathlib import Path

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_YAML   = "data.yaml"           # relative to project root
BASE_MODEL  = "yolov8n.pt"          # nano=fastest, s/m for better accuracy
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 8
DEVICE      = 0 if torch.cuda.is_available() else "cpu"
PROJECT_DIR = "models"
RUN_NAME    = "gridlock_v1"

# ─── Train ────────────────────────────────────────────────────────────────────

def train():
    print(f"[Train] Device  : {DEVICE}")
    print(f"[Train] Model   : {BASE_MODEL}")
    print(f"[Train] Epochs  : {EPOCHS}")
    print(f"[Train] Batch   : {BATCH_SIZE}")
    print(f"[Train] ImgSize : {IMG_SIZE}")
    print("-" * 40)

    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=10,        # early stop if no val improvement
        save=True,
        save_period=10,     # save checkpoint every 10 epochs
        plots=True,         # generate results.png, confusion_matrix.png
        verbose=True,
        cache=False,        # set True to cache images in RAM (faster if RAM ≥ 8GB)
        workers=4,
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        mosaic=1.0,         # data augmentation
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n✅ Training complete!")
    print(f"   Best weights : {best_weights}")
    print(f"   mAP50       : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"   mAP50-95    : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")

    return best_weights


# ─── Validate ─────────────────────────────────────────────────────────────────

def validate(weights_path: str = None):
    if weights_path is None:
        weights_path = f"models/{RUN_NAME}/weights/best.pt"

    print(f"\n[Validate] Loading weights: {weights_path}")
    model = YOLO(weights_path)
    metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, device=DEVICE)

    print(f"\n📊 Validation Results:")
    print(f"   mAP50       : {metrics.box.map50:.4f}")
    print(f"   mAP50-95    : {metrics.box.map:.4f}")
    print(f"   Precision   : {metrics.box.mp:.4f}")
    print(f"   Recall      : {metrics.box.mr:.4f}")

    return metrics


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gridlock YOLOv8 Training")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training, only validate existing weights")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights for validation")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch

    if args.validate_only:
        validate(args.weights)
    else:
        best = train()
        validate(str(best))
