# -*- coding: utf-8 -*-
"""
smoke_test.py — Quick sanity check that all imports and core functionality work.
Run after: pip install -r requirements.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  Gridlock — Dependency Smoke Test")
print("=" * 55)

results = []

def check(label, fn):
    try:
        fn()
        print(f"  [OK]  {label}")
        results.append((label, True))
    except Exception as e:
        print(f"  [XX]  {label}: {e}")
        results.append((label, False))

# Core packages
check("numpy",          lambda: __import__("numpy"))
check("pandas",         lambda: __import__("pandas"))
check("cv2 (OpenCV)",   lambda: __import__("cv2"))
check("PIL (Pillow)",   lambda: __import__("PIL"))
check("streamlit",      lambda: __import__("streamlit"))
check("torch",          lambda: __import__("torch"))
check("ultralytics",    lambda: __import__("ultralytics"))
check("matplotlib",     lambda: __import__("matplotlib"))
check("kagglehub",      lambda: __import__("kagglehub"))

# CUDA check
def cuda_check():
    import torch
    avail = torch.cuda.is_available()
    print(f"        → CUDA available: {avail}", end="")
    if avail:
        print(f"  [{torch.cuda.get_device_name(0)}]")
    else:
        print(" (CPU mode)")
check("torch CUDA",     cuda_check)

# YOLOv8 model load
def yolo_check():
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")   # downloads ~6MB on first run
    print(f"        → Model loaded: {type(m).__name__}", end="")
check("YOLOv8 nano load", yolo_check)

# Src modules
check("src.utils",      lambda: __import__("src.utils"))
check("src.detect",     lambda: __import__("src.detect"))
check("src.violations", lambda: __import__("src.violations"))
check("src.alpr",       lambda: __import__("src.alpr"))

print()
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"  Result: {passed}/{total} checks passed")

if passed == total:
    print("\n  >> All good! Run:  python run_app.py")
else:
    failed = [l for l, ok in results if not ok]
    print(f"\n  !! Failed: {', '.join(failed)}")
    print("  Try: pip install -r requirements.txt")
print("=" * 55)
