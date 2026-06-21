import cv2
import glob
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.alpr import ALPRPipeline

def test_heuristic():
    alpr = ALPRPipeline()
    # Find some test images
    img_paths = glob.glob("static/violations/*.jpg")
    if not img_paths:
        print("No test images found in static/violations/")
        return

    print(f"Testing {len(img_paths)} images...")
    for path in img_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        
        bboxes, tier = alpr.detect_plates(frame)
        print(f"\nImage: {path}")
        print(f"Tier: {tier}")
        print(f"BBoxes: {bboxes}")
        
        # Read the plate using the full pipeline
        res = alpr.read_plate(path)
        if res:
            print(f"Read Plate: {res.plate_number} (Conf: {res.confidence})")
        else:
            print("Read Plate: None")

if __name__ == "__main__":
    test_heuristic()
