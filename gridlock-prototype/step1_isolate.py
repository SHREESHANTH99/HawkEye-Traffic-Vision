import cv2
import sys
from src.alpr import ALPRPipeline

def test_image(alpr, path, name):
    print(f"\n--- Testing {name} ({path}) ---")
    img = cv2.imread(path)
    if img is None:
        print("Could not read image!")
        return
        
    bboxes, tier = alpr.detect_plates(img)
    print(f"Detection Tier: {tier}")
    print(f"Candidates returned: {len(bboxes)}")
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        aspect_ratio = float(w) / float(h) if h > 0 else 0
        print(f"  Candidate {i+1}: coords={bbox}, w={w}, h={h}, aspect_ratio={aspect_ratio:.2f}")
        
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"static/violations/cand_{name}_{i+1}.jpg", crop)
        
    if len(bboxes) > 0:
        print(f"Top candidate selected: Candidate 1 (coords={bboxes[0]})")
    else:
        print("Top candidate selected: None")

def main():
    alpr = ALPRPipeline()
    test_image(alpr, "static/violations/NO_HELMET_v11_f3_1782005662.jpg", "moto1")
    test_image(alpr, "static/violations/NO_HELMET_v160_f267_1782005700.jpg", "moto2")
    test_image(alpr, "static/violations/car_crop.jpg", "car")

if __name__ == "__main__":
    main()
