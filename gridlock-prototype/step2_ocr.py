import cv2
import json
from src.alpr import ALPRPipeline

def main():
    alpr = ALPRPipeline()
    
    print("\n=== STEP 2: Clear Plate ===")
    clear_img = cv2.imread("static/violations/clear_plate.jpg")
    res_clear = alpr.read_plate_from_crop(clear_img)
    print(f"Result: raw_text='{res_clear.raw_text}', plate_number='{res_clear.plate_number}', confidence={res_clear.confidence}, detection_tier='{res_clear.detection_tier}', ocr_confidence_raw={res_clear.ocr_confidence_raw}")

    print("\n=== STEP 3: Double Row Plate ===")
    double_img = cv2.imread("static/violations/double_plate.jpg")
    
    # Before double row split (naive single pass)
    raw_text, conf = alpr._ocr_plate(double_img)
    print(f"BEFORE Split (Naive OCR): {raw_text} (Conf: {conf})")
    
    # After double row split (via read_plate_from_crop)
    res_double = alpr.read_plate_from_crop(double_img)
    print(f"AFTER Split (read_plate_from_crop): {res_double.plate_number}")

if __name__ == "__main__":
    main()
