import cv2
import sys
import re
from src.alpr import ALPRPipeline

def main():
    alpr = ALPRPipeline()
    
    images = [
        ("clear_plate.jpg", "MH 12 AB 1234"),
        ("plate2.jpg", "KA 03 MN 5678"),
        ("plate3.jpg", "DL 8C AB 0001")
    ]
    
    for filename, expected in images:
        print(f"=== Testing: {expected} ({filename}) ===")
        img = cv2.imread(filename)
        res = alpr.read_plate_from_crop(img)
        print(f"raw_text:       {res.raw_text}")
        print(f"plate_number:   {res.plate_number}")
        print(f"plate_valid:    {res.plate_valid}")
        print(f"confidence:     {res.confidence}")
        print(f"ocr_confidence_raw: {res.ocr_confidence_raw}\n")

if __name__ == "__main__":
    main()
