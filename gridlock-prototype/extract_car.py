import cv2
from ultralytics import YOLO
import sys

def main():
    model = YOLO("yolov8s.pt")
    cap = cv2.VideoCapture("hybrid_mvp/test_traffic.mp4")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # classes: 2 is car in coco
        results = model(frame, classes=[2], verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (x2 - x1) > 100 and (y2 - y1) > 100:  # reasonably sized car
                    crop = frame[y1:y2, x1:x2]
                    cv2.imwrite("static/violations/car_crop.jpg", crop)
                    print("Saved car crop from frame", frame_count)
                    return
    print("No car found")

if __name__ == "__main__":
    main()
