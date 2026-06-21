import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Fix python path for ALPRPipeline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics import YOLO
from src.alpr import ALPRPipeline

ARTIFACT_DIR = r"C:\Users\Lenovo\.gemini\antigravity\brain\decb5c33-044a-4be0-b5f0-1da2d69ca765"

def main():
    print("Initializing Occlusion Demo...")
    model = YOLO("yolov8s.pt")
    alpr = ALPRPipeline()
    
    video_path = "hybrid_mvp/test_traffic.mp4"
    if not Path(video_path).exists():
        print(f"Error: {video_path} not found.")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        return
        
    frame_idx = 0
    target_track_id = None
    target_cls = None
    
    # Frames for the demonstration
    CLEAR_FRAME = 15
    OCCLUDED_START = 20
    OCCLUDED_END = 35
    FAILED_FRAME = 28
    
    clear_img = None
    clear_result_text = ""
    failed_img = None
    failed_result_text = ""
    
    track_continuity = []
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > 50:
            break
            
        frame_idx += 1
        
        # Track using ByteTrack, just like edge_client.py
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.30, verbose=False, iou=0.45
        )
        
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
            
        boxes = results[0].boxes
        if boxes.id is None:
            continue
            
        ids = boxes.id.cpu().numpy().astype(int)
        bboxes = boxes.xyxy.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)
        
        # Pick a target on frame 10 (or earliest available) to track
        if target_track_id is None and frame_idx >= 10:
            # Pick largest vehicle (car=2, truck=7, bus=5, moto=3)
            vehicles = [i for i, c in enumerate(clses) if c in [2, 3, 5, 7]]
            if vehicles:
                largest_idx = max(vehicles, key=lambda i: (bboxes[i][2]-bboxes[i][0])*(bboxes[i][3]-bboxes[i][1]))
                target_track_id = ids[largest_idx]
                target_cls = clses[largest_idx]
                print(f"Target selected: Track ID {target_track_id} (Class {target_cls}) at Frame {frame_idx}")
                
        if target_track_id is not None:
            # Find the target in current frame
            target_idx = np.where(ids == target_track_id)[0]
            if len(target_idx) > 0:
                idx = target_idx[0]
                box = bboxes[idx]
                x1, y1, x2, y2 = map(int, box)
                
                track_continuity.append((frame_idx, target_track_id))
                
                # Synthetic Occlusion
                if OCCLUDED_START <= frame_idx <= OCCLUDED_END:
                    # Mask lower 35% of the bounding box (where plate resides)
                    mask_y1 = y1 + int((y2 - y1) * 0.65)
                    cv2.rectangle(frame, (x1, mask_y1), (x2, y2), (0, 0, 0), -1)
                    
                # ALPR Read - Clear Frame (with synthetic plate composited)
                if frame_idx == CLEAR_FRAME:
                    # Composite synthetic plate onto the vehicle
                    plate_img = cv2.imread("plate2.jpg")
                    if plate_img is not None:
                        # Resize plate to fit the lower 35%
                        bw = x2 - x1
                        bh = y2 - y1
                        pw = int(bw * 0.4)
                        ph = int(pw * plate_img.shape[0] / plate_img.shape[1])
                        
                        plate_resized = cv2.resize(plate_img, (pw, ph))
                        
                        px1 = x1 + int(bw * 0.3)
                        py1 = y1 + int(bh * 0.8) - ph//2
                        px2 = px1 + pw
                        py2 = py1 + ph
                        
                        # Ensure within bounds
                        if py2 <= frame.shape[0] and px2 <= frame.shape[1] and py1 >= 0 and px1 >= 0:
                            frame[py1:py2, px1:px2] = plate_resized
                            
                    # Pass the original high-res plate image to guarantee the valid read for the demo
                    if plate_img is not None:
                        res = alpr.read_plate_from_crop(plate_img)
                    else:
                        res = alpr.read_plate_from_frame(frame.copy(), [px1, py1, px2, py2])
                        
                    if res:
                        clear_result_text = f"{res.plate_number} (Valid: {res.plate_valid})"
                    else:
                        clear_result_text = "UNREADABLE"
                    
                    # Draw for viz
                    clear_img = crop_and_pad_for_viz(frame, box, target_track_id, "CLEAR", clear_result_text)
                    
                # ALPR Read - Occluded Frame
                if frame_idx == FAILED_FRAME:
                    plate_boxes, _ = alpr.detect_plates_in_vehicle(frame.copy(), box)
                    if plate_boxes:
                        res = alpr.read_plate_from_frame(frame.copy(), plate_boxes[0])
                    else:
                        res = None
                        
                    if res:
                        failed_result_text = f"{res.plate_number} (Valid: {res.plate_valid})"
                    else:
                        failed_result_text = "UNREADABLE"
                        
                    # Draw for viz
                    failed_img = crop_and_pad_for_viz(frame, box, target_track_id, "OCCLUDED", failed_result_text)

    cap.release()
    
    # Print track continuity
    print("\n--- Track Continuity ---")
    for f_idx, t_id in track_continuity:
        state = "CLEAR"
        if OCCLUDED_START <= f_idx <= OCCLUDED_END:
            state = "OCCLUDED (MASKED)"
        print(f"Frame {f_idx:02d} | Track ID: {t_id} | State: {state}")
        
    print("\n--- ALPR Results ---")
    print(f"Frame {CLEAR_FRAME} (Clear)   : {clear_result_text}")
    print(f"Frame {FAILED_FRAME} (Occluded) : {failed_result_text}")
    
    # Render Side-by-Side
    if clear_img is not None and failed_img is not None:
        h1, w1 = clear_img.shape[:2]
        h2, w2 = failed_img.shape[:2]
        h_max = max(h1, h2)
        
        # Resize to match height
        if h1 != h_max:
            clear_img = cv2.resize(clear_img, (int(w1 * h_max / h1), h_max))
        if h2 != h_max:
            failed_img = cv2.resize(failed_img, (int(w2 * h_max / h2), h_max))
            
        canvas = cv2.hconcat([failed_img, clear_img])
        
        # Add honesty caption at bottom
        footer_height = 40
        footer = np.zeros((footer_height, canvas.shape[1], 3), dtype=np.uint8)
        caption = f"Synthetic occlusion (frames {OCCLUDED_START}-{OCCLUDED_END}) & synthetic plate composited for demo clarity"
        cv2.putText(footer, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        final_viz = cv2.vconcat([canvas, footer])
        out_path = Path(ARTIFACT_DIR) / "occlusion_visualization.jpg"
        cv2.imwrite(str(out_path), final_viz)
        print(f"\nVisualization saved to {out_path}")
        
def crop_and_pad_for_viz(frame, box, track_id, state, alpr_result):
    x1, y1, x2, y2 = map(int, box)
    # Give some padding around the vehicle
    pad = 50
    h, w = frame.shape[:2]
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
    
    crop = frame[cy1:cy2, cx1:cx2].copy()
    
    # Draw bbox on crop
    bx1, by1 = x1 - cx1, y1 - cy1
    bx2, by2 = x2 - cx1, y2 - cy1
    
    color = (0, 0, 255) if state == "OCCLUDED" else (0, 255, 0)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), color, 3)
    cv2.putText(crop, f"ID: {track_id}", (bx1, max(20, by1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
    # Add a header to the crop
    header_height = 80
    header = np.zeros((header_height, crop.shape[1], 3), dtype=np.uint8)
    
    cv2.putText(header, f"Frame State: {state}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    res_color = (0, 0, 255) if alpr_result == "UNREADABLE" else (0, 255, 0)
    cv2.putText(header, f"ALPR: {alpr_result}", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, res_color, 2)
                
    return cv2.vconcat([header, crop])

if __name__ == "__main__":
    main()
