import cv2
import json
from pathlib import Path
VIDEO_PATH  = "test_traffic.mp4"
CONFIG_PATH = "zones_config.json"
points: list[tuple[int, int]] = []
frame_display = None


def mouse_callback(event, x: int, y: int, flags, param) -> None:
    global points, frame_display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"  [ZONE] Point {len(points)} added: ({x}, {y})")

        else:
            print("  [ZONE] Already have 4 points. Press R to reset or S to save.")


def draw_overlay(base_frame) -> None:
    disp = base_frame.copy()
    color_point = (0, 255, 255)
    color_line  = (0, 165, 255)
    color_fill  = (0, 255, 0)

    for i, pt in enumerate(points):
        cv2.circle(disp, pt, 7, color_point, -1)
        cv2.putText(disp, str(i + 1), (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_point, 2)

        if i > 0:
            cv2.line(disp, points[i - 1], pt, color_line, 2)

    if len(points) == 4:
        cv2.line(disp, points[3], points[0], color_fill, 2)
        cv2.polylines(disp, [__import__("numpy").array(points)], True, color_fill, 2)
        cv2.putText(disp, "POLYGON COMPLETE — Press S to save",
                    (20, disp.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_fill, 2)

    else:
        remaining = 4 - len(points)
        cv2.putText(disp, f"Click {remaining} more point(s) to complete ROI",
                    (20, disp.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    instructions = [
        "Left-click : Add ROI point",
        "R          : Reset points",
        "S          : Save & Exit",
        "Q          : Quit without saving",
    ]

    for i, txt in enumerate(instructions):
        cv2.putText(disp, txt, (10, 26 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    cv2.imshow("HawkEye Zone Setup", disp)


def save_config() -> None:
    if len(points) != 4:
        print(f"[ERROR] Need exactly 4 points, have {len(points)}. Not saving.")
        return

    config = {
        "roi_polygon": points,
        "dominant_direction": "right",
        "min_vehicle_size_px": 60,
        "notes": (
            "dominant_direction = direction vehicles are SUPPOSED to travel. "
            "Change to 'left', 'up', or 'down' if needed."
        ),
    }
    Path(CONFIG_PATH).write_text(json.dumps(config, indent=2))
    print(f"\n[SAVED] {CONFIG_PATH}")
    print(f"        ROI points  : {points}")
    print(f"        Direction   : {config['dominant_direction']}")
    print("  Edit 'dominant_direction' in zones_config.json if needed.")


def main() -> None:
    global points, frame_display
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open '{VIDEO_PATH}'")
        return

    ret, base_frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read first frame.")
        return

    h, w = base_frame.shape[:2]

    if w > 1280:
        scale = 1280 / w
        base_frame = cv2.resize(base_frame, (1280, int(h * scale)))

    print("\n=== HawkEye Zone Calibration ===")
    print(f"  Video   : {VIDEO_PATH}  ({base_frame.shape[1]}x{base_frame.shape[0]})")
    print(f"  Config  : {CONFIG_PATH}")
    print("  Click 4 points to define the ROI polygon.")
    print("  Press S to save, R to reset, Q to quit.\n")
    cv2.namedWindow("HawkEye Zone Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HawkEye Zone Setup", base_frame.shape[1], base_frame.shape[0])
    cv2.setMouseCallback("HawkEye Zone Setup", mouse_callback)

    while True:
        draw_overlay(base_frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            print("[QUIT] Exiting without saving.")
            break

        elif key == ord("r"):
            points = []
            print("  [RESET] Points cleared.")

        elif key == ord("s"):
            save_config()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
