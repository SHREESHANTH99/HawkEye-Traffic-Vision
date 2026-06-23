import base64
import cv2
import io
import numpy as np
import queue
import requests
import sys
import threading
import time
from pathlib import Path
from ultralytics import YOLO
import os
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.alpr import ALPRPipeline

except ImportError:
    ALPRPipeline = None


VIDEO_PATH = "hybrid_mvp/test_traffic.mp4"
MODEL_PATH = "yolov8s.pt"
SERVER_URL = os.environ.get("SERVER_URL", "https://hawkeye-judge-backend.onrender.com/api/violation")
HELMET_MODEL_URL   = "https://raw.githubusercontent.com/Viddesh1/Bike-Helmet-Detectionv2/main/weights/best.pt"
HELMET_MODEL_LOCAL = Path("models") / "bike_helmet_yolov8.pt"
HELMET_CONF        = 0.40
CONFIRMED_DIR = Path("violations_local")
REVIEW_DIR    = Path("violations_review")
CONFIRMED_DIR.mkdir(exist_ok=True)
REVIEW_DIR.mkdir(exist_ok=True)
USE_MOONDREAM_FALLBACK = True
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL   = "moondream:latest"
OLLAMA_TIMEOUT = 45.0
_HELMET_PROMPT = (
    "Look at the rider's head in this image. Are they wearing a helmet? "
    "Describe specifically whether a helmet is visible on the head."
)
_HELMET_NO  = ["no helmet", "not wearing a helmet", "without a helmet",
               "bare head", "no head protection", "doesn't appear to be wearing a helmet"]

_HELMET_YES = ["wearing a helmet", "helmet is visible", "has a helmet",
               "helmet on", "protective helmet", "appears to be wearing a helmet"]


CLS_PERSON     = 0
CLS_BICYCLE    = 1
CLS_CAR        = 2
CLS_MOTORCYCLE = 3
CLS_BUS        = 5
CLS_TRUCK      = 7
BIKE_CLASSES  = {CLS_MOTORCYCLE, CLS_BICYCLE}
MOTOR_CLASSES = {CLS_CAR, CLS_BUS, CLS_TRUCK}
ALL_CLASSES   = [CLS_PERSON, CLS_BICYCLE, CLS_CAR, CLS_MOTORCYCLE, CLS_BUS, CLS_TRUCK]
CLS_LABEL = {
    CLS_PERSON:     "PERSON",
    CLS_BICYCLE:    "BICYCLE",
    CLS_CAR:        "CAR",
    CLS_MOTORCYCLE: "MOTO",
    CLS_BUS:        "BUS",
    CLS_TRUCK:      "TRUCK",
}
CONF_THRESHOLD  = 0.30
RESIZE_WIDTH    = 640
HEAD_PAD_RATIO  = 0.35
MIN_VEH_PX      = 40
PROCESS_EVERY_N = 1
MEMORY_FRAMES   = 15
COLORS = {
    "TRIPLE_RIDING": (0,   0,   255),
    "NO_HELMET":     (0,   80,  255),
    "CHECKING":      (0,   180, 255),
    "OK":            (0,   200, 0),
    "SMALL":         (60,  60,  60),
    "MEMORY":        (120, 120, 120),
    "SCAN":          (160, 160, 0),
}
_STOP = object()


def crop_region(frame: np.ndarray, box: list,
                 pad: int = 20, top_extra_ratio: float = 0.0) -> np.ndarray:
    h, w   = frame.shape[:2]
    bh     = box[3] - box[1]
    top_ex = int(bh * top_extra_ratio)
    x1 = max(0, int(box[0]) - pad)
    y1 = max(0, int(box[1]) - pad - top_ex)
    x2 = min(w, int(box[2]) + pad)
    y2 = min(h, int(box[3]) + pad)
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else frame[:10, :10].copy()


def enhance_crop(img: np.ndarray, min_side: int = 224) -> np.ndarray:
    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return np.zeros((min_side, min_side, 3), dtype=np.uint8)

    if min(h, w) < min_side:
        scale = min_side / min(h, w)
        img = cv2.resize(img,
                          (max(int(w * scale), min_side), max(int(h * scale), min_side)),
                          interpolation=cv2.INTER_LANCZOS4)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def encode_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    img = enhance_crop(img)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])

    if not ok:
        raise RuntimeError("JPEG encode failed")

    return buf.tobytes()


def boxes_overlap(boxA: list, boxB: list) -> bool:
    return (boxA[0] < boxB[2] and boxA[2] > boxB[0] and
            boxA[1] < boxB[3] and boxA[3] > boxB[1])


def ensure_helmet_weights() -> Path | None:
    if HELMET_MODEL_LOCAL.exists() and HELMET_MODEL_LOCAL.stat().st_size > 1_000_000:
        return HELMET_MODEL_LOCAL

    HELMET_MODEL_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading helmet weights (~88MB, first run only) from:\n       {HELMET_MODEL_URL}")

    try:
        with requests.get(HELMET_MODEL_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = HELMET_MODEL_LOCAL.with_suffix(".part")

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

            tmp.rename(HELMET_MODEL_LOCAL)

        size_mb = HELMET_MODEL_LOCAL.stat().st_size / 1e6
        print(f"[INFO] Saved helmet weights → {HELMET_MODEL_LOCAL} ({size_mb:.1f} MB)")
        return HELMET_MODEL_LOCAL

    except Exception as e:
        print(f"[WARN] Could not download helmet weights ({e}).")
        return None


def load_helmet_model(device: str):
    weights = ensure_helmet_weights()

    if weights is None:
        print("[WARN] No helmet weights available. Falling back to Moondream.")
        return None, set(), set()

    try:
        print(f"[INFO] Loading helmet model: {weights}")
        hmodel = YOLO(str(weights))
        hmodel.to(device)
        print(f"[INFO] Helmet model classes: {hmodel.names}")
        no_helmet_ids = {
            cid for cid, name in hmodel.names.items()
            if any(kw in name.lower() for kw in ["no_helmet", "no-helmet", "without", "bare", "no helmet"])
        }
        helmet_ids = {
            cid for cid, name in hmodel.names.items()
            if "helmet" in name.lower() and cid not in no_helmet_ids
        }
        print(f"[INFO] Helmet IDs={helmet_ids}  No-helmet IDs={no_helmet_ids}")
        return hmodel, no_helmet_ids, helmet_ids

    except Exception as e:
        print(f"[WARN] Could not load helmet model ({e}). Falling back to Moondream.")
        return None, set(), set()


def check_helmet_yolo(helmet_model, crop_img: np.ndarray,
                       no_helmet_ids: set, helmet_ids: set) -> str:
    if helmet_model is None or crop_img.size == 0:
        return "SKIP"

    try:
        results = helmet_model.predict(crop_img, conf=HELMET_CONF, verbose=False, imgsz=320)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return "NO_HELMET"

        boxes = results[0].boxes
        clses = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        relevant = [(c, cf) for c, cf in zip(clses, confs) if c in helmet_ids or c in no_helmet_ids]

        if not relevant:
            return "NO_HELMET"

        best_cls = max(relevant, key=lambda x: x[1])[0]
        return "NO_HELMET" if best_cls in no_helmet_ids else "HELMET"

    except Exception as e:
        print(f"  [HELMET ERR] {e}")
        return "SKIP"


def parse_helmet_text(text: str) -> str:
    t = text.lower()

    if not t.strip():
        return "SKIP"

    no  = any(k in t for k in _HELMET_NO)
    if no: return "NO_HELMET"
    yes = any(k in t for k in _HELMET_YES)
    if yes: return "HELMET"
    return "SKIP"


def check_helmet_moondream(img_bytes: bytes) -> str:
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "model":   OLLAMA_MODEL,
        "prompt":  _HELMET_PROMPT,
        "images":  [img_b64],
        "stream":  False,
        "options": {"temperature": 0.0},
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if data.get("eval_count", 0) <= 1:
            return "SKIP"

        return parse_helmet_text(data.get("response", ""))

    except requests.exceptions.ConnectionError:
        return "SKIP"

    except requests.exceptions.Timeout:
        return "SKIP"

    except Exception as e:
        print(f"  [MOON ERR] {e}")
        return "SKIP"


def save_review_image(vehicle_id: int, violation_type: str,
                       crop_bytes: bytes, frame_id: int, verdict: str) -> None:
    fname = REVIEW_DIR / f"{violation_type}_{verdict}_id{vehicle_id}_f{frame_id}.jpg"
    fname.write_bytes(crop_bytes)


def save_confirmed(vehicle_id: int, violation_type: str,
                    crop_bytes: bytes, frame_id: int) -> None:
    fname = CONFIRMED_DIR / f"{violation_type}_id{vehicle_id}_f{frame_id}.jpg"
    fname.write_bytes(crop_bytes)


def post_to_server(vehicle_id: int, violation_type: str,
                    crop_bytes: bytes, frame_id: int, plate_text: str = "UNKNOWN", plate_valid: bool = False) -> None:
    try:
        resp = requests.post(
            SERVER_URL,
            files={"image": ("crop.jpg", io.BytesIO(crop_bytes), "image/jpeg")},
            data={"vehicle_id": str(vehicle_id), "violation_type": violation_type,
                  "frame_id": str(frame_id), "plate_text": plate_text,
                  "plate_valid": str(plate_valid).lower()},
            timeout=4.0,
        )
        v = resp.json().get("verdict", "?") if resp.status_code == 200 else f"HTTP{resp.status_code}"
        print(f"  [SRV ] {violation_type} veh={vehicle_id} → {v}")

    except Exception:
        pass


def confirm_violation(vehicle_id: int, violation_type: str, crop_bytes: bytes, frame_id: int,
                       confirmed_set: set, lock: threading.Lock, counters: dict, alpr=None) -> bool:
    with lock:
        key = (vehicle_id, violation_type)

        if key in confirmed_set:
            return False

        confirmed_set.add(key)
        counters["confirmed"] += 1

    plate_text = "UNKNOWN"
    plate_valid = False

    if alpr is not None:
        try:
            crop_img = cv2.imdecode(np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR)
            h, w = crop_img.shape[:2]
            plate_region_y = int(h * 0.65)
            lower_crop = crop_img[plate_region_y:h, 0:w]
            plate_bboxes, _ = alpr.detect_plates(lower_crop)

            if plate_bboxes:
                pb = plate_bboxes[0]
                pb[1] += plate_region_y
                pb[3] += plate_region_y
                res = alpr.read_plate_from_frame(crop_img, pb)

                if res:
                    plate_text = res.plate_number
                    plate_valid = res.plate_valid

        except Exception as exc:
            print(f"  [ALPR ERR] {exc}")

    save_confirmed(vehicle_id, violation_type, crop_bytes, frame_id)
    save_review_image(vehicle_id, violation_type, crop_bytes, frame_id, "CONFIRMED")
    post_to_server(vehicle_id, violation_type, crop_bytes, frame_id, plate_text, plate_valid)
    print(f"  [★ CONFIRMED] {violation_type} id={vehicle_id} f={frame_id} plate={plate_text}")
    return True


def process_ai_item(item: tuple, helmet_model, no_helmet_ids: set, helmet_ids: set,
                     confirmed_set: set, confirmed_lock: threading.Lock,
                     vid_status: dict, status_lock: threading.Lock,
                     counters: dict, counters_lock: threading.Lock, alpr=None) -> None:
    vehicle_id, violation_type, crop_bytes, frame_id = item

    if violation_type != "HELMET_CHECK":
        return

    crop_img = cv2.imdecode(np.frombuffer(crop_bytes, np.uint8), cv2.IMREAD_COLOR)

    if helmet_model is not None:
        verdict = check_helmet_yolo(helmet_model, crop_img, no_helmet_ids, helmet_ids)

    elif USE_MOONDREAM_FALLBACK:
        verdict = check_helmet_moondream(crop_bytes)

    else:
        verdict = "SKIP"

    print(f"  [AI← ] f={frame_id} id={vehicle_id} HELMET → {verdict}")
    save_review_image(vehicle_id, "HELMET", crop_bytes, frame_id, verdict)

    with counters_lock:
        counters["reviewed"] += 1

    if verdict == "NO_HELMET":
        confirmed = confirm_violation(vehicle_id, "NO_HELMET", crop_bytes, frame_id,
                                       confirmed_set, confirmed_lock, counters, alpr)

        if confirmed:
            with status_lock:
                vid_status[vehicle_id] = "NO_HELMET"

    elif verdict == "HELMET":
        with status_lock:
            if vid_status.get(vehicle_id) != "NO_HELMET":
                vid_status[vehicle_id] = "HELMET_OK"


def queue_worker(q: queue.Queue, helmet_model, no_helmet_ids: set, helmet_ids: set,
                  confirmed_set: set, confirmed_lock: threading.Lock,
                  vid_status: dict, status_lock: threading.Lock,
                  counters: dict, counters_lock: threading.Lock, alpr=None) -> None:
    while True:
        item = q.get()

        if item is _STOP:
            q.task_done()
            break

        try:
            process_ai_item(item, helmet_model, no_helmet_ids, helmet_ids,
                             confirmed_set, confirmed_lock,
                             vid_status, status_lock, counters, counters_lock, alpr)

        except Exception as exc:
            print(f"  [WORKER ERR] {exc}")

        finally:
            q.task_done()


def draw_hud(frame: np.ndarray, frame_id: int, total: int, t_start: float,
             queued: int, confirmed: int, reviewed: int, scanned: int) -> None:
    elapsed = time.time() - t_start
    fps_r   = frame_id / max(elapsed, 0.001)
    pct     = round(frame_id / total * 100, 1) if total > 0 else 0
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 6), (600, 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, f"Frame {frame_id}/{total} ({pct}%)  FPS:{fps_r:.1f}  [Q] Quit",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1)

    cv2.putText(frame, f"Scanned:{scanned}  AI queue:{queued}  Confirmed:{confirmed}  Review:{reviewed}",
                (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 120), 1)


def draw_vehicle(display: np.ndarray, box: list, vid: int, lbl: str,
                  status: str, extra: str = "") -> None:
    x1, y1, x2, y2 = [int(v) for v in box]

    if status == "NO_HELMET":
        color, tag = COLORS["NO_HELMET"], f"{lbl}#{vid} X NO HELMET"

    elif status == "HELMET_OK":
        color, tag = COLORS["OK"], f"{lbl}#{vid} OK HELMET"

    elif status == "TRIPLE":
        color, tag = COLORS["TRIPLE_RIDING"], f"{lbl}#{vid} {extra}"

    elif status == "CHECKING":
        color, tag = COLORS["CHECKING"], f"{lbl}#{vid} {extra}"

    else:
        color, tag = COLORS["OK"], f"{lbl}#{vid}"

    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
    cv2.putText(display, tag, (x1 + 1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3)

    cv2.putText(display, tag, (x1, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)


def draw_scanned_vehicle(display: np.ndarray, box: list, vid: int, lbl: str) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(display, (x1, y1), (x2, y2), COLORS["SCAN"], 1)
    cv2.putText(display, f"{lbl}#{vid}", (x1, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["SCAN"], 1)


def main() -> None:
    try:
        import torch
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"

    except ImportError:
        device_arg = "cpu"

    print(f"[INFO] Compute device : {device_arg.upper()}")
    print(f"[INFO] Loading COCO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(device_arg)
    helmet_model, no_helmet_ids, helmet_ids = load_helmet_model(device_arg)
    video = Path(VIDEO_PATH)

    if not video.exists():
        for candidate in [Path("/mnt/user-data/uploads"), Path("."), Path("..")]:
            found = list(candidate.glob("*.mp4"))

            if found:
                video = found[0]
                print(f"[INFO] Auto-found video: {video}")
                break

        else:
            print(f"[ERROR] Video not found: {VIDEO_PATH}")
            sys.exit(1)

    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video}")
        sys.exit(1)

    fps_native   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_ms     = 1000.0 / fps_native
    print(f"[INFO] Video          : {W}x{H} @ {fps_native:.1f}fps  ({total_frames} frames)")
    print(f"[INFO] Frame skip     : every {PROCESS_EVERY_N} frames")
    print(f"[INFO] Min vehicle px : {MIN_VEH_PX}")
    print("[INFO] Checks         : NO_HELMET (bike/moto riders), TRIPLE_RIDING")
    print("[INFO] Press Q to quit\n" + "=" * 70)
    confirmed_set  : set            = set()
    confirmed_lock : threading.Lock = threading.Lock()
    vid_status  : dict           = {}
    status_lock : threading.Lock = threading.Lock()
    helmet_queued : set            = set()
    helmet_lock   : threading.Lock = threading.Lock()
    box_memory  : dict           = {}
    memory_lock : threading.Lock = threading.Lock()
    counters       : dict           = {"confirmed": 0, "reviewed": 0}
    counters_lock  : threading.Lock = threading.Lock()
    scanned_ids    : set            = set()
    ai_queue = queue.Queue()
    alpr = None

    if ALPRPipeline is not None:
        alpr = ALPRPipeline(plate_model_path="models/plate_yolov8.pt")

    worker   = threading.Thread(
        target=queue_worker,
        args=(ai_queue, helmet_model, no_helmet_ids, helmet_ids,
              confirmed_set, confirmed_lock,
              vid_status, status_lock,
              counters, counters_lock, alpr),
        daemon=True,
    )
    worker.start()
    frame_id = 0
    t_start  = time.time()

    while True:
        t_loop_start = time.time()
        ret, frame = cap.read()

        if not ret:
            print("\n[INFO] End of video.")
            break

        frame_id += 1
        h0, w0 = frame.shape[:2]

        if w0 > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / w0
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(h0 * scale)))

        display = frame.copy()

        if frame_id % PROCESS_EVERY_N != 0:
            with memory_lock:
                for vid, mem in box_memory.items():
                    if frame_id - mem["last_frame"] <= MEMORY_FRAMES:
                        with status_lock:
                            st = vid_status.get(vid)

                        draw_vehicle(display, mem["box"], vid, mem["lbl"], st or "")

            with counters_lock:
                confirmed, reviewed = counters["confirmed"], counters["reviewed"]

            draw_hud(display, frame_id, total_frames, t_start,
                     ai_queue.qsize(), confirmed, reviewed, len(scanned_ids))

            cv2.imshow("HawkEye — Traffic Violation Detector", display)
            elapsed_ms = (time.time() - t_loop_start) * 1000
            wait_ms    = max(1, int(frame_ms - elapsed_ms))

            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                print("[INFO] User quit.")
                break

            continue

        t_yolo = time.time()

        try:
            results = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                conf=CONF_THRESHOLD, classes=ALL_CLASSES,
                device=device_arg, verbose=False, imgsz=416, iou=0.45,
            )

        except Exception as exc:
            print(f"[YOLO ERR] {exc}")

            with memory_lock:
                for vid, mem in box_memory.items():
                    if frame_id - mem["last_frame"] <= MEMORY_FRAMES:
                        with status_lock:
                            st = vid_status.get(vid)

                        draw_vehicle(display, mem["box"], vid, mem["lbl"], st or "")

            continue

        yolo_ms = (time.time() - t_yolo) * 1000
        persons   : list = []
        bikes     : list = []
        motor_veh : list = []
        seen_ids  : set  = set()

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            res    = results[0]
            bboxes = res.boxes.xyxy.cpu().numpy()
            clses  = res.boxes.cls.cpu().numpy().astype(int)
            confs  = res.boxes.conf.cpu().numpy()
            ids    = (res.boxes.id.cpu().numpy().astype(int)
                      if res.boxes.id is not None else np.full(len(clses), -1, dtype=int))

            for i in range(len(clses)):
                box = bboxes[i].tolist()
                det = {"box": box, "conf": float(confs[i]), "id": int(ids[i]),
                       "cls": int(clses[i]), "bw": box[2] - box[0], "bh": box[3] - box[1]}

                c = det["cls"]
                if   c == CLS_PERSON:    persons.append(det)
                elif c in BIKE_CLASSES:  bikes.append(det)
                elif c in MOTOR_CLASSES: motor_veh.append(det)

        for p in persons:
            x1, y1, x2, y2 = [int(v) for v in p["box"]]
            cv2.rectangle(display, (x1, y1), (x2, y2), (190, 190, 0), 1)

        for veh in bikes:
            vid, box, cls = veh["id"], veh["box"], veh["cls"]
            lbl = CLS_LABEL.get(cls, "MOTO")
            bw, bh = veh["bw"], veh["bh"]

            if bw < MIN_VEH_PX or bh < MIN_VEH_PX:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(display, (x1, y1), (x2, y2), COLORS["SMALL"], 1)
                continue

            if vid < 0:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(display, (x1, y1), (x2, y2), COLORS["OK"], 1)
                continue

            seen_ids.add(vid)
            scanned_ids.add(vid)

            with memory_lock:
                box_memory[vid] = {"box": box, "lbl": lbl, "last_frame": frame_id}

            riders = [p for p in persons if boxes_overlap(p["box"], box)]
            count  = len(riders)

            if count >= 3:
                with confirmed_lock:
                    key = (vid, "TRIPLE_RIDING")

                    if key not in confirmed_set:
                        try:
                            cb = encode_jpeg(crop_region(frame, box))
                            confirm_violation(vid, "TRIPLE_RIDING", cb, frame_id, confirmed_set, confirmed_lock, counters, alpr)

                        except Exception as e:
                            print(f"  [ERR] triple: {e}")

                with status_lock:
                    vid_status[vid] = "TRIPLE"

                draw_vehicle(display, box, vid, lbl, "TRIPLE", extra=f"TRIPLE({count})")
                continue

            with helmet_lock:
                already_checked = vid in helmet_queued

            if not already_checked:
                with helmet_lock:
                    helmet_queued.add(vid)

                with status_lock:
                    vid_status[vid] = "CHECKING"

                try:
                    crop_bytes = encode_jpeg(crop_region(frame, box, top_extra_ratio=HEAD_PAD_RATIO))
                    ai_queue.put((vid, "HELMET_CHECK", crop_bytes, frame_id))
                    print(f"[DETECT] f={frame_id} {lbl}#{vid} → HELMET_CHECK [queued]")

                except Exception as e:
                    print(f"  [ERR] helmet crop: {e}")

                    with helmet_lock:
                        helmet_queued.discard(vid)

            with status_lock:
                cur_status = vid_status.get(vid)

            extra = "HELMET?" if cur_status == "CHECKING" else ""
            draw_vehicle(display, box, vid, lbl, cur_status or "CHECKING", extra=extra)

        for veh in motor_veh:
            vid, box, cls = veh["id"], veh["box"], veh["cls"]
            lbl = CLS_LABEL.get(cls, "VEH")
            bw, bh = veh["bw"], veh["bh"]

            if bw < MIN_VEH_PX or bh < MIN_VEH_PX:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(display, (x1, y1), (x2, y2), COLORS["SMALL"], 1)
                continue

            if vid < 0:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(display, (x1, y1), (x2, y2), COLORS["SCAN"], 1)
                continue

            seen_ids.add(vid)
            scanned_ids.add(vid)

            with memory_lock:
                box_memory[vid] = {"box": box, "lbl": lbl, "last_frame": frame_id}

            draw_scanned_vehicle(display, box, vid, lbl)

        with memory_lock:
            for vid, mem in box_memory.items():
                if vid in seen_ids:
                    continue

                if frame_id - mem["last_frame"] <= MEMORY_FRAMES:
                    with status_lock:
                        st = vid_status.get(vid)

                    x1, y1, x2, y2 = [int(v) for v in mem["box"]]
                    color = COLORS.get(st or "", COLORS["MEMORY"])
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)

        with counters_lock:
            confirmed, reviewed = counters["confirmed"], counters["reviewed"]

        draw_hud(display, frame_id, total_frames, t_start,
                 ai_queue.qsize(), confirmed, reviewed, len(scanned_ids))

        h_d, w_d = display.shape[:2]
        cv2.putText(display, f"YOLO:{yolo_ms:.0f}ms", (w_d - 130, h_d - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        cv2.imshow("HawkEye — Traffic Violation Detector", display)
        elapsed_ms = (time.time() - t_loop_start) * 1000
        wait_ms    = max(1, int(frame_ms - elapsed_ms))

        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("[INFO] User quit.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Video done. {frame_id} frames processed.")
    print(f"[INFO] Draining AI queue ({ai_queue.qsize()} remaining)…\n")
    ai_queue.put(_STOP)
    worker.join()
    confirmed_files = sorted(CONFIRMED_DIR.glob("*.jpg"))
    review_files    = sorted(REVIEW_DIR.glob("*.jpg"))
    by_type: dict = {}

    for f in confirmed_files:
        vt = f.name.split("_id")[0] if "_id" in f.name else f.stem
        by_type.setdefault(vt, []).append(f.name)

    review_counts: dict = {}

    for f in review_files:
        parts   = f.name.split("_")
        verdict = parts[1] if len(parts) >= 2 else "?"
        review_counts[verdict] = review_counts.get(verdict, 0) + 1

    print("\n" + "=" * 70)
    print("  HAWKEYE FINAL REPORT  v6.0  (Helmet + Triple-Riding MVP)")
    print("=" * 70)
    print(f"  Frames processed    : {frame_id} (every {PROCESS_EVERY_N})")
    print(f"  Vehicles scanned    : {len(scanned_ids)}")
    print(f"  Helmet candidates   : {len(helmet_queued)}")
    print(f"  Confirmed violations: {len(confirmed_files)}")
    print(f"  Review images       : {len(review_files)}")
    print("-" * 70)
    print("  CONFIRMED VIOLATIONS:")

    if by_type:
        for vt, files in sorted(by_type.items()):
            print(f"    {vt} ({len(files)}):")

            for fn in files:
                print(f"      → {fn}")

    else:
        print("    None confirmed.")

    print("-" * 70)
    print("  REVIEW BREAKDOWN:")

    for verdict, cnt in sorted(review_counts.items()):
        print(f"    {verdict:<20}: {cnt} images")

    print("=" * 70)
    print(f"  Confirmed → {CONFIRMED_DIR.resolve()}")
    print(f"  Review    → {REVIEW_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
