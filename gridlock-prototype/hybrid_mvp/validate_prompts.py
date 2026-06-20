"""
Quick validation: Test new descriptive prompts against an actual crop from the video.
"""
import requests, base64, cv2

# ── keyword sets (mirrors edge_client.py) ──────────────────────────────────────
_HELMET_WORN    = ["wearing a helmet","wears a helmet","has a helmet","helmet on",
                   "wearing helmet","with a helmet","protected by","safety helmet",
                   "head gear","headgear","crash helmet"]
_HELMET_MISSING = ["no helmet","without a helmet","without helmet","not wearing a helmet",
                   "bare head","bareheaded","bare-headed","no protective","unprotected head",
                   "no head","head is bare","head appears bare","exposed head",
                   "not protected","lacking a helmet"]
_BELT_WORN      = ["wearing a seatbelt","seatbelt visible","strap across","belt across",
                   "safety strap","seatbelt strap","wearing seatbelt","buckled",
                   "seat belt","can see a strap","strap is visible","diagonal strap"]
_BELT_MISSING   = ["no seatbelt","not wearing a seatbelt","without seatbelt","no seat belt",
                   "cannot see a seatbelt","no visible seatbelt","no safety belt",
                   "not buckled","without a seat belt","no strap","strap is not",
                   "belt is not","lacking a seatbelt","absence of a seatbelt"]

PROMPTS = {
    "HELMET_CHECK": (
        "Describe the person riding this motorcycle or bicycle. "
        "Focus specifically on their head — are they wearing any protective helmet or head gear? "
        "Describe exactly what you see on their head."
    ),
    "SEATBELT_CHECK": (
        "Describe the driver visible in this vehicle. "
        "Specifically describe whether you can see a seatbelt or safety strap "
        "across their chest and shoulder area. What do you observe?"
    ),
}

def parse_description(text, vtype):
    t = text.lower()
    if vtype == "HELMET_CHECK":
        worn    = any(k in t for k in _HELMET_WORN)
        missing = any(k in t for k in _HELMET_MISSING)
        if missing and not worn: return "YES → VIOLATION"
        if worn:                 return "NO  → compliant"
        return "SKIP → ambiguous"
    if vtype == "SEATBELT_CHECK":
        worn    = any(k in t for k in _BELT_WORN)
        missing = any(k in t for k in _BELT_MISSING)
        if missing and not worn: return "YES - VIOLATION"
        if worn:                 return "NO  - compliant"
        return "SKIP - ambiguous"
    return "SKIP"

def ask(crop, vtype):
    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.b64encode(buf.tobytes()).decode()
    resp = requests.post('http://localhost:11434/api/generate', json={
        'model': 'moondream:latest',
        'prompt': PROMPTS[vtype],
        'images': [b64],
        'stream': False,
        'options': {'temperature': 0.0},
    }, timeout=60)
    data  = resp.json()
    raw   = data.get('response', '')
    ec    = data.get('eval_count', 0)
    result = parse_description(raw, vtype)
    print(f"\n[{vtype}]")
    print(f"  eval_count : {ec}")
    print(f"  response   : {raw[:200]}")
    print(f"  verdict    : {result}")
    return raw

# Load video
cap = cv2.VideoCapture('test_traffic.mp4')
ret, frame = cap.read()
cap.release()

h, w = frame.shape[:2]
print(f"Frame: {w}x{h}\n")

# Test 1: motorcycle crop (top portion of frame where bikes tend to be)
bike_crop = frame[150:350, 400:600]
cv2.imwrite('validate_bike_crop.jpg', bike_crop)
print("Testing HELMET_CHECK with bike-region crop...")
ask(bike_crop, "HELMET_CHECK")

# Test 2: car crop (driver window — left side of car-sized region)
car_region = frame[100:280, 50:300]
driver_win = car_region[:, :int(car_region.shape[1]*0.45)]
cv2.imwrite('validate_car_crop.jpg', driver_win)
print("\nTesting SEATBELT_CHECK with car driver-window crop...")
ask(driver_win, "SEATBELT_CHECK")

print("\n\nDone! Check validate_bike_crop.jpg and validate_car_crop.jpg to see what was sent.")
