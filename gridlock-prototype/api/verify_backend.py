"""
api/verify_backend.py — Backend integration verification
  Runs without real YOLO weights by injecting a FakeDetector singleton.
  Tests:
    1. Route list matches spec
    2. GET /health → 200 {"status": "ok"}
    3. GET /violations/summary → 200 with correct shape
    4. POST /detect/image?run_alpr=false →
         - 200 OK
         - violations contains NO_HELMET and TRIPLE_RIDING
         - TRIPLE_RIDING has rider_count == 3
         - annotated_image_b64 is non-empty valid base64 JPEG
  Prints PASS/FAIL per test and exits 0 on all pass, 1 on any failure.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import base64
import traceback

import numpy as np
import cv2

# ─── 1. Inject fake detector BEFORE importing api.main ───────────────────────
#       so the lazy singleton is pre-populated and YOLO weights aren't needed

class _FakeDetector:
    """
    Mimics GridlockDetector.predict().
    Returns 1 motorcycle + 3 heavily-overlapping persons, 0 helmets.
    ViolationChecker (real) should flag NO_HELMET + TRIPLE_RIDING.
    """
    conf = 0.45

    def predict(self, frame, track=False):
        from src.detect import Detection
        return [
            Detection("motorcycle", 0.91, [100.0, 100.0, 400.0, 400.0], track_id=1),
            Detection("person",     0.88, [120.0, 120.0, 300.0, 380.0], track_id=2),
            Detection("person",     0.85, [150.0, 100.0, 320.0, 350.0], track_id=3),
            Detection("person",     0.87, [130.0, 150.0, 280.0, 400.0], track_id=4),
        ]

    def stream_video(self, *args, **kwargs):
        return iter([])  # no-op for video streaming


import api.main as api_module
api_module._detector = _FakeDetector()

from fastapi.testclient import TestClient
client = TestClient(api_module.app)

# ─── Test harness ─────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0

def test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        FAIL += 1
    except Exception as e:
        print(f"  [ERR]  {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# ─── Test 1: Route list ───────────────────────────────────────────────────────

print("\n[1] Route list check")

EXPECTED_ROUTES = {
    ("GET",       "/health"),
    ("POST",      "/detect/image"),
    ("POST",      "/alpr/read"),
    ("GET",       "/violations/log"),
    ("GET",       "/violations/summary"),
    ("WEBSOCKET", "/ws/detect/video"),
    ("POST",      "/dataset/fetch/roboflow"),
    ("POST",      "/dataset/fetch/kaggle"),
}

def check_routes():
    actual = set()
    for route in api_module.app.routes:
        if hasattr(route, "methods") and route.methods:
            for method in route.methods:
                actual.add((method.upper(), route.path))
        elif hasattr(route, "path") and "ws" in route.path.lower():
            actual.add(("WEBSOCKET", route.path))

    missing = EXPECTED_ROUTES - actual
    assert not missing, f"Missing routes: {missing}"
    print(f"       Registered routes:")
    for method, path in sorted(actual):
        print(f"         {method:9s} {path}")

test("All required routes registered", check_routes)


# ─── Test 2: GET /health ──────────────────────────────────────────────────────

print("\n[2] GET /health")

def check_health():
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data.get("status") == "ok", f"Expected status=ok, got {data}"

test("GET /health -> 200 {status: ok}", check_health)


# ─── Test 3: GET /violations/summary ─────────────────────────────────────────

print("\n[3] GET /violations/summary")

def check_summary():
    r = client.get("/violations/summary")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    required_keys = {"total", "NO_HELMET", "TRIPLE_RIDING", "SIGNAL_JUMP"}
    missing = required_keys - set(data.keys())
    assert not missing, f"Response missing keys: {missing}"
    for k in required_keys:
        assert isinstance(data[k], int), f"{k} should be int, got {type(data[k])}"

test("GET /violations/summary → 200 with correct shape", check_summary)


# ─── Test 4: POST /detect/image (core integration test) ──────────────────────

print("\n[4] POST /detect/image (mocked detector, real ViolationChecker)")

# Create a small synthetic 480×640 BGR test image
_img = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a rough motorcycle shape (orange rectangle)
cv2.rectangle(_img, (100, 100), (400, 400), (0, 165, 255), -1)
# Draw person shapes (green)
for x1, y1, x2, y2 in [(120, 120, 300, 380), (150, 100, 320, 350), (130, 150, 280, 400)]:
    cv2.rectangle(_img, (x1, y1), (x2, y2), (50, 205, 50), 2)

_, _jpg = cv2.imencode(".jpg", _img)
_jpg_bytes = _jpg.tobytes()

def check_detect_image():
    r = client.post(
        "/detect/image?run_alpr=false&conf_threshold=0.45&overlap_threshold=0.30&triple_threshold=3",
        files={"file": ("test.jpg", _jpg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()

    # Violations
    violations = data.get("violations", [])
    types = [v["violation_type"] for v in violations]

    assert "NO_HELMET" in types, (
        f"Expected NO_HELMET in violations, got: {types}\n"
        f"Detections: {data.get('detections')}"
    )
    assert "TRIPLE_RIDING" in types, (
        f"Expected TRIPLE_RIDING in violations, got: {types}"
    )

    # Triple riding rider count
    triple = next((v for v in violations if v["violation_type"] == "TRIPLE_RIDING"), None)
    assert triple is not None
    assert triple["rider_count"] == 3, (
        f"Expected rider_count=3, got {triple['rider_count']}"
    )

    # Annotated image base64
    b64 = data.get("annotated_image_b64", "")
    assert b64, "annotated_image_b64 is empty"
    decoded = base64.b64decode(b64)
    assert len(decoded) > 100, f"Decoded image too small ({len(decoded)} bytes)"
    # Verify it's a valid JPEG (starts with FF D8)
    assert decoded[:2] == b"\xff\xd8", "Decoded bytes don't look like a JPEG"

    print(f"       violations found: {types}")
    print(f"       rider_count: {triple['rider_count']}")
    print(f"       annotated_image_b64 length: {len(b64)} chars")

test(
    "POST /detect/image -> NO_HELMET + TRIPLE_RIDING(rider_count=3) + valid b64 JPEG",
    check_detect_image,
)


# ─── Test 5: POST /detect/image — bad image → 400 ────────────────────────────

print("\n[5] POST /detect/image — invalid image bytes")

def check_bad_image():
    r = client.post(
        "/detect/image",
        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert r.status_code == 400, f"Expected 400, got {r.status_code}: {r.text}"

test("POST /detect/image with garbage bytes → 400", check_bad_image)


# ─── Summary ──────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  Backend verification: {PASS} passed, {FAIL} failed")
print(f"{'='*55}\n")

if FAIL > 0:
    sys.exit(1)
